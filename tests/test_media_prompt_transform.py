"""Participant identity, typed compilation and legacy Collector pruning."""
import copy
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "media_transform_testpkg"
for suffix in ("", ".api", ".api.orchestration", ".utils", ".serverless_bridge"):
    module = types.ModuleType(PACKAGE + suffix)
    module.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(ROOT)]
    sys.modules[module.__name__] = module
pt = importlib.import_module(PACKAGE + ".api.orchestration.prompt_transform")
compiler = importlib.import_module(PACKAGE + ".api.orchestration.serverless_media_prompt")
SOCKETS = ("images", "audio")


def prompt(sockets=SOCKETS):
    nodes = {
        "source": {"class_type": "MediaSource", "inputs": {}},
        "collector": {"class_type": "DistributedCollector", "inputs": {
            socket: ["source", SOCKETS.index(socket)] for socket in sockets}},
    }
    for socket in sockets:
        nodes["sink:" + socket] = {"class_type": "Capture", "inputs": {"value": ["collector", SOCKETS.index(socket)]}}
    return nodes


def prepare(graph, participant, delegate):
    index = pt.PromptIndex(graph)
    jobs = pt.generate_job_id_map(index, "run")
    assert jobs == {"collector": "run_collector"}
    return pt.apply_participant_overrides(index.copy_prompt(), participant,
        ["serverless:pool:1", "serverless:pool:2"], jobs, "http://master", delegate, index)


def test_typed_worker_preparation_compiles_both_bindings_without_source_mutation():
    graph = prompt()
    before = copy.deepcopy(graph)
    worker = prepare(graph, "serverless:pool:2", True)
    compiled, bindings = compiler.compile_media_prompt(worker, "request:2")
    assert len(bindings) == 2
    assert {row["participant_id"] for row in bindings} == {"serverless:pool:2"}
    assert {row["job_id"] for row in bindings} == {"run_collector"}
    assert {row["socket"] for row in bindings} == set(SOCKETS)
    assert set(compiled) == {"source"} | {row["node_id"] for row in bindings}
    assert graph == before


@pytest.mark.parametrize("sockets", [("images",), ("audio",), SOCKETS])
def test_delegate_master_keeps_collector_socket_contract_and_legacy_placeholder(sockets):
    graph = prompt(sockets)
    prepared = prepare(graph, "master", True)
    before = copy.deepcopy(prepared)
    delegated = pt.prepare_delegate_master_prompt(prepared, ["collector"])
    assert "source" not in delegated
    assert {"collector"} | {"sink:" + s for s in sockets} <= set(delegated)
    inputs = delegated["collector"]["inputs"]
    assert "audio" not in inputs and "expected_media" not in inputs
    placeholders = [key for key, node in delegated.items() if node['class_type'] == 'DistributedEmptyImage']
    if "images" in sockets:
        assert len(placeholders) == 1
        assert inputs['images'] == [placeholders[0], 0]
    else:
        assert not placeholders and 'images' not in inputs
    assert inputs["delegate_only"] is True
    assert json.loads(inputs["enabled_worker_ids"]) == ["serverless:pool:1", "serverless:pool:2"]
    assert prepared == before


@pytest.mark.parametrize("socket", SOCKETS)
def test_participating_master_retains_exact_media_links(socket):
    graph = prompt()
    original = copy.deepcopy(graph)
    master = prepare(graph, "master", False)
    assert master["collector"]["inputs"][socket] == graph["collector"]["inputs"][socket]
    assert master["collector"]["inputs"]["delegate_only"] is False
    assert "worker_id" not in master["collector"]["inputs"]
    assert graph == original
