"""Pure integration of participant identity, typed compilation and delegate pruning."""
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
contract = importlib.import_module(PACKAGE + ".serverless_bridge.contract")


def prompt():
    nodes = {
        "source": {"class_type": "MediaSource", "inputs": {}},
        "collector": {"class_type": "DistributedMediaCollector", "inputs": {
            socket: ["source", index] for index, socket in enumerate(contract.SOCKET_KINDS)}},
    }
    for index, socket in enumerate(contract.SOCKET_KINDS):
        nodes["sink:" + socket] = {"class_type": "Capture", "inputs": {"value": ["collector", index]}}
    return nodes


def prepare(graph, participant, delegate):
    index = pt.PromptIndex(graph)
    jobs = pt.generate_job_id_map(index, "run")
    assert jobs == {"collector": "run_collector"}
    return pt.apply_participant_overrides(index.copy_prompt(), participant,
        ["serverless:pool:1", "serverless:pool:2"], jobs, "http://master", delegate, index)


def test_typed_worker_preparation_compiles_all_bindings_without_source_mutation():
    graph = prompt()
    before = copy.deepcopy(graph)
    worker = prepare(graph, "serverless:pool:2", True)
    compiled, bindings = compiler.compile_media_prompt(worker, "request:2")
    assert len(bindings) == 5
    assert {row["participant_id"] for row in bindings} == {"serverless:pool:2"}
    assert {row["job_id"] for row in bindings} == {"run_collector"}
    assert {row["socket"] for row in bindings} == set(contract.SOCKET_KINDS)
    assert set(compiled) == {"source"} | {row["node_id"] for row in bindings}
    assert graph == before


def test_typed_delegate_master_keeps_outputs_without_sources_or_placeholders():
    graph = prompt()
    prepared = prepare(graph, "master", True)
    before = copy.deepcopy(prepared)
    delegated = pt.prepare_delegate_master_prompt(prepared, ["collector"])
    assert set(delegated) == {"collector"} | {"sink:" + s for s in contract.SOCKET_KINDS}
    inputs = delegated["collector"]["inputs"]
    assert not set(contract.SOCKET_KINDS) & set(inputs)
    assert json.loads(inputs["expected_media"]) == list(contract.SOCKET_KINDS)
    assert inputs["delegate_only"] is True
    assert prepared == before


def test_text_source_is_not_retained_as_lightweight_delegate_scalar(monkeypatch):
    graph = {
        "text": {"class_type": "PrimitiveString", "inputs": {"value": "local text"}},
        "collector": {"class_type": "DistributedMediaCollector", "inputs": {"texts": ["text", 0]}},
        "sink": {"class_type": "Capture", "inputs": {"value": ["collector", 3]}},
    }
    monkeypatch.setattr(pt, "_delegate_master_input_is_safe_scalar", lambda *args: True)
    monkeypatch.setattr(pt, "_is_delegate_master_retained_upstream_node", lambda *args: True)
    delegated = pt.prepare_delegate_master_prompt(prepare(graph, "master", True), ["collector"])
    assert "text" not in delegated
    assert "texts" not in delegated["collector"]["inputs"]
    assert json.loads(delegated["collector"]["inputs"]["expected_media"]) == ["texts"]


@pytest.mark.parametrize("socket", contract.SOCKET_KINDS)
def test_participating_master_retains_exact_media_links(socket):
    graph = prompt()
    original = copy.deepcopy(graph)
    master = prepare(graph, "master", False)
    assert master["collector"]["inputs"][socket] == graph["collector"]["inputs"][socket]
    assert master["collector"]["inputs"]["delegate_only"] is False
    assert "worker_id" not in master["collector"]["inputs"]
    assert graph == original
