import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "api"
    / "orchestration"
    / "serverless_prompt.py"
)
_spec = importlib.util.spec_from_file_location("serverless_prompt_under_test", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_serverless_prompt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serverless_prompt)
compile_serverless_prompt = _serverless_prompt.compile_serverless_prompt


def _prepared_prompt():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}, "_meta": {"title": "source"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": ["3", 0]}},
        "3": {
            "class_type": "DistributedSeed",
            "inputs": {"seed": 40, "is_worker": True, "worker_id": "worker_0"},
        },
        "4": {
            "class_type": "DistributedCollector",
            "inputs": {"images": ["2", 0], "multi_job_id": "job/42", "worker_id": "worker_0", "is_worker": True},
        },
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
        "7": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}, "_meta": {"title": "Preview Image (auto-added)"}},
        "8": {"class_type": "UnrelatedNode", "inputs": {}},
    }


def _run(prompt):
    return compile_serverless_prompt(prompt)


def _load_prompt_transform():
    module_path = MODULE_PATH.with_name("prompt_transform.py")
    package_name = "serverless_prompt_transform_testpkg"
    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg
    for suffix in ("api", "api.orchestration", "utils"):
        pkg = types.ModuleType(f"{package_name}.{suffix}")
        pkg.__path__ = []
        sys.modules[f"{package_name}.{suffix}"] = pkg
    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    setattr(logging_module, "debug_log", lambda *_args, **_kwargs: None)
    setattr(logging_module, "log", lambda *_args, **_kwargs: None)
    sys.modules[f"{package_name}.utils.logging"] = logging_module
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api.orchestration.prompt_transform", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiles_prompt_prepared_by_existing_worker_helpers():
    prompt_transform = _load_prompt_transform()
    original = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": ["3", 0]}},
        "3": {"class_type": "DistributedSeed", "inputs": {"seed": 100}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["2", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }
    pruned = prompt_transform.prune_prompt_for_worker(original)
    prompt_index = prompt_transform.PromptIndex(pruned)
    prepared = prompt_transform.apply_participant_overrides(
        pruned,
        participant_id="worker-a",
        enabled_worker_ids=["worker-a"],
        job_id_map=prompt_transform.generate_job_id_map(prompt_index, "run"),
        master_url="http://master.invalid",
        delegate_master=False,
        prompt_index=prompt_index,
    )

    graph, manifest = _run(prepared)

    assert graph["2"]["inputs"]["seed"] == 101
    assert set(graph) == {"1", "2", "7"}
    assert graph["7"]["class_type"] == "SaveImage"
    assert manifest[0]["job_id"] == "run_4"
    assert manifest[0]["collector_id"] == "4"


def test_compiles_collector_input_to_flat_saveimage_and_prunes_downstream():
    original = _prepared_prompt()

    graph, manifest = _run(original)

    assert manifest == [
        {
            "collector_id": "4",
            "job_id": "job/42",
            "node_id": "9",
            "kind": "image",
            "prefix": graph["9"]["inputs"]["filename_prefix"],
        }
    ]
    prefix = manifest[0]["prefix"]
    assert graph["9"] == {
        "class_type": "SaveImage",
        "inputs": {"images": ["2", 0], "filename_prefix": prefix},
        "_meta": {"title": "Serverless output: collector 4 (image)"},
    }
    assert "/" not in prefix and "\\" not in prefix
    assert graph["2"]["inputs"]["seed"] == 41
    assert set(graph) == {"1", "2", "9"}
    assert all("Distributed" not in node["class_type"] for node in graph.values())
    assert original["3"]["inputs"]["seed"] == 40
    assert original["4"]["inputs"]["images"] == ["2", 0]


def test_manifest_and_prefix_are_repeatable_and_safe_for_untrusted_identity_parts():
    prompt = _prepared_prompt()
    prompt["4"]["inputs"].update(multi_job_id="../run/a", worker_id="worker/unsafe")

    first_graph, first_manifest = _run(prompt)
    second_graph, second_manifest = _run(prompt)

    assert first_manifest == second_manifest
    assert first_graph == second_graph
    prefix = first_manifest[0]["prefix"]
    assert prefix == first_graph[first_manifest[0]["node_id"]]["inputs"]["filename_prefix"]
    assert not any(char in prefix for char in ("/", "\\", ":"))
    assert first_manifest[0]["job_id"] == "../run/a"


def test_compiles_multiple_disconnected_collectors_in_stable_order():
    prompt = {
        "10": {"class_type": "LoadImage", "inputs": {}},
        "20": {"class_type": "LoadImage", "inputs": {}},
        "2": {"class_type": "DistributedCollector", "inputs": {"images": ["20", 0], "multi_job_id": "b", "worker_id": "worker_1"}},
        "1": {"class_type": "DistributedCollector", "inputs": {"images": ["10", 0], "multi_job_id": "a", "worker_id": "worker_0"}},
        "30": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }

    graph, manifest = _run(prompt)

    assert [item["collector_id"] for item in manifest] == ["1", "2"]
    assert [item["node_id"] for item in manifest] == ["31", "32"]
    assert graph["31"]["inputs"]["images"] == ["10", 0]
    assert graph["32"]["inputs"]["images"] == ["20", 0]
    assert set(graph) == {"10", "20", "31", "32"}


def test_rejects_audio_input_instead_of_emitting_audio_saver():
    prompt = {
        "1": {"class_type": "LoadAudio", "inputs": {}},
        "2": {"class_type": "DistributedCollector", "inputs": {"audio": ["1", 0]}},
    }

    with pytest.raises(ValueError, match="audio.*unsupported"):
        _run(prompt)


def test_rejects_collector_without_image_input():
    with pytest.raises(ValueError, match="collector.*image"):
        _run({"1": {"class_type": "DistributedCollector", "inputs": {}}})


@pytest.mark.parametrize(
    "class_type",
    ["UltimateSDUpscaleDistributed", "DistributedEmptyImage", "ImageBatchDivider", "AudioBatchDivider"],
)
def test_rejects_unsupported_distributed_and_batch_nodes(class_type):
    prompt = {
        "1": {"class_type": class_type, "inputs": {}},
        "2": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
        "3": {"class_type": "LoadImage", "inputs": {}},
    }

    with pytest.raises(ValueError, match="unsupported"):
        _run(prompt)


def test_rejects_collector_to_collector_dependency_through_processing_node():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {}},
        "2": {"class_type": "DistributedCollector", "inputs": {"images": ["1", 0]}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0]}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
    }

    with pytest.raises(ValueError, match="collector-to-collector"):
        _run(prompt)


@pytest.mark.parametrize(
    "link",
    [["missing", 0], ["1", -1], ["1", "0"], ["1", True]],
)
def test_rejects_malformed_api_links(link):
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {}},
        "2": {"class_type": "DistributedCollector", "inputs": {"images": link}},
    }

    with pytest.raises(ValueError, match="link"):
        _run(prompt)


def test_rejects_unexpected_output_slot_from_removed_distributed_node():
    prompt = {
        "1": {"class_type": "DistributedSeed", "inputs": {"seed": 1}},
        "2": {"class_type": "KSampler", "inputs": {"seed": ["1", 1]}},
        "3": {"class_type": "DistributedCollector", "inputs": {"images": ["2", 0]}},
    }

    with pytest.raises(ValueError, match="output slot"):
        _run(prompt)


def test_distributed_seed_matches_worker_offset_and_master_passthrough():
    prompt = {
        "1": {"class_type": "DistributedSeed", "inputs": {"seed": 10, "is_worker": True, "worker_id": "worker_1"}},
        "2": {"class_type": "DistributedSeed", "inputs": {"seed": 7, "is_worker": False}},
        "3": {"class_type": "KSampler", "inputs": {"seed": ["1", 0], "steps": ["2", 0]}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
    }

    graph, _manifest = _run(prompt)

    assert graph["3"]["inputs"] == {"seed": 12, "steps": 7}
    assert "1" not in graph and "2" not in graph


def test_distributed_value_uses_one_based_worker_values_and_type_coercion():
    prompt = {
        "1": {"class_type": "DistributedValue", "inputs": {"default_value": "3.9", "worker_values": json.dumps({"_type": "INT", "1": "8.7"}), "is_worker": True, "worker_id": "worker_0"}},
        "2": {"class_type": "DistributedValue", "inputs": {"default_value": "3.9", "worker_values": json.dumps({"_type": "INT", "2": ""}), "is_worker": True, "worker_id": "worker_1"}},
        "3": {"class_type": "KSampler", "inputs": {"steps": ["1", 0], "seed": ["2", 0]}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
    }

    graph, _manifest = _run(prompt)

    assert graph["3"]["inputs"] == {"steps": 8, "seed": 3}
    assert "1" not in graph and "2" not in graph


def test_resolves_stock_primitive_links_for_distributed_seed_and_value():
    prompt = {
        "1": {"class_type": "PrimitiveInt", "inputs": {"value": 25}},
        "2": {"class_type": "DistributedSeed", "inputs": {"seed": ["1", 0], "is_worker": True, "worker_id": "worker_0"}},
        "3": {"class_type": "PrimitiveString", "inputs": {"value": "4.8"}},
        "4": {"class_type": "PrimitiveString", "inputs": {"value": '{"_type":"FLOAT","1":"2.75"}'}},
        "5": {"class_type": "DistributedValue", "inputs": {"default_value": ["3", 0], "worker_values": ["4", 0], "is_worker": True, "worker_id": "worker_0"}},
        "6": {"class_type": "KSampler", "inputs": {"seed": ["2", 0], "cfg": ["5", 0]}},
        "7": {"class_type": "DistributedCollector", "inputs": {"images": ["6", 0]}},
    }

    graph, _manifest = _run(prompt)

    assert graph["6"]["inputs"] == {"seed": 26, "cfg": 2.75}
    assert not {"1", "2", "3", "4", "5"}.intersection(graph)


def test_rejects_dynamic_linked_seed_or_value_without_executing_it():
    prompt = {
        "1": {"class_type": "ArbitraryNode", "inputs": {}},
        "2": {"class_type": "DistributedSeed", "inputs": {"seed": ["1", 0], "is_worker": True, "worker_id": "worker_0"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": ["2", 0]}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
    }

    with pytest.raises(ValueError, match="literal or supported Primitive"):
        _run(prompt)


def test_primitive_resolution_detects_cycles():
    prompt = {
        "1": {"class_type": "PrimitiveInt", "inputs": {"value": ["2", 0]}},
        "2": {"class_type": "PrimitiveInt", "inputs": {"value": ["1", 0]}},
        "3": {"class_type": "DistributedSeed", "inputs": {"seed": ["1", 0]}},
        "4": {"class_type": "KSampler", "inputs": {"seed": ["3", 0]}},
        "5": {"class_type": "DistributedCollector", "inputs": {"images": ["4", 0]}},
    }

    with pytest.raises(ValueError, match="cycle"):
        _run(prompt)


def test_model_name_trivial_text_passthrough_is_inlined():
    prompt = {
        "1": {"class_type": "DistributedModelName", "inputs": {"text": "model.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model_name": ["1", 0]}},
        "3": {"class_type": "DistributedCollector", "inputs": {"images": ["2", 0]}},
    }

    graph, _manifest = _run(prompt)

    assert graph["2"]["inputs"]["model_name"] == "model.safetensors"
    assert "1" not in graph


def test_rejects_dynamic_linked_model_name_text():
    prompt = {
        "1": {"class_type": "ArbitraryNode", "inputs": {}},
        "2": {"class_type": "DistributedModelName", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"model_name": ["2", 0]}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
    }

    with pytest.raises(ValueError, match="DistributedModelName.*literal or supported Primitive"):
        _run(prompt)


def test_resolves_boolean_and_float_primitive_links():
    prompt = {
        "1": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "2": {"class_type": "PrimitiveFloat", "inputs": {"value": 1.25}},
        "3": {"class_type": "DistributedValue", "inputs": {"default_value": ["1", 0], "is_worker": False}},
        "4": {"class_type": "DistributedValue", "inputs": {"default_value": ["2", 0], "worker_values": '{"_type":"FLOAT","1":"2.5"}', "is_worker": True, "worker_id": "worker_0"}},
        "5": {"class_type": "KSampler", "inputs": {"enabled": ["3", 0], "cfg": ["4", 0]}},
        "6": {"class_type": "DistributedCollector", "inputs": {"images": ["5", 0]}},
    }

    graph, _manifest = _run(prompt)

    assert graph["5"]["inputs"] == {"enabled": True, "cfg": 2.5}
    assert not {"1", "2", "3", "4"}.intersection(graph)
