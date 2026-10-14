"""CPU-only graph/compiler tests; no ComfyUI package bootstrap is needed."""

import copy
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "serverless_media_prompt_testpkg"
# Real paths let sibling imports use production code, without running the
# extension's ComfyUI-dependent __init__.py or API registration side effects.
for suffix in ("", ".api", ".api.orchestration", ".serverless_bridge"):
    package = types.ModuleType(PACKAGE + suffix)
    package.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(ROOT)]
    sys.modules[package.__name__] = package

compiler = importlib.import_module(f"{PACKAGE}.api.orchestration.serverless_media_prompt")
legacy = importlib.import_module(f"{PACKAGE}.api.orchestration.serverless_prompt")
contract = importlib.import_module(f"{PACKAGE}.serverless_bridge.contract")
compile_media_prompt = compiler.compile_media_prompt
SOCKETS = {"images": "IMAGE", "audio": "AUDIO"}


def _node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def _collector(**sockets):
    return _node(
        "DistributedCollector", multi_job_id="job/42", worker_id="worker_0", **sockets
    )


def _prompt():
    return {
        "source:1": _node("LoadImage", image="example.png"),
        "collector:1": _collector(images=["source:1", 0]),
    }


def _run(prompt):
    return compile_media_prompt(prompt, "request:1")


@pytest.mark.parametrize("socket,kind", SOCKETS.items())
def test_one_export_for_each_typed_socket_with_exact_canonical_binding(socket, kind):
    prompt = {"source": _node("Source"), "collector": _collector(**{socket: ["source", 0]})}
    graph, bindings = _run(prompt)
    assert len(bindings) == 1
    binding = bindings[0]
    assert set(binding) == contract.BINDING_FIELDS
    assert binding == {
        "binding_id": binding["binding_id"],
        "collector_id": "collector",
        "job_id": "job/42",
        "participant_id": "worker_0",
        "node_id": binding["node_id"],
        "socket": socket,
        "kind": kind,
    }
    assert all(isinstance(value, str) for value in binding.values())
    assert contract.validate_bindings(bindings) == {binding["binding_id"]: binding}
    assert graph[binding["node_id"]] == {
        "class_type": "MediaExportV1",
        "inputs": {
            "request_id": "request:1",
            "binding_json": json.dumps(binding),
            socket: ["source", 0],
            **({"transfer_format": "png", "transfer_quality": 90} if kind == "IMAGE" else {}),
        },
    }
    assert set(graph) == {"source", binding["node_id"]}


def test_multiple_collectors_have_independent_image_audio_targets():
    prompt = {
        "shared": _node("MultiMediaSource"),
        "collector:z": _collector(audio=["shared", 1]),
        "collector:a": _collector(images=["shared", 0], audio=["shared", 1]),
    }
    prompt["collector:z"]["inputs"].update(multi_job_id="job:z", worker_id="participant:z")
    graph, bindings = _run(prompt)
    assert [(b["collector_id"], b["socket"]) for b in bindings] == [
        ("collector:a", "images"), ("collector:a", "audio"), ("collector:z", "audio"),
    ]
    assert len({b["binding_id"] for b in bindings}) == 3
    targets = {b["node_id"] for b in bindings}
    assert len(targets) == 3
    assert set(graph) == {"shared"} | targets
    for binding in bindings:
        export = graph[binding["node_id"]]
        inputs = prompt[binding["collector_id"]]["inputs"]
        assert export["inputs"][binding["socket"]] == inputs[binding["socket"]]
        assert binding["job_id"] == inputs["multi_job_id"]
        assert binding["participant_id"] == inputs["worker_id"]


def test_prunes_descendants_and_unrelated_outputs_preserving_required_output_utility():
    prompt = _prompt()
    prompt.update({
        "required-output": _node("RequiredOutputUtility", images=["source:1", 0]),
        "consumer": _node("ImageTransform", image=["required-output", 0]),
        "descendant": _node("ImageTransform", image=["collector:1", 0]),
        "after": _node("SaveImage", images=["descendant", 0]),
        "unused-output": _node("SaveImage", images=["source:1", 0]),
        "unrelated": _node("LoadAudio"),
    })
    prompt["required-output"]["_meta"] = {"title": "Required OUTPUT_NODE utility"}
    prompt["collector:1"]["inputs"].update(images=["consumer", 0], load_balance=True, pass_through=False)
    graph, bindings = _run(prompt)
    assert set(graph) == {"source:1", "required-output", "consumer", bindings[0]["node_id"]}
    assert graph["required-output"] == prompt["required-output"]
    assert not {"load_balance", "pass_through"} & graph[bindings[0]["node_id"]]["inputs"].keys()


def test_upstream_primitive_string_preserves_empty_and_link_looking_text():
    for text in ("", '["source:1", 0]', "hello\nworld"):
        prompt = {"text": _node("PrimitiveString", value=text),
                  "source": _node("TextToImage", text=["text", 0]),
                  "collector": _collector(images=["source", 0])}
        graph, bindings = _run(prompt)
        assert graph["text"] == prompt["text"]
        assert graph["source"]["inputs"]["text"] == ["text", 0]
        assert graph[bindings[0]["node_id"]]["inputs"]["images"] == ["source", 0]


@pytest.mark.parametrize("class_type,inputs", [
    ("DistributedSeed", {}),
    ("DistributedSeed", {"seed": 10, "is_worker": True, "worker_id": "worker_1"}),
    ("DistributedSeed", {"seed": 10, "is_worker": False, "worker_id": "worker_1"}),
    ("DistributedSeed", {"seed": 10, "is_worker": True, "worker_id": "bad"}),
    ("DistributedValue", {"default_value": "3.9", "worker_values": '{"_type":"INT","1":"8.7"}', "is_worker": True, "worker_id": "worker_0"}),
    ("DistributedValue", {"default_value": "3.9", "worker_values": '{"_type":"INT","2":""}', "is_worker": True, "worker_id": "worker_1"}),
    ("DistributedValue", {"default_value": "fallback", "worker_values": "malformed"}),
    ("DistributedValue", {"default_value": 1.25, "worker_values": '{"_type":"FLOAT","1":"bad"}', "is_worker": True, "worker_id": "0"}),
    ("DistributedModelName", {"text": "model.safetensors"}),
    ("DistributedModelName", {"text": {"name": "checkpoint"}}),
    ("DistributedModelName", {"text": ["a", "b", "c"]}),
])
def test_upstream_scalar_values_match_legacy_compiler(class_type, inputs):
    prompt = _prompt()
    prompt["scalar"] = _node(class_type, **inputs)
    prompt["source:1"]["inputs"]["parameter"] = ["scalar", 0]
    old_prompt = copy.deepcopy(prompt)
    old_prompt["collector:1"]["class_type"] = "DistributedCollector"
    expected, _ = legacy.compile_serverless_prompt(old_prompt)
    graph, _ = _run(prompt)
    assert graph["source:1"] == expected["source:1"]
    assert "scalar" not in graph


def test_imports_same_scalar_resolver_and_resolves_primitive_chains():
    assert compiler._make_scalar_resolver is legacy._make_scalar_resolver
    prompt = _prompt()
    prompt.update({
        "int": _node("PrimitiveInt", value=25),
        "int-alias": _node("PrimitiveInt", value=["int", 0]),
        "float": _node("PrimitiveFloat", value=1.25),
        "bool": _node("PrimitiveBoolean", value=True),
        "string": _node("PrimitiveString", value="model.safetensors"),
        "seed": _node("DistributedSeed", seed=["int-alias", 0], is_worker=True, worker_id="worker_0"),
        "value": _node("DistributedValue", default_value=["float", 0], worker_values='{"_type":"FLOAT","1":"2.75"}', is_worker=True, worker_id="worker_0"),
        "enabled": _node("DistributedValue", default_value=["bool", 0]),
        "model": _node("DistributedModelName", text=["string", 0]),
    })
    prompt["source:1"]["inputs"].update(seed=["seed", 0], cfg=["value", 0], enabled=["enabled", 0], model=["model", 0])
    graph, bindings = _run(prompt)
    assert graph["source:1"]["inputs"] == {"image": "example.png", "seed": 26, "cfg": 2.75, "enabled": True, "model": "model.safetensors"}
    assert set(graph) == {"source:1", bindings[0]["node_id"]}


def test_distributed_text_value_can_be_wrapped_by_primitive_string():
    prompt = {
        "value": _node("DistributedModelName", text="model.safetensors"),
        "text": _node("PrimitiveString", value=["value", 0]),
        "source": _node("TextToImage", text=["text", 0]),
        "collector": _collector(images=["source", 0]),
    }
    graph, bindings = _run(prompt)
    assert graph["text"]["inputs"]["value"] == "model.safetensors"
    assert graph["source"]["inputs"]["text"] == ["text", 0]
    assert graph[bindings[0]["node_id"]]["inputs"]["images"] == ["source", 0]
    assert "value" not in graph


@pytest.mark.parametrize("class_type,input_name", [
    ("DistributedSeed", "seed"), ("DistributedValue", "default_value"), ("DistributedModelName", "text"),
])
def test_dynamic_scalar_resolution_stays_fail_closed(class_type, input_name):
    prompt = _prompt()
    prompt["dynamic"] = _node("ArbitraryValue")
    prompt["scalar"] = _node(class_type, **{input_name: ["dynamic", 0]})
    prompt["source:1"]["inputs"]["parameter"] = ["scalar", 0]
    with pytest.raises(ValueError, match="literal or supported Primitive"):
        _run(prompt)


def test_namespaced_export_ids_avoid_existing_ids_even_on_pruned_nodes():
    prompt = _prompt()
    _, first = _run(prompt)
    original_id = first[0]["node_id"]
    assert ":" in original_id and not original_id.isdigit()
    prompt[original_id] = _node("ExistingUtility")
    _, second = _run(prompt)
    assert second[0]["node_id"] not in prompt
    assert second[0]["binding_id"] == first[0]["binding_id"]
    prompt[second[0]["node_id"]] = _node("AnotherUtility")
    graph, third = _run(prompt)
    assert third[0]["node_id"] not in prompt
    assert third == _run(dict(reversed(list(prompt.items()))))[1]
    assert set(graph) == {"source:1", third[0]["node_id"]}


def test_long_valid_identifiers_remain_exact_and_generated_ids_fit_contract():
    collector_id = "namespace/" + "c" * 190
    prompt = {"source:1": _node("Source"), collector_id: _collector(images=["source:1", 0])}
    prompt[collector_id]["inputs"].update(multi_job_id="j" * 200, worker_id="p" * 200)
    _, bindings = _run(prompt)
    assert bindings[0]["collector_id"] == collector_id
    assert bindings[0]["job_id"] == "j" * 200
    assert bindings[0]["participant_id"] == "p" * 200
    contract.validate_bindings(bindings)


def test_deterministic_independent_copies_and_request_identity():
    prompt = _prompt()
    prompt["source:1"]["_meta"] = {"nested": {"values": [1, 2, 3]}}
    original = copy.deepcopy(prompt)
    graph, bindings = _run(prompt)
    assert prompt == original
    assert (graph, bindings) == _run(prompt)
    assert (graph, bindings) == _run(dict(reversed(list(prompt.items()))))
    other_graph, other_bindings = compile_media_prompt(prompt, "request:2")
    assert other_bindings == bindings
    target = bindings[0]["node_id"]
    assert other_graph[target]["inputs"]["request_id"] == "request:2"
    graph["source:1"]["_meta"]["nested"]["values"].append(4)
    graph[target]["inputs"]["images"][0] = "changed"
    bindings[0]["collector_id"] = "changed"
    assert prompt == original
    assert json.loads(graph[target]["inputs"]["binding_json"])["collector_id"] == "collector:1"
    assert other_graph["source:1"] == original["source:1"]


@pytest.mark.parametrize("bad", [None, [], {}, {"x": None}, {"x": []}, {"x": {}},
    {"x": {"class_type": None}}, {"x": {"class_type": ""}},
    {"x": {"class_type": "Source", "inputs": []}},
    {"x": {"class_type": "Source", "inputs": None}},
    {"x": {"class_type": "Source", "inputs": {1: "literal"}}},
])
def test_rejects_malformed_prompt_or_missing_collectors(bad):
    with pytest.raises(ValueError):
        _run(bad)


@pytest.mark.parametrize("bad", [None, 1, True, "", " bad", "bad ", "bad\\id", "bad\nid", "x" * 201])
def test_rejects_malformed_ids_without_normalizing(bad):
    prompt = _prompt()
    prompt[bad] = _node("Source")
    with pytest.raises(ValueError):
        _run(prompt)


@pytest.mark.parametrize("bad", [None, 1, True, "", " bad", "bad ", "bad\\id", "x" * 201])
@pytest.mark.parametrize("field", ["multi_job_id", "worker_id", "request_id"])
def test_rejects_malformed_binding_and_request_identity(field, bad):
    prompt = _prompt()
    request_id = "request:1"
    if field == "request_id":
        request_id = bad
    else:
        prompt["collector:1"]["inputs"][field] = bad
    with pytest.raises(ValueError):
        compile_media_prompt(prompt, request_id)


@pytest.mark.parametrize("field", ["multi_job_id", "worker_id"])
def test_requires_prepared_literal_hidden_identity(field):
    prompt = _prompt()
    del prompt["collector:1"]["inputs"][field]
    with pytest.raises(ValueError):
        _run(prompt)
    prompt["collector:1"]["inputs"][field] = ["source:1", 0]
    with pytest.raises(ValueError):
        _run(prompt)


@pytest.mark.parametrize("socket", SOCKETS)
@pytest.mark.parametrize("value", [None, "literal", 5, True, [], ["source:1"], ["source:1", 0, 1], ("source:1", 0)])
def test_every_present_media_socket_requires_an_api_link(socket, value):
    prompt = _prompt()
    prompt["collector:1"]["inputs"][socket] = value
    with pytest.raises(ValueError, match="link"):
        _run(prompt)


def test_each_collector_requires_at_least_one_media_socket():
    prompt = _prompt()
    prompt["empty"] = _collector(load_balance=True, pass_through=False)
    with pytest.raises(ValueError, match="connected|linked"):
        _run(prompt)


@pytest.mark.parametrize("link", [["missing", 0], ["source:1", -1], ["source:1", 0.0],
    ["source:1", "0"], ["source:1", True], ["source:1", False], [1, 0],
    [True, 0], [None, 0], [[], 0], [{}, 0], [" source:1", 0],
])
@pytest.mark.parametrize("location", ["collector", "ordinary", "unrelated"])
def test_rejects_invalid_links_everywhere_before_pruning(location, link):
    prompt = _prompt()
    prompt["1"] = _node("Source")  # An integer/bool source must not be stringified to this ID.
    if location == "collector":
        prompt["collector:1"]["inputs"]["images"] = link
    elif location == "ordinary":
        prompt["source:1"]["inputs"]["parameter"] = link
    else:
        prompt["unrelated"] = _node("Utility", value=link)
    with pytest.raises(ValueError):
        _run(prompt)


@pytest.mark.parametrize("class_type", [
    "DistributedMediaCollector", "DistributedEmptyImage", "UltimateSDUpscaleDistributed",
    "DistributedImageListStart", "DistributedImageListEnd", "DistributedWorkerSink",
    "ImageBatchDivider", "AudioBatchDivider", "CustomDistributedNode", "custombatchdivider",
])
def test_rejects_unsupported_distributed_families_even_if_unused(class_type):
    prompt = _prompt()
    prompt["unsupported"] = _node(class_type)
    with pytest.raises(ValueError, match="unsupported"):
        _run(prompt)


@pytest.mark.parametrize("class_type", ["DistributedSeed", "DistributedValue", "DistributedModelName", "PrimitiveInt", "PrimitiveString"])
def test_rejects_invalid_known_scalar_output_slots(class_type):
    prompt = _prompt()
    prompt["scalar"] = _node(class_type)
    prompt["source:1"]["inputs"]["parameter"] = ["scalar", 1]
    with pytest.raises(ValueError, match="slot"):
        _run(prompt)


@pytest.mark.parametrize("socket", SOCKETS)
def test_rejects_direct_distributed_scalar_media_links(socket):
    prompt = {"scalar": _node("DistributedValue", default_value="text"), "collector": _collector(**{socket: ["scalar", 0]})}
    with pytest.raises(ValueError, match="scalar"):
        _run(prompt)


@pytest.mark.parametrize("indirect", [False, True])
def test_rejects_collector_chains_across_different_media(indirect):
    prompt = _prompt()
    source = "collector:1"
    if indirect:
        prompt["processing"] = _node("ImageToAudio", images=[source, 0])
        source = "processing"
    prompt["collector:2"] = _collector(audio=[source, 0])
    with pytest.raises(ValueError, match="collector-to-collector"):
        _run(prompt)


@pytest.mark.parametrize("mode", ["ordinary", "self", "unrelated", "primitive", "collector"])
def test_rejects_cycles_in_entire_prompt(mode):
    prompt = _prompt()
    if mode == "ordinary":
        prompt["source:1"]["inputs"]["other"] = ["other", 0]
        prompt["other"] = _node("Utility", value=["source:1", 0])
    elif mode == "self":
        prompt["source:1"]["inputs"]["other"] = ["source:1", 0]
    elif mode == "unrelated":
        prompt["unused"] = _node("Utility", value=["unused", 0])
    elif mode == "primitive":
        prompt["p1"] = _node("PrimitiveInt", value=["p2", 0])
        prompt["p2"] = _node("PrimitiveInt", value=["p1", 0])
    else:
        prompt["source:1"]["inputs"]["other"] = ["collector:1", 0]
    with pytest.raises(ValueError, match="cycle"):
        _run(prompt)


def test_deep_acyclic_graph_does_not_depend_on_python_recursion_limit():
    prompt = {"n:0": _node("Source")}
    for index in range(1, 1500):
        prompt[f"n:{index}"] = _node("Utility", value=[f"n:{index - 1}", 0])
    prompt["collector"] = _collector(images=["n:1499", 0])
    graph, bindings = _run(prompt)
    assert len(graph) == 1501
    assert graph[bindings[0]["node_id"]]["inputs"]["images"] == ["n:1499", 0]


def test_binding_count_respects_bridge_contract():
    prompt = {"source": _node("Source")}
    for index in range(contract.MAX_ARTIFACTS + 1):
        prompt[f"collector:{index}"] = _collector(images=["source", 0])
    with pytest.raises(ValueError, match="count"):
        _run(prompt)


def test_import_and_compile_do_not_import_comfy_or_tensor_libraries():
    script = '''
import builtins, importlib, pathlib, sys, types
root = pathlib.Path(sys.argv[1])
name = "isolated_media_compile"
for suffix in ("", ".api", ".api.orchestration", ".serverless_bridge"):
    package = types.ModuleType(name + suffix)
    package.__path__ = [str(root.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(root)]
    sys.modules[package.__name__] = package
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"comfy", "comfy_api", "nodes", "server", "torch", "numpy"}:
        raise AssertionError("unexpected runtime import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
module = importlib.import_module(name + ".api.orchestration.serverless_media_prompt")
prompt = {"s": {"class_type": "Source", "inputs": {}}, "c": {"class_type": "DistributedCollector", "inputs": {"images": ["s", 0], "multi_job_id": "job", "worker_id": "worker"}}}
graph, bindings = module.compile_media_prompt(prompt, "request")
assert graph[bindings[0]["node_id"]]["class_type"] == "MediaExportV1"
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
