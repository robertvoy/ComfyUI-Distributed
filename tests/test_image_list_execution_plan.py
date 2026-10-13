"""CPU-only execution-copy planning; no ComfyUI imports or prompt execution."""
import copy
import sys
import types

import pytest

from image_list.execution_plan import prepare_image_list_execution
from image_list.runtime import (
    CONTEXT_OUT,
    END_CLASS,
    IMAGE_OUT,
    START_CLASS,
    build_worker_prompt,
    validate_image_list_regions,
)


class OutputNode:
    OUTPUT_NODE = True


class SchemaOutputNode:
    OUTPUT_NODE = False  # V3's schema, not this fallback, is authoritative.

    @classmethod
    def GET_NODE_INFO_V1(cls):
        return {"output_node": True, "output": ("IMAGE",)}


class ModelLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")


@pytest.fixture
def registry(monkeypatch):
    classes = {
        END_CLASS: SchemaOutputNode,
        "SaveImage": OutputNode,
        "PreviewImage": OutputNode,
        "OutputUtility": OutputNode,
        "SchemaUtility": SchemaOutputNode,
        "ModelLoader": ModelLoader,
    }
    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(NODE_CLASS_MAPPINGS=classes))
    return classes


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def graph():
    return {
        "loader": node("ModelLoader"),
        "source": node("ImageSource"),
        "start": node(START_CLASS, images=["source", 0]),
        "process": node("FrameOperation", image=["start", IMAGE_OUT], model=["loader", 0]),
        "end": node(END_CLASS, images=["process", 0], context=["start", CONTEXT_OUT]),
        "save": node("SaveImage", images=["end", 0]),
    }


def test_discards_multistep_previews_but_preserves_other_output_roots(registry):
    prompt = graph()
    prompt.update({
        "debug-transform": node("DebugTransform", images=["start", IMAGE_OUT]),
        "debug-preview": node("PreviewImage", images=["debug-transform", 0]),
        "process-preview": node("PreviewImage", images=["process", 0]),
        "unrelated-source": node("ImageSource"),
        "unrelated-preview": node("PreviewImage", images=["unrelated-source", 0]),
        "upstream-preview": node("PreviewImage", images=["source", 0]),
        "post-transform": node("FrameOperation", images=["end", 0]),
        "post-preview": node("PreviewImage", images=["post-transform", 0]),
    })
    before = copy.deepcopy(prompt)
    with pytest.raises(ValueError, match="cross-boundary"):
        validate_image_list_regions(prompt)
    prepared, targets = prepare_image_list_execution(prompt)
    assert set(prepared) == set(prompt) - {"debug-transform", "debug-preview", "process-preview"}
    assert targets == ["end", "save", "unrelated-preview", "upstream-preview", "post-preview"]
    validate_image_list_regions(prepared, explicit_targets=True)
    assert prompt == before


@pytest.mark.parametrize("class_type", ["OutputUtility", "SchemaUtility"])
def test_required_output_utility_keeps_complete_dependency_chain(registry, class_type):
    prompt = graph()
    prompt.update({
        "utility-config": node("Constant", value=7),
        "utility-first": node(class_type, images=["process", 0], config=["utility-config", 0]),
        "utility-second": node(class_type, images=["utility-first", 0]),
        "result": node("FrameOperation", image=["utility-second", 0]),
        "discarded-preview": node("PreviewImage", images=["utility-first", 0]),
    })
    prompt["end"]["inputs"]["images"] = ["result", 0]
    before = copy.deepcopy(prompt)
    registered = dict(registry)
    prepared, targets = prepare_image_list_execution(prompt)
    assert set(prepared) == set(prompt) - {"discarded-preview"}
    assert targets == ["end", "save"]
    with pytest.raises(ValueError, match="Output node"):
        validate_image_list_regions(prepared)
    region = validate_image_list_regions(prepared, explicit_targets=True)[0]
    worker = build_worker_prompt(prepared, region, "token", "context", 0, "http://master.invalid")
    assert set(worker) == set(prepared) - {"source", "save"}
    assert worker["utility-second"]["inputs"] == {"images": ["utility-first", 0]}
    assert prompt == before
    assert registry == registered
    assert OutputNode.OUTPUT_NODE is True
    assert SchemaOutputNode.OUTPUT_NODE is False
    assert SchemaOutputNode.GET_NODE_INFO_V1()["output_node"] is True


@pytest.mark.parametrize("source", ["start", "process"])
@pytest.mark.parametrize("direct", [False, True])
def test_internal_bypass_into_post_end_computation_is_rejected(registry, source, direct):
    prompt = graph()
    prompt["post-process"] = node("Composite", images=["end", 0])
    if not direct:
        prompt["bypass"] = node("FrameOperation", image=[source, 0])
        source = "bypass"
    prompt["post-process"]["inputs"]["reference"] = [source, 0]
    prompt["save"]["inputs"]["images"] = ["post-process", 0]
    before = copy.deepcopy(prompt)
    with pytest.raises(ValueError, match="cross-boundary"):
        prepare_image_list_execution(prompt)
    assert prompt == before


def test_sequential_namespaced_regions_and_intermediate_output_are_preserved(registry):
    prompt = graph()
    prompt.update({
        "next-start": node(START_CLASS, images=["end", 0]),
        "next-process": node("SchemaUtility", images=["next-start", IMAGE_OUT], model=["loader", 0]),
        "next-end": node(END_CLASS, images=["next-process", 0], context=["next-start", CONTEXT_OUT]),
        "next-save": node("SaveImage", images=["next-end", 0]),
        "first-debug": node("PreviewImage", images=["process", 0]),
        "second-debug": node("PreviewImage", images=["next-process", 0]),
    })
    renamed = {key: f"group:7/{key}" for key in prompt}
    for spec in prompt.values():
        for name, value in spec["inputs"].items():
            if isinstance(value, list):
                spec["inputs"][name] = [renamed[value[0]], value[1]]
    prompt = {renamed[key]: spec for key, spec in prompt.items()}
    prepared, targets = prepare_image_list_execution(prompt)
    assert set(prepared) == set(prompt) - {renamed["first-debug"], renamed["second-debug"]}
    assert targets == [renamed[name] for name in ("end", "save", "next-end", "next-save")]
    regions = validate_image_list_regions(prepared, explicit_targets=True)
    assert {(r["start_id"], r["end_id"]) for r in regions} == {
        (renamed["start"], renamed["end"]), (renamed["next-start"], renamed["next-end"]),
    }
    second = next(r for r in regions if r["start_id"] == renamed["next-start"])
    worker = build_worker_prompt(prepared, second, "token", "context", 0, "http://master.invalid")
    assert set(worker) == {renamed[name] for name in ("loader", "next-start", "next-process", "next-end")}


def test_shared_loader_and_its_independent_output_root_are_preserved(registry):
    registry["ModelLoader"] = type("OutputLoader", (ModelLoader, OutputNode), {})
    prompt = graph()
    prompt["source"]["inputs"]["model"] = ["loader", 0]
    prompt["independent-save"] = node("SaveImage", model=["loader", 0])
    prepared, targets = prepare_image_list_execution(prompt)
    assert prepared == prompt
    assert targets == ["loader", "end", "save", "independent-save"]
    region = validate_image_list_regions(prepared, explicit_targets=True)[0]
    assert "loader" in region["node_ids"]
    assert "loader" not in region["branch_ids"]


@pytest.mark.parametrize("unsafe_class", [None, type("ListLoader", (), {
    "RETURN_TYPES": ("MODEL",), "OUTPUT_IS_LIST": (True,),
})])
def test_unknown_or_list_producing_shared_loader_still_fails_closed(registry, unsafe_class):
    if unsafe_class is None:
        registry.pop("ModelLoader")
    else:
        registry["ModelLoader"] = unsafe_class
    prompt = graph()
    prompt["source"]["inputs"]["model"] = ["loader", 0]
    prompt["debug"] = node("PreviewImage", images=["process", 0])
    with pytest.raises(ValueError, match="bypasses Start.*loader"):
        prepare_image_list_execution(prompt)


def test_upstream_per_item_bypass_is_not_hidden_by_preview_pruning(registry):
    prompt = graph()
    prompt["process"]["inputs"]["reference"] = ["source", 0]
    prompt["debug"] = node("PreviewImage", images=["process", 0])
    with pytest.raises(ValueError, match="bypasses Start.*source"):
        prepare_image_list_execution(prompt)


@pytest.mark.parametrize("nested", [False, True])
def test_overlapping_or_nested_regions_cannot_be_discarded_as_side_branches(registry, nested):
    prompt = graph()
    prompt["other-start"] = node(START_CLASS, images=["process" if nested else "source", 0])
    prompt["other-process"] = node("FrameOperation", image=["other-start", IMAGE_OUT], reference=["process", 0])
    prompt["other-end"] = node(END_CLASS, images=["other-process", 0], context=["other-start", CONTEXT_OUT])
    with pytest.raises(ValueError, match="Nested/ambiguous|cross-boundary|ambiguous processing"):
        prepare_image_list_execution(prompt)


@pytest.mark.parametrize("invalid", ["missing-end", "wrong-context", "missing-images", "unrelated-images"])
def test_pairing_and_result_validation_cannot_be_pruned_away(registry, invalid):
    prompt = graph()
    if invalid == "missing-end":
        del prompt["end"]
    elif invalid == "wrong-context":
        prompt["end"]["inputs"]["context"] = ["source", 0]
    elif invalid == "missing-images":
        del prompt["end"]["inputs"]["images"]
    else:
        prompt["end"]["inputs"]["images"] = ["source", 0]
    before = copy.deepcopy(prompt)
    with pytest.raises(ValueError):
        prepare_image_list_execution(prompt)
    assert prompt == before


def test_execution_copy_is_deep_and_repeated_preparation_is_stable(registry):
    prompt = graph()
    prompt["process"]["_meta"] = {"title": "Original", "tags": ["saved"]}
    before = copy.deepcopy(prompt)
    prepared, targets = prepare_image_list_execution(prompt)
    prepared_again, targets_again = prepare_image_list_execution(prepared)
    assert prepared_again == prepared
    assert targets_again == targets
    prepared["start"]["inputs"]["images"][0] = "changed"
    prepared["process"]["_meta"]["tags"].append("execution-only")
    assert prompt == before
    assert prepared_again == before


def test_v3_false_output_flag_overrides_legacy_true(registry):
    registry["NotOutput"] = type("NotOutput", (), {
        "OUTPUT_NODE": True,
        "GET_NODE_INFO_V1": classmethod(lambda cls: {"output_node": False}),
    })
    prompt = graph()
    prompt["not-output"] = node("NotOutput", images=["end", 0])
    prepared, targets = prepare_image_list_execution(prompt)
    assert prepared == prompt
    assert targets == ["end", "save"]
