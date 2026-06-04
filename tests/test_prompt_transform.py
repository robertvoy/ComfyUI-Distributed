import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


def _load_prompt_transform_module():
    module_path = Path(__file__).resolve().parents[1] / "api" / "orchestration" / "prompt_transform.py"
    package_name = "dist_pt_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    api_pkg = types.ModuleType(f"{package_name}.api")
    api_pkg.__path__ = []
    sys.modules[f"{package_name}.api"] = api_pkg

    orch_pkg = types.ModuleType(f"{package_name}.api.orchestration")
    orch_pkg.__path__ = []
    sys.modules[f"{package_name}.api.orchestration"] = orch_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api.orchestration.prompt_transform",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


pt = _load_prompt_transform_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_prompt():
    """1 → 2 → 3 → 4(DistributedCollector) → 5(SaveImage)"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0]}},
        "4": {"class_type": "DistributedCollector", "inputs": {"images": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }


def _collector_only_prompt():
    """1(Checkpoint) → 2(DistributedCollector) [no downstream from 2]"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "DistributedCollector", "inputs": {"images": ["1", 0]}},
    }


def _delegate_prompt():
    """1 → 2 → 3(DistributedCollector) → 4(SaveImage)"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "DistributedCollector", "inputs": {"images": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }


def _apply(prompt, participant_id, enabled_worker_ids=None, delegate_master=False):
    if enabled_worker_ids is None:
        enabled_worker_ids = ["worker-a", "worker-b"]
    idx = pt.PromptIndex(prompt)
    job_id_map = pt.generate_job_id_map(idx, "run")
    return pt.apply_participant_overrides(
        prompt,
        participant_id=participant_id,
        enabled_worker_ids=enabled_worker_ids,
        job_id_map=job_id_map,
        master_url="http://master.example.com",
        delegate_master=delegate_master,
        prompt_index=idx,
    )


# ---------------------------------------------------------------------------
# PromptIndex
# ---------------------------------------------------------------------------

class PromptIndexTests(unittest.TestCase):
    def test_nodes_by_class_groups_correctly(self):
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "DistributedCollector", "inputs": {}},
            "3": {"class_type": "DistributedCollector", "inputs": {}},
        }
        idx = pt.PromptIndex(prompt)
        self.assertCountEqual(idx.nodes_for_class("DistributedCollector"), ["2", "3"])
        self.assertEqual(idx.nodes_for_class("CheckpointLoaderSimple"), ["1"])

    def test_nodes_for_class_unknown_returns_empty(self):
        idx = pt.PromptIndex({"1": {"class_type": "KSampler", "inputs": {}}})
        self.assertEqual(idx.nodes_for_class("Nonexistent"), [])

    def test_nodes_without_class_type_are_indexed_under_none(self):
        prompt = {"1": {"inputs": {}}}
        idx = pt.PromptIndex(prompt)
        # Should not raise; nodes_for_class with None key or missing class_type
        self.assertEqual(idx.nodes_for_class("KSampler"), [])

    def test_copy_prompt_is_a_deep_copy(self):
        prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
        idx = pt.PromptIndex(prompt)
        copy = idx.copy_prompt()
        copy["1"]["inputs"]["seed"] = 999
        self.assertEqual(prompt["1"]["inputs"]["seed"], 42)

    def test_has_upstream_direct_connection(self):
        """Node 4 reads directly from node 3 (KSampler)."""
        idx = pt.PromptIndex(_linear_prompt())
        self.assertTrue(idx.has_upstream("4", "KSampler"))

    def test_has_upstream_transitive_connection(self):
        """Node 4 → 3 → 2 → 1 (CheckpointLoaderSimple)."""
        idx = pt.PromptIndex(_linear_prompt())
        self.assertTrue(idx.has_upstream("4", "CheckpointLoaderSimple"))

    def test_has_upstream_returns_false_when_no_path(self):
        idx = pt.PromptIndex(_linear_prompt())
        # CheckpointLoaderSimple has no upstream nodes
        self.assertFalse(idx.has_upstream("1", "DistributedCollector"))

    def test_has_upstream_result_is_cached(self):
        idx = pt.PromptIndex(_linear_prompt())
        r1 = idx.has_upstream("4", "KSampler")
        r2 = idx.has_upstream("4", "KSampler")
        self.assertEqual(r1, r2)
        self.assertIn(("4", "KSampler"), idx._upstream_cache)

    def test_has_upstream_does_not_infinite_loop_on_cycle(self):
        """Cyclic references in inputs should not cause infinite recursion."""
        prompt = {
            "1": {"class_type": "A", "inputs": {"x": ["2", 0]}},
            "2": {"class_type": "B", "inputs": {"x": ["1", 0]}},
        }
        idx = pt.PromptIndex(prompt)
        # Should terminate without error
        result = idx.has_upstream("1", "NonExistent")
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# find_nodes_by_class
# ---------------------------------------------------------------------------

class FindNodesByClassTests(unittest.TestCase):
    def test_finds_matching_nodes(self):
        prompt = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "DistributedCollector", "inputs": {}},
        }
        result = pt.find_nodes_by_class(prompt, "KSampler")
        self.assertEqual(result, ["1"])

    def test_returns_empty_when_no_match(self):
        prompt = {"1": {"class_type": "KSampler", "inputs": {}}}
        self.assertEqual(pt.find_nodes_by_class(prompt, "DistributedCollector"), [])

    def test_skips_non_dict_nodes(self):
        prompt = {"1": "not a dict", "2": {"class_type": "KSampler", "inputs": {}}}
        result = pt.find_nodes_by_class(prompt, "KSampler")
        self.assertEqual(result, ["2"])


# ---------------------------------------------------------------------------
# prune_prompt_for_worker
# ---------------------------------------------------------------------------

class PrunePromptForWorkerTests(unittest.TestCase):
    def test_no_distributed_nodes_returns_prompt_unchanged(self):
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        }
        result = pt.prune_prompt_for_worker(prompt)
        self.assertCountEqual(result.keys(), ["1", "2"])

    def test_keeps_collector_and_upstream(self):
        prompt = _linear_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        for node_id in ("1", "2", "3", "4"):
            self.assertIn(node_id, result)

    def test_removes_downstream_of_collector(self):
        prompt = _linear_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        self.assertNotIn("5", result)

    def test_injects_preview_image_when_downstream_exists(self):
        prompt = _linear_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        preview_nodes = [n for n in result.values() if n.get("class_type") == "PreviewImage"]
        self.assertEqual(len(preview_nodes), 1)
        self.assertEqual(preview_nodes[0]["inputs"]["images"], ["4", 0])

    def test_no_preview_image_when_no_downstream(self):
        result = pt.prune_prompt_for_worker(_collector_only_prompt())
        preview_nodes = [n for n in result.values() if n.get("class_type") == "PreviewImage"]
        self.assertEqual(len(preview_nodes), 0)

    def test_unrelated_nodes_are_pruned(self):
        prompt = {
            "1": {"class_type": "DistributedCollector", "inputs": {}},
            "2": {"class_type": "UnrelatedNode", "inputs": {}},  # no connection to 1
        }
        result = pt.prune_prompt_for_worker(prompt)
        self.assertIn("1", result)
        self.assertNotIn("2", result)

    def test_result_is_a_copy_not_same_object(self):
        prompt = _linear_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        # Mutating the result should not affect the original
        original_keys = set(prompt.keys())
        result["NEW"] = {"class_type": "Test", "inputs": {}}
        self.assertEqual(set(prompt.keys()), original_keys)

    def test_upscale_node_is_treated_as_distributed(self):
        prompt = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "UltimateSDUpscaleDistributed", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        }
        result = pt.prune_prompt_for_worker(prompt)
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertNotIn("3", result)


# ---------------------------------------------------------------------------
# prepare_delegate_master_prompt
# ---------------------------------------------------------------------------

class PrepareDelegateMasterPromptTests(unittest.TestCase):
    def test_keeps_collector_and_downstream(self):
        prompt = _delegate_prompt()
        result = pt.prepare_delegate_master_prompt(prompt, ["3"])
        self.assertIn("3", result)
        self.assertIn("4", result)
        self.assertNotIn("1", result)
        self.assertNotIn("2", result)

    def test_removes_dangling_upstream_refs(self):
        """Collector must not retain dangling refs to pruned upstream nodes."""
        prompt = _delegate_prompt()
        result = pt.prepare_delegate_master_prompt(prompt, ["3"])
        collector_inputs = result["3"].get("inputs", {})
        # Original "images" pointed at node 2, which is pruned.
        # It should now point at a newly injected placeholder node.
        self.assertIn("images", collector_inputs)
        source_id = str(collector_inputs["images"][0])
        self.assertNotEqual(source_id, "2")
        self.assertIn(source_id, result)
        self.assertEqual(result[source_id].get("class_type"), "DistributedEmptyImage")

    def test_injects_empty_image_placeholder(self):
        prompt = _delegate_prompt()
        result = pt.prepare_delegate_master_prompt(prompt, ["3"])
        empty_nodes = [(nid, n) for nid, n in result.items() if n.get("class_type") == "DistributedEmptyImage"]
        self.assertEqual(len(empty_nodes), 1)
        placeholder_id = empty_nodes[0][0]
        self.assertEqual(result["3"]["inputs"]["images"], [placeholder_id, 0])

    def test_one_placeholder_per_collector(self):
        """Two collectors → two placeholders."""
        prompt = {
            "1": {"class_type": "DistributedCollector", "inputs": {}},
            "2": {"class_type": "DistributedCollector", "inputs": {}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        }
        result = pt.prepare_delegate_master_prompt(prompt, ["1", "2"])
        empty_nodes = [n for n in result.values() if n.get("class_type") == "DistributedEmptyImage"]
        self.assertEqual(len(empty_nodes), 2)

    def test_preserves_primitive_string_for_downstream_required_input(self):
        """Delegate-only master keeps primitive inputs needed by SaveImage."""
        save_image = type(
            "SaveImage",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "filename_prefix": ("STRING",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {"SaveImage": save_image}
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "15": {"class_type": "PrimitiveString", "inputs": {"value": "test/input_bug"}},
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": ["15", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        self.assertIn("15", result)
        self.assertEqual(result["15"], prompt["15"])
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], ["15", 0])

    def test_does_not_preserve_non_primitive_upstream_for_collector(self):
        prompt = _delegate_prompt()
        result = pt.prepare_delegate_master_prompt(prompt, ["3"])

        self.assertNotIn("2", result)
        self.assertNotEqual(result["3"]["inputs"]["images"], ["2", 0])

    def test_preserves_load_image_for_switch_alternate_required_input(self):
        """Delegate-only master keeps LoadImage inputs needed by switches."""
        prompt = {
            "848": {"class_type": "LoadImage", "inputs": {"image": "input_bug2_00003_.png"}},
            "862": {"class_type": "VAEDecode", "inputs": {}},
            "854": {"class_type": "DistributedCollector", "inputs": {"images": ["862", 0]}},
            "850": {
                "class_type": "ComfySwitchNode",
                "inputs": {
                    "on_false": ["848", 0],
                    "on_true": ["854", 0],
                },
            },
            "851": {"class_type": "PreviewImage", "inputs": {"images": ["850", 0]}},
        }

        result = pt.prepare_delegate_master_prompt(prompt, ["854"])

        self.assertIn("848", result)
        self.assertEqual(result["850"]["inputs"]["on_false"], ["848", 0])
        self.assertEqual(result["850"]["inputs"]["on_true"], ["854", 0])
        self.assertNotIn("862", result)

    def test_preserves_registered_string_utility_subgraph_for_downstream_required_input(self):
        """Delegate-only master keeps scalar utility chains used by SaveImage."""
        string_concat = type(
            "StringConcatenate",
            (),
            {
                "RETURN_TYPES": ("STRING",),
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "string_a": ("STRING",),
                            "string_b": ("STRING",),
                        }
                    }
                ),
            },
        )
        save_image = type(
            "SaveImage",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "filename_prefix": ("STRING",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {
            "SaveImage": save_image,
            "StringConcatenate": string_concat,
        }
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "15": {"class_type": "PrimitiveString", "inputs": {"value": "test"}},
                "16": {"class_type": "PrimitiveString", "inputs": {"value": "input_bug3"}},
                "17": {
                    "class_type": "StringConcatenate",
                    "inputs": {
                        "string_a": ["15", 0],
                        "string_b": ["16", 0],
                    },
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": ["17", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        self.assertIn("15", result)
        self.assertIn("16", result)
        self.assertIn("17", result)
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], ["17", 0])
        self.assertEqual(result["17"]["inputs"]["string_a"], ["15", 0])
        self.assertEqual(result["17"]["inputs"]["string_b"], ["16", 0])

    def test_preserves_registered_multi_string_join_subgraph_for_downstream_required_input(self):
        """Delegate-only master keeps multi-input scalar utility chains."""
        join_string_multi = type(
            "JoinStringMulti",
            (),
            {
                "RETURN_TYPES": ("STRING",),
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {"string_1": ("STRING",)},
                        "optional": {
                            "string_2": ("STRING",),
                            "string_3": ("STRING",),
                        },
                    }
                ),
            },
        )
        save_image = type(
            "SaveImage",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "filename_prefix": ("STRING",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {
            "JoinStringMulti": join_string_multi,
            "SaveImage": save_image,
        }
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "15": {"class_type": "PrimitiveString", "inputs": {"value": "test"}},
                "16": {"class_type": "PrimitiveString", "inputs": {"value": "input"}},
                "17": {"class_type": "PrimitiveString", "inputs": {"value": "bug4"}},
                "18": {
                    "class_type": "JoinStringMulti",
                    "inputs": {
                        "string_1": ["15", 0],
                        "string_2": ["16", 0],
                        "string_3": ["17", 0],
                    },
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": ["18", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        self.assertIn("15", result)
        self.assertIn("16", result)
        self.assertIn("17", result)
        self.assertIn("18", result)
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], ["18", 0])

    def test_does_not_preserve_scalar_utility_with_heavy_upstream_dependency(self):
        """Scalar utility nodes are retained only when their full input branch is safe."""
        string_concat = type(
            "StringConcatenate",
            (),
            {
                "RETURN_TYPES": ("STRING",),
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "string_a": ("STRING",),
                            "string_b": ("STRING",),
                        }
                    }
                ),
            },
        )
        save_image = type(
            "SaveImage",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "filename_prefix": ("STRING",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {
            "SaveImage": save_image,
            "StringConcatenate": string_concat,
        }
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "15": {"class_type": "PrimitiveString", "inputs": {"value": "test"}},
                "17": {
                    "class_type": "StringConcatenate",
                    "inputs": {
                        "string_a": ["15", 0],
                        "string_b": ["8", 0],
                    },
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": ["17", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        self.assertNotIn("8", result)
        self.assertNotIn("17", result)
        self.assertNotIn("filename_prefix", result["9"]["inputs"])

    def test_does_not_preserve_scalar_output_for_non_scalar_downstream_input(self):
        """Scalar outputs are retained only for scalar/config downstream inputs."""
        string_provider = type("StringProvider", (), {"RETURN_TYPES": ("STRING",)})
        image_consumer = type(
            "ImageConsumer",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "mask": ("IMAGE",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {
            "ImageConsumer": image_consumer,
            "StringProvider": string_provider,
        }
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "17": {"class_type": "StringProvider", "inputs": {}},
                "9": {
                    "class_type": "ImageConsumer",
                    "inputs": {
                        "images": ["11", 0],
                        "mask": ["17", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        self.assertNotIn("17", result)
        self.assertNotIn("mask", result["9"]["inputs"])

    def test_preserves_scalar_list_join_subgraph_for_downstream_required_input(self):
        """Delegate-only master keeps list-shaped scalar config chains."""
        create_list = type(
            "CreateList",
            (),
            {
                "RETURN_TYPES": ("LIST",),
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {"inputs.input0": ("STRING",)},
                        "optional": {
                            "inputs.input1": ("STRING",),
                            "inputs.input2": ("STRING",),
                        },
                    }
                ),
            },
        )
        string_data_list_join = type(
            "StringDataListJoin",
            (),
            {
                "RETURN_TYPES": ("STRING",),
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "strings": ("LIST",),
                            "delimiter": ("STRING",),
                        }
                    }
                ),
            },
        )
        save_image = type(
            "SaveImage",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "filename_prefix": ("STRING",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {
            "CreateList": create_list,
            "SaveImage": save_image,
            "StringDataListJoin": string_data_list_join,
        }
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "15": {"class_type": "PrimitiveString", "inputs": {"value": "test"}},
                "17": {"class_type": "PrimitiveString", "inputs": {"value": "input_bug"}},
                "29": {"class_type": "PrimitiveString", "inputs": {"value": "new"}},
                "28": {
                    "class_type": "CreateList",
                    "inputs": {
                        "inputs.input0": ["15", 0],
                        "inputs.input1": ["17", 0],
                        "inputs.input2": ["29", 0],
                    },
                },
                "32": {
                    "class_type": "StringDataListJoin",
                    "inputs": {
                        "strings": ["28", 0],
                        "delimiter": "/",
                    },
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": ["32", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        self.assertIn("15", result)
        self.assertIn("17", result)
        self.assertIn("28", result)
        self.assertIn("29", result)
        self.assertIn("32", result)
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], ["32", 0])
        self.assertEqual(result["32"]["inputs"]["strings"], ["28", 0])
        self.assertEqual(result["28"]["inputs"]["inputs.input0"], ["15", 0])
        self.assertNotIn("8", result)

    def test_preserves_builtin_create_list_string_data_list_join_subgraph(self):
        """Delegate-only master handles ComfyUI 0.23 CreateList data-list output."""
        string_data_list_join = type(
            "StringDataListJoin",
            (),
            {
                "RETURN_TYPES": ("STRING",),
                "INPUT_IS_LIST": True,
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "strings": ("STRING", {"forceInput": True}),
                            "sep": ("STRING", {"default": " "}),
                        }
                    }
                ),
            },
        )
        save_image = type(
            "SaveImage",
            (),
            {
                "INPUT_TYPES": classmethod(
                    lambda cls: {
                        "required": {
                            "images": ("IMAGE",),
                            "filename_prefix": ("STRING",),
                        }
                    }
                )
            },
        )
        previous_mappings = getattr(pt, "_DELEGATE_MASTER_NODE_CLASS_MAPPINGS", None)
        pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = {
            "Basic data handling: StringDataListJoin": string_data_list_join,
            "SaveImage": save_image,
        }
        try:
            prompt = {
                "8": {"class_type": "KSampler", "inputs": {}},
                "11": {"class_type": "DistributedCollector", "inputs": {"images": ["8", 0]}},
                "15": {"class_type": "PrimitiveString", "inputs": {"value": "input_bug"}},
                "17": {"class_type": "PrimitiveString", "inputs": {"value": "test"}},
                "29": {"class_type": "PrimitiveString", "inputs": {"value": "new"}},
                "28": {
                    "class_type": "CreateList",
                    "inputs": {
                        "inputs.input0": ["17", 0],
                        "inputs.input1": ["15", 0],
                        "inputs.input2": ["29", 0],
                    },
                },
                "32": {
                    "class_type": "Basic data handling: StringDataListJoin",
                    "inputs": {
                        "strings": ["28", 0],
                        "sep": "/",
                    },
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": ["32", 0],
                    },
                },
            }

            result = pt.prepare_delegate_master_prompt(prompt, ["11"])
        finally:
            pt._DELEGATE_MASTER_NODE_CLASS_MAPPINGS = previous_mappings

        for node_id in ("15", "17", "28", "29", "32"):
            self.assertIn(node_id, result)
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], ["32", 0])
        self.assertEqual(result["32"]["inputs"]["strings"], ["28", 0])
        self.assertEqual(result["28"]["inputs"]["inputs.input0"], ["17", 0])
        self.assertNotIn("8", result)

    def test_result_is_independent_copy(self):
        prompt = _delegate_prompt()
        result = pt.prepare_delegate_master_prompt(prompt, ["3"])
        result["3"]["inputs"]["NEW"] = "injected"
        # Original should be untouched
        self.assertNotIn("NEW", prompt["3"].get("inputs", {}))


# ---------------------------------------------------------------------------
# generate_job_id_map
# ---------------------------------------------------------------------------

class GenerateJobIdMapTests(unittest.TestCase):
    def test_maps_collector_nodes(self):
        prompt = {
            "1": {"class_type": "DistributedCollector", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        idx = pt.PromptIndex(prompt)
        job_map = pt.generate_job_id_map(idx, "prefix")
        self.assertEqual(job_map["1"], "prefix_1")
        self.assertNotIn("2", job_map)

    def test_maps_upscale_nodes(self):
        prompt = {
            "5": {"class_type": "UltimateSDUpscaleDistributed", "inputs": {}},
        }
        idx = pt.PromptIndex(prompt)
        job_map = pt.generate_job_id_map(idx, "run")
        self.assertEqual(job_map["5"], "run_5")

    def test_empty_prompt_returns_empty_map(self):
        idx = pt.PromptIndex({})
        self.assertEqual(pt.generate_job_id_map(idx, "prefix"), {})

    def test_stable_ids_across_calls(self):
        prompt = {"1": {"class_type": "DistributedCollector", "inputs": {}}}
        idx = pt.PromptIndex(prompt)
        m1 = pt.generate_job_id_map(idx, "run")
        m2 = pt.generate_job_id_map(idx, "run")
        self.assertEqual(m1, m2)


# ---------------------------------------------------------------------------
# apply_participant_overrides – DistributedCollector
# ---------------------------------------------------------------------------

class ApplyOverridesCollectorTests(unittest.TestCase):
    def _collector_prompt(self):
        return {"1": {"class_type": "DistributedCollector", "inputs": {}}}

    def test_worker_sets_is_worker_true(self):
        result = _apply(self._collector_prompt(), "worker-a")
        self.assertTrue(result["1"]["inputs"]["is_worker"])

    def test_worker_sets_master_url(self):
        result = _apply(self._collector_prompt(), "worker-a")
        self.assertEqual(result["1"]["inputs"]["master_url"], "http://master.example.com")

    def test_worker_sets_worker_id(self):
        result = _apply(self._collector_prompt(), "worker-a")
        self.assertEqual(result["1"]["inputs"]["worker_id"], "worker-a")

    def test_worker_sets_delegate_only_false(self):
        result = _apply(self._collector_prompt(), "worker-a")
        self.assertFalse(result["1"]["inputs"]["delegate_only"])

    def test_master_sets_is_worker_false(self):
        result = _apply(self._collector_prompt(), "master")
        self.assertFalse(result["1"]["inputs"]["is_worker"])

    def test_master_clears_stale_master_url(self):
        prompt = {"1": {"class_type": "DistributedCollector", "inputs": {"master_url": "stale"}}}
        result = _apply(prompt, "master")
        self.assertNotIn("master_url", result["1"]["inputs"])

    def test_master_clears_stale_worker_id(self):
        prompt = {"1": {"class_type": "DistributedCollector", "inputs": {"worker_id": "stale"}}}
        result = _apply(prompt, "master")
        self.assertNotIn("worker_id", result["1"]["inputs"])

    def test_master_with_delegate_master_sets_delegate_only_true(self):
        result = _apply(self._collector_prompt(), "master", delegate_master=True)
        self.assertTrue(result["1"]["inputs"]["delegate_only"])

    def test_master_without_delegate_master_sets_delegate_only_false(self):
        result = _apply(self._collector_prompt(), "master", delegate_master=False)
        self.assertFalse(result["1"]["inputs"]["delegate_only"])

    def test_enabled_worker_ids_serialized_as_json(self):
        enabled = ["worker-a", "worker-b"]
        result = _apply(self._collector_prompt(), "master", enabled_worker_ids=enabled)
        self.assertEqual(result["1"]["inputs"]["enabled_worker_ids"], json.dumps(enabled))

    def test_multi_job_id_is_set_from_job_map(self):
        prompt = {"1": {"class_type": "DistributedCollector", "inputs": {}}}
        idx = pt.PromptIndex(prompt)
        job_id_map = {"1": "run_abc_1"}
        result = pt.apply_participant_overrides(
            prompt,
            participant_id="worker-a",
            enabled_worker_ids=["worker-a"],
            job_id_map=job_id_map,
            master_url="http://master",
            delegate_master=False,
            prompt_index=idx,
        )
        self.assertEqual(result["1"]["inputs"]["multi_job_id"], "run_abc_1")


# ---------------------------------------------------------------------------
# apply_participant_overrides – DistributedSeed
# ---------------------------------------------------------------------------

class ApplyOverridesSeedTests(unittest.TestCase):
    def _seed_prompt(self):
        return {"1": {"class_type": "DistributedSeed", "inputs": {}}}

    def test_worker_sets_is_worker_true(self):
        result = _apply(self._seed_prompt(), "worker-a")
        self.assertTrue(result["1"]["inputs"]["is_worker"])

    def test_worker_id_reflects_index_in_enabled_list(self):
        result = _apply(self._seed_prompt(), "worker-b", enabled_worker_ids=["worker-a", "worker-b"])
        self.assertEqual(result["1"]["inputs"]["worker_id"], "worker_1")

    def test_master_sets_is_worker_false(self):
        result = _apply(self._seed_prompt(), "master")
        self.assertFalse(result["1"]["inputs"]["is_worker"])

    def test_master_sets_empty_worker_id(self):
        result = _apply(self._seed_prompt(), "master")
        self.assertEqual(result["1"]["inputs"]["worker_id"], "")


# ---------------------------------------------------------------------------
# apply_participant_overrides – UltimateSDUpscaleDistributed
# ---------------------------------------------------------------------------

class ApplyOverridesUpscaleTests(unittest.TestCase):
    def _upscale_prompt(self):
        return {"1": {"class_type": "UltimateSDUpscaleDistributed", "inputs": {}}}

    def test_worker_sets_is_worker_true(self):
        result = _apply(self._upscale_prompt(), "worker-a")
        self.assertTrue(result["1"]["inputs"]["is_worker"])

    def test_worker_sets_master_url_and_worker_id(self):
        result = _apply(self._upscale_prompt(), "worker-a")
        self.assertEqual(result["1"]["inputs"]["master_url"], "http://master.example.com")
        self.assertEqual(result["1"]["inputs"]["worker_id"], "worker-a")

    def test_master_clears_master_url_and_worker_id(self):
        prompt = {"1": {"class_type": "UltimateSDUpscaleDistributed", "inputs": {"master_url": "x", "worker_id": "y"}}}
        result = _apply(prompt, "master")
        self.assertNotIn("master_url", result["1"]["inputs"])
        self.assertNotIn("worker_id", result["1"]["inputs"])

    def test_collector_downstream_of_upscale_gets_pass_through(self):
        """A DistributedCollector that is downstream of UltimateSDUpscaleDistributed → pass_through=True."""
        prompt = {
            "1": {"class_type": "UltimateSDUpscaleDistributed", "inputs": {}},
            "2": {"class_type": "DistributedCollector", "inputs": {"images": ["1", 0]}},
        }
        result = _apply(prompt, "worker-a", enabled_worker_ids=["worker-a"])
        self.assertTrue(result["2"]["inputs"].get("pass_through"))


# ---------------------------------------------------------------------------
# apply_participant_overrides – DistributedValue
# ---------------------------------------------------------------------------

class ApplyOverridesValueTests(unittest.TestCase):
    def _value_prompt(self):
        return {"1": {"class_type": "DistributedValue", "inputs": {}}}

    def test_worker_sets_is_worker_true(self):
        result = _apply(self._value_prompt(), "worker-a")
        self.assertTrue(result["1"]["inputs"]["is_worker"])

    def test_worker_id_reflects_index_in_enabled_list(self):
        result = _apply(self._value_prompt(), "worker-b", enabled_worker_ids=["worker-a", "worker-b"])
        self.assertEqual(result["1"]["inputs"]["worker_id"], "worker_1")

    def test_master_sets_is_worker_false(self):
        result = _apply(self._value_prompt(), "master")
        self.assertFalse(result["1"]["inputs"]["is_worker"])

    def test_master_sets_empty_worker_id(self):
        result = _apply(self._value_prompt(), "master")
        self.assertEqual(result["1"]["inputs"]["worker_id"], "")


if __name__ == "__main__":
    unittest.main()
