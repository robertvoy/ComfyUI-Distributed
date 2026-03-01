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


def _list_splitter_prompt():
    """1(List source) → 2(DistributedListSplitter) → 3(DistributedListCollector) → 4(SaveImage)."""
    return {
        "1": {"class_type": "ImageListSource", "inputs": {}},
        "2": {"class_type": "DistributedListSplitter", "inputs": {"images": ["1", 0]}},
        "3": {"class_type": "DistributedListCollector", "inputs": {"images": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }


def _branch_prompt():
    """1 → 2(DistributedBranch) -> slot0:3->4, slot1:5->6."""
    return {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "DistributedBranch", "inputs": {"input": ["1", 0], "num_branches": 2}},
        "3": {"class_type": "Blur", "inputs": {"image": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        "5": {"class_type": "Sharpen", "inputs": {"image": ["2", 1]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }


def _join_prompt():
    """1 → 2(DistributedBranch) -> 3(branch0) and 4(branch1) -> 5(DistributedJoin) -> 6(SaveImage)."""
    return {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "DistributedBranch", "inputs": {"input": ["1", 0], "num_branches": 2}},
        "3": {"class_type": "Blur", "inputs": {"image": ["2", 0]}},
        "4": {"class_type": "Sharpen", "inputs": {"image": ["2", 1]}},
        "5": {"class_type": "DistributedJoin", "inputs": {"input": ["3", 0], "num_branches": 2}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }


def _apply(prompt, participant_id, enabled_worker_ids=None, delegate_master=False):
    if enabled_worker_ids is None:
        enabled_worker_ids = ["worker-a", "worker-b"]
    prompt_copy = json.loads(json.dumps(prompt))
    idx = pt.PromptIndex(prompt_copy)
    job_id_map = pt.generate_job_id_map(idx, "run")
    return pt.apply_participant_overrides(
        prompt_copy,
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

    def test_list_collector_is_treated_as_distributed_anchor(self):
        prompt = _list_splitter_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertIn("3", result)
        self.assertNotIn("4", result)

    def test_branch_anchor_keeps_downstream_for_later_branch_pruning(self):
        prompt = _branch_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        self.assertIn("2", result)
        self.assertIn("3", result)
        self.assertIn("4", result)
        self.assertIn("5", result)
        self.assertIn("6", result)

    def test_join_anchor_keeps_join_and_downstream(self):
        prompt = _join_prompt()
        result = pt.prune_prompt_for_worker(prompt)
        self.assertIn("5", result)
        self.assertIn("6", result)


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

    def test_maps_list_collector_nodes(self):
        prompt = {
            "8": {"class_type": "DistributedListCollector", "inputs": {}},
        }
        idx = pt.PromptIndex(prompt)
        job_map = pt.generate_job_id_map(idx, "run")
        self.assertEqual(job_map["8"], "run_8")

    def test_maps_list_splitter_nodes(self):
        prompt = {
            "7": {"class_type": "DistributedListSplitter", "inputs": {}},
        }
        idx = pt.PromptIndex(prompt)
        job_map = pt.generate_job_id_map(idx, "run")
        self.assertEqual(job_map["7"], "run_7")

    def test_maps_join_nodes(self):
        prompt = {
            "10": {"class_type": "DistributedJoin", "inputs": {}},
        }
        idx = pt.PromptIndex(prompt)
        job_map = pt.generate_job_id_map(idx, "run")
        self.assertEqual(job_map["10"], "run_10")

    def test_maps_branch_nodes(self):
        prompt = {
            "9": {"class_type": "DistributedBranch", "inputs": {"num_branches": 2}},
        }
        idx = pt.PromptIndex(prompt)
        job_map = pt.generate_job_id_map(idx, "run")
        self.assertEqual(job_map["9"], "run_9")

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


class ApplyOverridesListSplitterTests(unittest.TestCase):
    def _splitter_prompt(self):
        return {"1": {"class_type": "DistributedListSplitter", "inputs": {}}}

    def test_master_gets_participant_index_zero(self):
        result = _apply(self._splitter_prompt(), "master", enabled_worker_ids=["worker-a", "worker-b"])
        self.assertEqual(result["1"]["inputs"]["participant_index"], 0)
        self.assertEqual(result["1"]["inputs"]["total_participants"], 3)
        self.assertFalse(result["1"]["inputs"]["is_worker"])
        self.assertNotIn("master_url", result["1"]["inputs"])
        self.assertEqual(result["1"]["inputs"]["worker_id"], "master")
        self.assertEqual(result["1"]["inputs"]["multi_job_id"], "run_1")

    def test_worker_gets_incremented_participant_index(self):
        result = _apply(self._splitter_prompt(), "worker-a", enabled_worker_ids=["worker-a", "worker-b"])
        self.assertEqual(result["1"]["inputs"]["participant_index"], 1)
        self.assertEqual(result["1"]["inputs"]["total_participants"], 3)
        self.assertTrue(result["1"]["inputs"]["is_worker"])
        self.assertEqual(result["1"]["inputs"]["worker_id"], "worker-a")
        self.assertEqual(result["1"]["inputs"]["master_url"], "http://master.example.com")

    def test_delegate_mode_shifts_worker_to_index_zero(self):
        result = _apply(
            self._splitter_prompt(),
            "worker-a",
            enabled_worker_ids=["worker-a", "worker-b"],
            delegate_master=True,
        )
        self.assertEqual(result["1"]["inputs"]["participant_index"], 0)
        self.assertEqual(result["1"]["inputs"]["total_participants"], 2)


class ApplyOverridesListCollectorTests(unittest.TestCase):
    def _collector_prompt(self):
        return {"1": {"class_type": "DistributedListCollector", "inputs": {}}}

    def test_worker_sets_master_url_and_worker_id(self):
        result = _apply(self._collector_prompt(), "worker-a", enabled_worker_ids=["worker-a"])
        self.assertTrue(result["1"]["inputs"]["is_worker"])
        self.assertEqual(result["1"]["inputs"]["master_url"], "http://master.example.com")
        self.assertEqual(result["1"]["inputs"]["worker_id"], "worker-a")

    def test_master_sets_delegate_only_and_clears_worker_fields(self):
        prompt = {
            "1": {
                "class_type": "DistributedListCollector",
                "inputs": {"master_url": "stale", "worker_id": "stale"},
            }
        }
        result = _apply(prompt, "master", enabled_worker_ids=["worker-a"], delegate_master=True)
        self.assertFalse(result["1"]["inputs"]["is_worker"])
        self.assertTrue(result["1"]["inputs"]["delegate_only"])
        self.assertNotIn("master_url", result["1"]["inputs"])
        self.assertNotIn("worker_id", result["1"]["inputs"])


class ApplyOverridesBranchTests(unittest.TestCase):
    def test_master_gets_branch_zero_worker_gets_branch_one(self):
        prompt = _branch_prompt()
        master_result = _apply(prompt, "master", enabled_worker_ids=["worker-a"])
        worker_result = _apply(prompt, "worker-a", enabled_worker_ids=["worker-a"])

        self.assertEqual(master_result["2"]["inputs"]["assigned_branch"], 0)
        self.assertEqual(worker_result["2"]["inputs"]["assigned_branch"], 1)
        self.assertIn("3", master_result)
        self.assertIn("4", master_result)
        self.assertNotIn("5", master_result)
        self.assertNotIn("6", master_result)
        self.assertIn("5", worker_result)
        self.assertIn("6", worker_result)
        self.assertNotIn("3", worker_result)
        self.assertNotIn("4", worker_result)

    def test_delegate_mode_assigns_worker_a_to_branch_zero(self):
        prompt = _branch_prompt()
        worker_result = _apply(
            prompt,
            "worker-a",
            enabled_worker_ids=["worker-a", "worker-b"],
            delegate_master=True,
        )
        self.assertEqual(worker_result["2"]["inputs"]["assigned_branch"], 0)

    def test_pruning_keeps_shared_nodes_between_branches(self):
        prompt = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "DistributedBranch", "inputs": {"input": ["1", 0], "num_branches": 2}},
            "3": {"class_type": "NodeA", "inputs": {"image": ["2", 0]}},
            "4": {"class_type": "NodeB", "inputs": {"image": ["2", 1]}},
            "5": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "aux": ["4", 0]}},
        }
        worker_result = _apply(prompt, "master", enabled_worker_ids=["worker-a"])
        self.assertIn("3", worker_result)
        self.assertNotIn("4", worker_result)
        self.assertIn("5", worker_result)
        self.assertNotIn("aux", worker_result["5"]["inputs"])

    def test_more_participants_than_branches_sets_unassigned_to_minus_one(self):
        prompt = _branch_prompt()
        worker_b_result = _apply(
            prompt,
            "worker-b",
            enabled_worker_ids=["worker-a", "worker-b", "worker-c"],
        )
        self.assertEqual(worker_b_result["2"]["inputs"]["assigned_branch"], -1)
        self.assertNotIn("3", worker_b_result)
        self.assertNotIn("4", worker_b_result)
        self.assertNotIn("5", worker_b_result)
        self.assertNotIn("6", worker_b_result)


class ApplyOverridesJoinTests(unittest.TestCase):
    def test_join_inherits_assigned_branch_from_upstream_branch_node(self):
        master_prompt = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "DistributedBranch", "inputs": {"input": ["1", 0], "num_branches": 2}},
            "3": {"class_type": "DistributedJoin", "inputs": {"input": ["2", 0], "num_branches": 2}},
        }
        worker_prompt = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "DistributedBranch", "inputs": {"input": ["1", 0], "num_branches": 2}},
            "3": {"class_type": "DistributedJoin", "inputs": {"input": ["2", 1], "num_branches": 2}},
        }

        master_result = _apply(master_prompt, "master", enabled_worker_ids=["worker-a"])
        worker_result = _apply(worker_prompt, "worker-a", enabled_worker_ids=["worker-a"])

        self.assertEqual(master_result["3"]["inputs"]["assigned_branch"], 0)
        self.assertEqual(worker_result["3"]["inputs"]["assigned_branch"], 1)

    def test_join_sets_multi_job_id(self):
        prompt = {
            "1": {"class_type": "DistributedJoin", "inputs": {"input": ["2", 0]}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        result = _apply(prompt, "master", enabled_worker_ids=["worker-a"])
        self.assertEqual(result["1"]["inputs"]["multi_job_id"], "run_1")

if __name__ == "__main__":
    unittest.main()
