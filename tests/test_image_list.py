"""Regression tests for typed dynamic image-list normalization and graph scheduling."""
import asyncio
import unittest

import torch

from image_list.runtime import (
    END_CLASS,
    IMAGE_OUT,
    CONTEXT_OUT,
    START_CLASS,
    VALUE_OUT_BASE,
    ImageListCancelledError,
    build_worker_prompt,
    create_master_job,
    decode_item,
    decode_result,
    dynamic_schedule,
    encode_item,
    encode_result,
    normalize_image_list_inputs,
    normalize_worker_result,
    remove_master_job,
    set_master_result,
    validate_image_list_regions,
)


def start_node(image_link):
    return {"class_type": START_CLASS, "inputs": {"images": image_link}}


def graph_one_region():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "EmptyImage", "inputs": {"height": 8, "width": 8, "batch_size": 2}},
        "3": start_node(["2", 0]),
        "4": {"class_type": "GenericFrameOperation", "inputs": {
            "image": ["3", IMAGE_OUT], "model": ["1", 0], "seed": ["3", VALUE_OUT_BASE],
        }},
        "5": {"class_type": END_CLASS, "inputs": {
            "images": ["4", 0], "context": ["3", CONTEXT_OUT],
        }},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }


def graph_two_sequential_regions():
    return {
        "1": {"class_type": "EmptyImage", "inputs": {"height": 4, "width": 4, "batch_size": 2}},
        "2": start_node(["1", 0]),
        "3": {"class_type": "FirstFrameOperation", "inputs": {"image": ["2", IMAGE_OUT]}},
        "4": {"class_type": END_CLASS, "inputs": {"images": ["3", 0], "context": ["2", CONTEXT_OUT]}},
        "5": start_node(["4", 0]),
        "6": {"class_type": "SecondFrameOperation", "inputs": {"image": ["5", IMAGE_OUT]}},
        "7": {"class_type": END_CLASS, "inputs": {"images": ["6", 0], "context": ["5", CONTEXT_OUT]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0]}},
    }


class ImageListNormalizationTests(unittest.TestCase):
    def test_batch_frame_normalization_preserves_sizes_channels_and_float_values(self):
        images = [
            torch.tensor([[[[0.125, 0.25, 0.5]]]], dtype=torch.float64),
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4) / 17,
        ]
        refs = torch.ones((2, 5, 7, 3), dtype=torch.float16) * 0.333
        masks = [torch.full((1, 2, 3), 0.125, dtype=torch.float64),
                 torch.full((1, 8, 9), 0.875, dtype=torch.float32)]
        rows = normalize_image_list_inputs(images, masks, {
            "value_1": [refs[:1], refs[1:]], "value_2": ["first", "second"],
            "value_3": [0.2], "value_4": [37],
        })
        self.assertEqual(len(rows), 2)
        self.assertEqual([tuple(row["image"].shape) for row in rows], [(1, 1, 1, 3), (1, 2, 3, 4)])
        self.assertEqual([row["image"].dtype for row in rows], [torch.float64, torch.float32])
        self.assertTrue(torch.equal(rows[0]["values"]["value_1"], refs[:1]))
        self.assertEqual(rows[1]["mask"].dtype, torch.float32)
        self.assertEqual([row["index"] for row in rows], [0, 1])
        self.assertEqual([row["values"]["value_2"] for row in rows], ["first", "second"])
        self.assertEqual([row["values"]["value_3"] for row in rows], [0.2, 0.2])
        self.assertEqual([row["values"]["value_4"] for row in rows], [37, 37])

    def test_one_value_is_shared_and_n_values_pair_in_order(self):
        image = torch.zeros((3, 2, 4, 3), dtype=torch.float32)
        rows = normalize_image_list_inputs(image, values={
            "value_1": ["same"], "value_2": [0.4], "value_3": [9], "value_4": [True]})
        self.assertEqual([row["values"]["value_1"] for row in rows], ["same"] * 3)
        self.assertEqual([row["values"]["value_2"] for row in rows], [0.4] * 3)
        self.assertEqual([row["values"]["value_3"] for row in rows], [9] * 3)
        self.assertEqual([row["values"]["value_4"] for row in rows], [True] * 3)
        rows = normalize_image_list_inputs(image, values={"value_2": [1, 2, 3]})
        self.assertEqual([row["values"] for row in rows], [{"value_2": 1}, {"value_2": 2}, {"value_2": 3}])
        with self.assertRaisesRegex(ValueError, "value_1 has 2 values; connect exactly 1.*or 3"):
            normalize_image_list_inputs(image, values={"value_1": ["one", "two"]})
        with self.assertRaisesRegex(ValueError, "mask has 2 values"):
            normalize_image_list_inputs(image, mask=torch.zeros((2, 2, 4)))
        shared_mask = normalize_image_list_inputs(image, mask=torch.ones((1, 2, 4)))
        self.assertTrue(all(torch.equal(row["mask"], torch.ones((1, 2, 4))) for row in shared_mask))

    def test_value_batches_stay_whole_and_only_images_split(self):
        images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        pair = torch.rand((2, 4, 4, 3))  # e.g. two references for one item
        rows = normalize_image_list_inputs(images, values={"value_1": [pair]})
        self.assertTrue(all(torch.equal(row["values"]["value_1"], pair) for row in rows))
        per_item = [torch.rand((2, 4, 4, 3)), torch.rand((2, 4, 4, 3))]
        rows = normalize_image_list_inputs(images, values={"value_1": per_item})
        self.assertTrue(torch.equal(rows[1]["values"]["value_1"], per_item[1]))

    def test_rejects_unsupported_values_and_counts(self):
        images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "value_1 has 3 values"):
            normalize_image_list_inputs(images, values={"value_1": [1, 2, 3]})
        with self.assertRaisesRegex(TypeError, "unsupported type dict"):
            normalize_image_list_inputs(images, values={"value_1": [{"samples": 1}]})
        with self.assertRaisesRegex(TypeError, "floating-point"):
            normalize_image_list_inputs(images, values={"value_1": [torch.zeros(2, dtype=torch.int64)]})
        with self.assertRaisesRegex(TypeError, "finite"):
            normalize_image_list_inputs(images, values={"value_1": [float("nan")]})
        with self.assertRaisesRegex(ValueError, "Unknown Start value slot"):
            normalize_image_list_inputs(images, values={"value_11": [1]})
        with self.assertRaisesRegex(ValueError, "at least one image"):
            normalize_image_list_inputs(None)
        # Unconnected Autogrow slots arrive as None and are ignored.
        rows = normalize_image_list_inputs(images, values={"value_1": [None], "value_2": None})
        self.assertEqual([row["values"] for row in rows], [{}, {}])

    def test_typed_transport_round_trips_float_images_masks_and_scalar_values(self):
        item = {"image": torch.arange(24, dtype=torch.float64).reshape(1, 2, 4, 3) / 7,
                "mask": torch.full((1, 5, 6), 0.375, dtype=torch.float32),
                "values": {"value_1": torch.ones((2, 1, 3, 4), dtype=torch.float16),
                           "value_2": "safe text", "value_3": 0.625, "value_4": 42,
                           "value_5": False},
                "index": 3}
        restored = decode_item(encode_item(item))
        self.assertTrue(torch.equal(restored["image"], item["image"]))
        self.assertEqual(restored["image"].dtype, torch.float64)
        self.assertTrue(torch.equal(restored["values"]["value_1"], item["values"]["value_1"]))
        self.assertEqual(restored["values"]["value_1"].dtype, torch.float16)
        self.assertTrue(torch.equal(restored["mask"], item["mask"]))
        self.assertEqual(restored["values"]["value_2"], "safe text")
        self.assertEqual(restored["values"]["value_3"], 0.625)
        self.assertIs(restored["values"]["value_4"], 42)
        self.assertIs(restored["values"]["value_5"], False)
        with self.assertRaisesRegex(ValueError, "unknown slot"):
            decode_item({**encode_item(item), "values": {"strength": {"type": "FLOAT", "value": 1.0}}})
        with self.assertRaisesRegex(ValueError, "unsupported type"):
            decode_item({**encode_item(item), "values": {"value_1": {"type": "PATH", "value": "/x"}}})
        with self.assertRaisesRegex(TypeError, "floating-point tensor"):
            encode_item({**item, "image": torch.ones((1, 2, 4, 3), dtype=torch.int32)})
        with self.assertRaisesRegex(ValueError, "Unsupported image-list transport type"):
            from image_list.runtime import encode_value
            encode_value("/untrusted/path", "PATH")

    def test_result_transport_preserves_ragged_result_shapes(self):
        result = normalize_worker_result(
            [torch.ones((1, 3, 5, 4), dtype=torch.float32)],
            [torch.ones((1, 3, 5), dtype=torch.float16)],
            ["done"],
        )
        encoded = encode_result(result)
        decoded = decode_result(encoded)
        self.assertEqual(tuple(decoded["image"].shape), (1, 3, 5, 4))
        self.assertEqual(decoded["image"].dtype, torch.float32)
        self.assertEqual(decoded["mask"].dtype, torch.float16)
        self.assertEqual(decoded["text"], "done")
        with self.assertRaisesRegex(ValueError, "exactly one IMAGE"):
            normalize_worker_result(torch.zeros((2, 3, 4, 3)))


class ImageListGraphBoundaryTests(unittest.TestCase):
    def test_prunes_upstream_item_source_but_keeps_external_model_loaders(self):
        prompt = graph_one_region()
        region = validate_image_list_regions(prompt)[0]
        worker = build_worker_prompt(prompt, region, "token_12345678901234567890123456789012",
                                     "ctx", 1, "http://master.invalid")
        self.assertEqual(set(worker), {"1", "3", "4", "5"})
        self.assertEqual(worker["3"]["inputs"]["worker_item_token"], "token_12345678901234567890123456789012")
        self.assertNotIn("images", worker["3"]["inputs"])

    def test_worker_prompt_drops_per_item_value_links(self):
        prompt = graph_one_region()
        prompt["8"] = {"class_type": "StringList", "inputs": {}}
        prompt["3"]["inputs"].update({"values.value_1": ["8", 0], "values.value_2": ["8", 0]})
        import sys, types
        from unittest.mock import patch
        registry = {"CheckpointLoaderSimple": type("Loader", (), {"RETURN_TYPES": ("MODEL",)})}
        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS=registry)}):
            region = validate_image_list_regions(prompt)[0]
        worker = build_worker_prompt(prompt, region, "token_12345678901234567890123456789012", "ctx", 0, "http://m")
        self.assertFalse([name for name in worker["3"]["inputs"] if name.startswith("values.")])
        self.assertNotIn("8", worker)
        self.assertEqual(worker["4"]["inputs"]["model"], ["1", 0])
        self.assertNotIn("6", worker)

    def test_rejects_upstream_processing_and_leaf_image_bypasses(self):
        # Both a leaf source and a generated batch must cross Start exactly once.
        for generated in (False, True):
            with self.subTest(generated=generated):
                prompt = graph_one_region()
                if generated:
                    prompt["2"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0]}}
                    prompt["7"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0]}}
                prompt["4"]["inputs"]["reference"] = ["2", 0]
                for explicit_targets in (False, True):
                    with self.subTest(explicit_targets=explicit_targets):
                        with self.assertRaisesRegex(ValueError, "bypasses Start.*2"):
                            validate_image_list_regions(prompt, explicit_targets=explicit_targets)

    def test_shared_typed_loader_and_leaf_constant_remain_valid(self):
        import sys
        import types
        from unittest.mock import patch
        prompt = graph_one_region()
        prompt["7"] = {"class_type": "SeedConstant", "inputs": {"value": 9}}
        prompt["2"]["inputs"].update({"model": ["1", 0], "seed": ["7", 0]})
        prompt["4"]["inputs"]["shared_seed"] = ["7", 0]
        registry = {
            "CheckpointLoaderSimple": type("Loader", (), {"RETURN_TYPES": ("MODEL", "CLIP", "VAE")}),
            "SeedConstant": type("Constant", (), {"RETURN_TYPES": ("INT",)}),
        }
        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS=registry)}):
            region = validate_image_list_regions(prompt)[0]
        worker = build_worker_prompt(prompt, region, "token", "ctx", 0, "http://master")
        self.assertNotIn("2", worker)
        self.assertTrue({"1", "7"}.issubset(worker))

    def test_unknown_or_list_producing_shared_dependency_is_not_exempt(self):
        import sys
        import types
        from unittest.mock import patch
        for node_class in (None, type("ListLoader", (), {
                "RETURN_TYPES": ("MODEL",), "OUTPUT_IS_LIST": (True,)})):
            prompt = graph_one_region()
            prompt["2"]["inputs"]["model"] = ["1", 0]
            registry = {} if node_class is None else {"CheckpointLoaderSimple": node_class}
            with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS=registry)}):
                with self.assertRaisesRegex(ValueError, "bypasses Start.*1"):
                    validate_image_list_regions(prompt)

    def test_two_sequential_regions_are_valid_and_isolated(self):
        regions = validate_image_list_regions(graph_two_sequential_regions())
        self.assertEqual([(r["start_id"], r["end_id"]) for r in regions], [("2", "4"), ("5", "7")])
        p = graph_two_sequential_regions()
        second = build_worker_prompt(p, regions[1], "item_token_123456789012345678901234", "ctx", 0,
                                     "http://master.invalid")
        self.assertEqual(set(second), {"5", "6", "7"})

    def test_rejects_missing_context_pair_nested_region_and_cross_boundary_branch(self):
        prompt = graph_one_region()
        prompt["5"]["inputs"]["context"] = ["2", 0]
        for explicit_targets in (False, True):
            with self.subTest(explicit_targets=explicit_targets):
                with self.assertRaisesRegex(ValueError, "typed context output"):
                    validate_image_list_regions(prompt, explicit_targets=explicit_targets)
        nested = graph_one_region()
        nested["7"] = start_node(["4", 0])
        nested["8"] = {"class_type": "NestedOperation", "inputs": {"image": ["7", IMAGE_OUT]}}
        nested["5"]["inputs"]["images"] = ["8", 0]
        nested["9"] = {"class_type": END_CLASS, "inputs": {"images": ["8", 0], "context": ["7", CONTEXT_OUT]}}
        for explicit_targets in (False, True):
            with self.subTest(explicit_targets=explicit_targets):
                with self.assertRaisesRegex(ValueError, "Nested/ambiguous"):
                    validate_image_list_regions(nested, explicit_targets=explicit_targets)
        outside = graph_one_region()
        outside["7"] = {"class_type": "PreviewImage", "inputs": {"images": ["3", IMAGE_OUT]}}
        for explicit_targets in (False, True):
            with self.subTest(explicit_targets=explicit_targets):
                with self.assertRaisesRegex(ValueError, "cross-boundary"):
                    validate_image_list_regions(outside, explicit_targets=explicit_targets)

    def test_rejects_duplicate_or_missing_results(self):
        prompt = graph_one_region()
        prompt["5"]["inputs"].pop("images")
        with self.assertRaisesRegex(ValueError, "images must be connected"):
            validate_image_list_regions(prompt)


class DynamicImageSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_workers_get_more_items_and_order_is_stable(self):
        assignments = {"fast": [], "slow": []}
        async def dispatch(worker, index):
            assignments[worker].append(index)
            delay = 0.002 if worker == "fast" else 0.025
            task = asyncio.create_task(asyncio.sleep(delay, result=(index, f"result-{index}")))
            return {"index": index, "task": task}
        async def wait_next(inflight):
            done, _ = await asyncio.wait([h["task"] for h in inflight.values()], return_when=asyncio.FIRST_COMPLETED)
            task = next(iter(done))
            worker = next(w for w, handle in inflight.items() if handle["task"] is task)
            index, result = task.result()
            return worker, index, result
        async def cleanup(_worker, handle):
            handle["task"].cancel()
        ordered = await dynamic_schedule(12, ["fast", "slow"], dispatch, wait_next, cleanup)
        self.assertEqual(ordered, [f"result-{i}" for i in range(12)])
        self.assertGreater(len(assignments["fast"]), len(assignments["slow"]))
        self.assertEqual(sorted(assignments["fast"] + assignments["slow"]), list(range(12)))

    async def test_failure_timeout_and_cancel_clean_only_current_assignments(self):
        async def dispatch(worker, index):
            return {"index": index, "prompt_id": f"{worker}-{index}"}
        cancelled = []
        async def cleanup(worker, handle):
            cancelled.append((worker, handle["prompt_id"]))
        async def timeout(_inflight):
            raise TimeoutError("worker timed out")
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            await dynamic_schedule(4, ["one", "two"], dispatch, timeout, cleanup)
        self.assertCountEqual(cancelled, [("one", "one-0"), ("two", "two-1")])

        cancelled.clear()
        async def wrong_index(inflight):
            worker, handle = next(iter(inflight.items()))
            return worker, handle["index"] + 99, "bad"
        with self.assertRaisesRegex(RuntimeError, "unexpected image index"):
            await dynamic_schedule(2, ["one"], dispatch, wrong_index, cleanup)
        self.assertEqual(len(cancelled), 1)

        cancelled.clear()
        with self.assertRaises(ImageListCancelledError):
            await dynamic_schedule(2, ["one", "two"], dispatch, timeout, cleanup,
                                   interrupted=lambda: True)
        self.assertEqual(len(cancelled), 0)

    async def test_no_workers_and_duplicate_results_fail_closed(self):
        async def unused(*_args):
            self.fail("must not dispatch without workers")
        with self.assertRaisesRegex(RuntimeError, "No free"):
            await dynamic_schedule(2, [], unused, unused, unused)
        async def dispatch(worker, index):
            return {"index": index}
        async def duplicate(inflight):
            worker, _handle = next(iter(inflight.items()))
            return worker, 0, "same"
        with self.assertRaisesRegex(RuntimeError, "unexpected image index"):
            await dynamic_schedule(2, ["one"], dispatch, duplicate, lambda *_: asyncio.sleep(0))


if __name__ == "__main__":
    unittest.main()
