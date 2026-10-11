"""Regression tests for typed dynamic image-list normalization and graph scheduling."""
import asyncio
import unittest

import torch

from image_list.runtime import (
    END_CLASS,
    IMAGE_OUT,
    CONTEXT_OUT,
    START_CLASS,
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
    return {"class_type": START_CLASS, "inputs": {
        "images": image_link,
        "reference_image_mode": "per_item", "mask_mode": "per_item",
        "text_mode": "shared", "strength_mode": "shared", "seed_mode": "shared",
    }}


def graph_one_region():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "EmptyImage", "inputs": {"height": 8, "width": 8, "batch_size": 2}},
        "3": start_node(["2", 0]),
        "4": {"class_type": "GenericFrameOperation", "inputs": {
            "image": ["3", IMAGE_OUT], "model": ["1", 0], "seed": ["3", 5],
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
        rows = normalize_image_list_inputs(
            images, refs, masks, ["first", "second"], [0.2], [37],
            "per_item", "per_item", "per_item", "shared", "shared",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([tuple(row["image"].shape) for row in rows], [(1, 1, 1, 3), (1, 2, 3, 4)])
        self.assertEqual([row["image"].dtype for row in rows], [torch.float64, torch.float32])
        self.assertTrue(torch.equal(rows[0]["reference_image"], refs[:1]))
        self.assertEqual(rows[1]["mask"].dtype, torch.float32)
        self.assertEqual([row["index"] for row in rows], [0, 1])
        self.assertEqual([row["seed"] for row in rows], [37, 37])
        self.assertEqual([row["strength"] for row in rows], [0.2, 0.2])

    def test_explicit_per_item_and_shared_scalar_contracts(self):
        image = torch.zeros((3, 2, 4, 3), dtype=torch.float32)
        rows = normalize_image_list_inputs(image, text="same", strength=0.4, seed=9,
                                           text_mode="shared", strength_mode="shared", seed_mode="shared")
        self.assertEqual([row["text"] for row in rows], ["same"] * 3)
        self.assertEqual([row["strength"] for row in rows], [0.4] * 3)
        self.assertEqual([row["seed"] for row in rows], [9] * 3)
        with self.assertRaisesRegex(ValueError, "expected exactly 3"):
            normalize_image_list_inputs(image, text=["one", "two"], text_mode="per_item")
        with self.assertRaisesRegex(ValueError, "shared inputs require exactly one"):
            normalize_image_list_inputs(image, seed=[1, 2], seed_mode="shared")
        with self.assertRaisesRegex(ValueError, "expected exactly 3"):
            normalize_image_list_inputs(image, mask=torch.zeros((2, 2, 4)), mask_mode="per_item")

    def test_no_implicit_tail_repetition_and_no_empty_primary_batch(self):
        images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "expected exactly 2"):
            normalize_image_list_inputs(images, reference_image=torch.zeros((1, 2, 2, 3)),
                                        reference_image_mode="per_item")
        with self.assertRaisesRegex(ValueError, "at least one image"):
            normalize_image_list_inputs(None)

    def test_typed_transport_round_trips_float_images_masks_and_scalar_values(self):
        item = {"image": torch.arange(24, dtype=torch.float64).reshape(1, 2, 4, 3) / 7,
                "reference_image": torch.ones((1, 1, 3, 4), dtype=torch.float16),
                "mask": torch.full((1, 5, 6), 0.375, dtype=torch.float32),
                "text": "safe text", "strength": 0.625, "seed": 42, "index": 3}
        restored = decode_item(encode_item(item))
        self.assertTrue(torch.equal(restored["image"], item["image"]))
        self.assertEqual(restored["image"].dtype, torch.float64)
        self.assertTrue(torch.equal(restored["reference_image"], item["reference_image"]))
        self.assertTrue(torch.equal(restored["mask"], item["mask"]))
        self.assertEqual(restored["text"], "safe text")
        self.assertEqual(restored["seed"], 42)
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
        self.assertEqual(worker["4"]["inputs"]["model"], ["1", 0])
        self.assertNotIn("6", worker)

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
        with self.assertRaisesRegex(ValueError, "typed context output"):
            validate_image_list_regions(prompt)
        nested = graph_one_region()
        nested["7"] = start_node(["4", 0])
        nested["8"] = {"class_type": "NestedOperation", "inputs": {"image": ["7", IMAGE_OUT]}}
        nested["5"]["inputs"]["images"] = ["8", 0]
        nested["9"] = {"class_type": END_CLASS, "inputs": {"images": ["8", 0], "context": ["7", CONTEXT_OUT]}}
        with self.assertRaisesRegex(ValueError, "Nested/ambiguous"):
            validate_image_list_regions(nested)
        outside = graph_one_region()
        outside["7"] = {"class_type": "PreviewImage", "inputs": {"images": ["3", IMAGE_OUT]}}
        with self.assertRaisesRegex(ValueError, "cross-boundary"):
            validate_image_list_regions(outside)

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
