import unittest

import torch

from upscale.conditioning import clone_conditioning, clone_control_chain


class _FakeControl:
    def __init__(self, hint, previous=None):
        self.cond_hint_original = hint
        self.previous_controlnet = previous


class ConditioningUtilsTests(unittest.TestCase):
    def test_clone_control_chain_clones_hints_and_links(self):
        tail = _FakeControl(torch.ones((1, 2)))
        head = _FakeControl(torch.zeros((1, 2)), previous=tail)

        cloned = clone_control_chain(head, clone_hint=True)

        self.assertIsNot(cloned, head)
        self.assertIsNot(cloned.previous_controlnet, tail)
        self.assertTrue(torch.equal(cloned.cond_hint_original, head.cond_hint_original))
        self.assertTrue(torch.equal(cloned.previous_controlnet.cond_hint_original, tail.cond_hint_original))
        self.assertIsNot(cloned.cond_hint_original, head.cond_hint_original)

    def test_clone_conditioning_clones_emb_and_known_fields(self):
        control = _FakeControl(torch.ones((1, 3)))
        cond = [
            [
                torch.tensor([[1.0, 2.0, 3.0]]),
                {
                    "control": control,
                    "mask": torch.ones((1, 1, 1)),
                    "pooled_output": torch.zeros((1, 4)),
                    "area": [0, 1, 2, 3],
                },
            ]
        ]

        cloned = clone_conditioning(cond, clone_hints=True)

        self.assertTrue(torch.equal(cloned[0][0], cond[0][0]))
        self.assertIsNot(cloned[0][0], cond[0][0])
        self.assertIsNot(cloned[0][1]["control"], cond[0][1]["control"])
        self.assertIsNot(cloned[0][1]["mask"], cond[0][1]["mask"])
        self.assertIsNot(cloned[0][1]["pooled_output"], cond[0][1]["pooled_output"])
        self.assertEqual(cloned[0][1]["area"], cond[0][1]["area"])
        self.assertIsNot(cloned[0][1]["area"], cond[0][1]["area"])


if __name__ == "__main__":
    unittest.main()
