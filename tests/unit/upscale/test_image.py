import unittest

import torch
from PIL import Image

from utils.image import ensure_contiguous, pil_to_tensor, tensor_to_pil


class ImageUtilsTests(unittest.TestCase):
    def test_tensor_to_pil_converts_batch_item(self):
        tensor = torch.tensor(
            [[[[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        image = tensor_to_pil(tensor, batch_index=0)

        self.assertEqual(image.size, (2, 1))
        self.assertEqual(image.getpixel((0, 0)), (0, 127, 255))
        self.assertEqual(image.getpixel((1, 0)), (255, 0, 0))

    def test_pil_to_tensor_adds_channel_for_grayscale(self):
        grayscale = Image.new("L", (3, 2), color=128)
        tensor = pil_to_tensor(grayscale)

        self.assertEqual(tuple(tensor.shape), (1, 2, 3, 1))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 128 / 255.0, places=6)

    def test_ensure_contiguous_makes_non_contiguous_tensor_contiguous(self):
        base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        non_contiguous = base.transpose(1, 2)
        self.assertFalse(non_contiguous.is_contiguous())

        contiguous = ensure_contiguous(non_contiguous)
        self.assertTrue(contiguous.is_contiguous())
        self.assertTrue(torch.equal(contiguous, non_contiguous))


if __name__ == "__main__":
    unittest.main()
