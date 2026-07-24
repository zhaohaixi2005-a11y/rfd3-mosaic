import unittest

import torch

from foundry.utils.alignment import weighted_rigid_align


class MixedPrecisionAlignmentTestCase(unittest.TestCase):
    def test_bfloat16_coordinates_use_float32_svd(self) -> None:
        torch.manual_seed(0)
        coordinates_float = torch.randn(1, 16, 3)
        angle = torch.tensor(1.0)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        rotation = torch.tensor(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = torch.tensor([3.0, -2.0, 5.0])
        coordinates = coordinates_float.to(torch.bfloat16)
        moved = (
            coordinates_float @ rotation.T + translation
        ).to(torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            aligned = weighted_rigid_align(coordinates, moved)

        self.assertEqual(aligned.dtype, torch.bfloat16)
        self.assertTrue(
            torch.allclose(
                aligned.float(),
                coordinates.float(),
                atol=2e-2,
            )
        )


if __name__ == "__main__":
    unittest.main()
