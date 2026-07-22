import unittest

import numpy as np

from rfd3.inference.symmetry.frames import get_dihedral_frames


class RFD3DihedralFrameCompatibilityTestCase(unittest.TestCase):
    def test_orders_divisible_by_three_have_all_unique_frames(self) -> None:
        for order in (3, 6):
            with self.subTest(order=order):
                rotations = [
                    rotation
                    for rotation, _ in get_dihedral_frames(order)
                ]
                for left_index, left in enumerate(rotations):
                    for right_index, right in enumerate(rotations):
                        if left_index == right_index:
                            continue
                        self.assertFalse(
                            np.allclose(left, right, atol=1e-9)
                        )

    def test_common_dihedral_groups_are_closed(self) -> None:
        for order in (2, 3, 5):
            with self.subTest(order=order):
                rotations = [
                    rotation
                    for rotation, _ in get_dihedral_frames(order)
                ]
                for left in rotations:
                    for right in rotations:
                        composed = left @ right
                        self.assertTrue(
                            any(
                                np.allclose(
                                    composed,
                                    candidate,
                                    atol=1e-9,
                                )
                                for candidate in rotations
                            )
                        )


if __name__ == "__main__":
    unittest.main()
