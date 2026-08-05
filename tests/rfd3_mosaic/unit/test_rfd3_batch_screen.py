import math
import unittest

import numpy as np

from rfd3_mosaic.rfd3_batch_screen import (
    cyclic_ring_descriptors,
    interchain_packing_descriptors,
)


def _cyclic_ca_coordinates(order: int) -> dict[str, np.ndarray]:
    coordinates = {}
    local = np.asarray(
        [
            [-1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    for index in range(order):
        angle = 2.0 * math.pi * index / order
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        center = np.asarray(
            [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        )
        coordinates[chr(ord("A") + index)] = (
            local @ rotation.T + center
        )
    return coordinates


class CyclicBatchScreenTestCase(unittest.TestCase):
    def test_exact_c5_ring_has_uniform_radius_and_angles(self) -> None:
        coordinates = _cyclic_ca_coordinates(5)

        descriptors = cyclic_ring_descriptors(coordinates, 5)

        self.assertTrue(descriptors["available"])
        self.assertAlmostEqual(
            descriptors["mean_chain_com_radius"],
            10.0,
            places=6,
        )
        self.assertLess(descriptors["chain_com_radial_cv"], 1e-7)
        self.assertLess(
            descriptors["angular_gap_rms_error_degrees"],
            1e-7,
        )
        self.assertLess(descriptors["chain_com_axial_rms"], 1e-7)

    def test_ring_descriptors_reject_wrong_chain_count(self) -> None:
        coordinates = _cyclic_ca_coordinates(5)

        descriptors = cyclic_ring_descriptors(coordinates, 6)

        self.assertFalse(descriptors["available"])
        self.assertIn("expected 6 chains", descriptors["reason"])

    def test_packing_separates_neighbor_and_nonneighbor_contacts(
        self,
    ) -> None:
        coordinates = _cyclic_ca_coordinates(5)
        ring = cyclic_ring_descriptors(coordinates, 5)

        packing = interchain_packing_descriptors(
            coordinates,
            ring["angular_chain_order"],
            contact_distance=15.0,
        )

        self.assertTrue(packing["available"])
        self.assertEqual(packing["neighbor_pair_count"], 5)
        self.assertGreater(packing["minimum_neighbor_ca_contacts"], 0)
        self.assertGreaterEqual(packing["nonneighbor_ca_contacts"], 0)
        self.assertGreater(
            packing["minimum_interchain_ca_distance"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
