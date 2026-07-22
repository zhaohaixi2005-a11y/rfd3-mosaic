import unittest

from rfd3_mosaic.pose_select import (
    quaternion_angular_distance_degrees,
    select_diverse_candidates,
)


def _candidate(seed: int, penalty: float, quaternion: list[float]):
    return {
        "pose_seed": seed,
        "accepted": True,
        "objective_penalty": penalty,
        "maximum_linker_endpoint_distance": 20.0 + seed,
        "minimum_axis_clearance": 8.0,
        "initialization_samples": {
            "primary_seed": {"quaternion_xyzw": quaternion}
        },
    }


class PoseSelectTestCase(unittest.TestCase):
    def test_quaternion_sign_does_not_change_rotation(self) -> None:
        self.assertAlmostEqual(
            quaternion_angular_distance_degrees(
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -1.0],
            ),
            0.0,
        )

    def test_known_rotation_distance(self) -> None:
        self.assertAlmostEqual(
            quaternion_angular_distance_degrees(
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2**-0.5, 2**-0.5],
            ),
            90.0,
        )

    def test_selection_preserves_rank_and_skips_near_duplicate(self) -> None:
        ranking = [
            _candidate(1, 1.0, [0.0, 0.0, 0.0, 1.0]),
            _candidate(2, 1.1, [0.0, 0.0, 0.01, 0.99995]),
            _candidate(3, 1.2, [0.0, 0.0, 2**-0.5, 2**-0.5]),
        ]

        group_id, selected = select_diverse_candidates(
            ranking,
            count=2,
            minimum_separation_degrees=30.0,
            pool_size=3,
        )

        self.assertEqual(group_id, "primary_seed")
        self.assertEqual([item["pose_seed"] for item in selected], [1, 3])
        self.assertAlmostEqual(
            selected[1]["minimum_orientation_separation_deg"],
            90.0,
        )


if __name__ == "__main__":
    unittest.main()
