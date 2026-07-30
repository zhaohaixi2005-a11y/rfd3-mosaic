import unittest

from rfd3_mosaic.pose_qd import (
    _parse_descriptor,
    select_quality_diverse_candidates,
)


def _candidate(
    seed: int,
    *,
    clearance: float,
    aspect: float,
    quaternion: list[float],
) -> dict:
    return {
        "pose_seed": seed,
        "accepted": True,
        "objective_penalty": float(seed),
        "minimum_axis_clearance": clearance,
        "maximum_axial_to_radial_aspect_ratio": aspect,
        "initialization_samples": {
            "primary_seed": {"quaternion_xyzw": quaternion}
        },
    }


class PoseQualityDiversityTestCase(unittest.TestCase):
    def test_descriptor_parser_accepts_numeric_edges(self) -> None:
        name, edges = _parse_descriptor("aspect=0,0.5,1,10")

        self.assertEqual(name, "aspect")
        self.assertEqual(edges[:3], [0.0, 0.5, 1.0])
        self.assertEqual(edges[-1], 10.0)

    def test_best_ranked_candidate_is_retained_per_shape_cell(self) -> None:
        ranking = [
            _candidate(
                1,
                clearance=7.0,
                aspect=0.2,
                quaternion=[0.0, 0.0, 0.0, 1.0],
            ),
            _candidate(
                2,
                clearance=8.0,
                aspect=0.3,
                quaternion=[0.0, 0.0, 0.5, 0.8660254],
            ),
            _candidate(
                3,
                clearance=12.0,
                aspect=0.8,
                quaternion=[0.0, 0.7071068, 0.0, 0.7071068],
            ),
        ]

        group, shortlist, coverage = select_quality_diverse_candidates(
            ranking,
            descriptor_edges={
                "minimum_axis_clearance": [0.0, 10.0, 20.0],
                "maximum_axial_to_radial_aspect_ratio": [
                    0.0,
                    0.5,
                    1.5,
                ],
            },
            minimum_orientation_separation_degrees=0.0,
        )

        self.assertEqual(group, "primary_seed")
        self.assertEqual(
            [candidate["pose_seed"] for candidate in shortlist],
            [1, 3],
        )
        self.assertEqual(
            sum(
                item.get("selected_count", 0)
                for item in coverage
                if "cell" in item
            ),
            2,
        )

    def test_near_duplicate_orientation_uses_next_candidate_in_cell(
        self,
    ) -> None:
        ranking = [
            _candidate(
                1,
                clearance=7.0,
                aspect=0.2,
                quaternion=[0.0, 0.0, 0.0, 1.0],
            ),
            _candidate(
                2,
                clearance=12.0,
                aspect=0.8,
                quaternion=[0.0, 0.0, 0.01, 0.99995],
            ),
            _candidate(
                3,
                clearance=12.0,
                aspect=0.8,
                quaternion=[0.0, 0.7071068, 0.0, 0.7071068],
            ),
        ]

        _, shortlist, _ = select_quality_diverse_candidates(
            ranking,
            descriptor_edges={
                "minimum_axis_clearance": [0.0, 10.0, 20.0],
                "maximum_axial_to_radial_aspect_ratio": [
                    0.0,
                    0.5,
                    1.5,
                ],
            },
            minimum_orientation_separation_degrees=20.0,
        )

        self.assertEqual(
            [candidate["pose_seed"] for candidate in shortlist],
            [1, 3],
        )

    def test_quality_pool_excludes_low_rank_shape_outlier(self) -> None:
        ranking = [
            _candidate(
                1,
                clearance=7.0,
                aspect=0.2,
                quaternion=[0.0, 0.0, 0.0, 1.0],
            ),
            _candidate(
                2,
                clearance=8.0,
                aspect=0.3,
                quaternion=[0.0, 0.0, 0.5, 0.8660254],
            ),
            _candidate(
                3,
                clearance=12.0,
                aspect=0.8,
                quaternion=[0.0, 0.7071068, 0.0, 0.7071068],
            ),
            _candidate(
                4,
                clearance=18.0,
                aspect=1.2,
                quaternion=[0.5, 0.5, 0.5, 0.5],
            ),
        ]

        _, shortlist, coverage = select_quality_diverse_candidates(
            ranking,
            descriptor_edges={
                "minimum_axis_clearance": [0.0, 10.0, 20.0],
                "maximum_axial_to_radial_aspect_ratio": [
                    0.0,
                    0.5,
                    1.5,
                ],
            },
            quality_pool_fraction=0.5,
        )

        self.assertEqual(
            [candidate["pose_seed"] for candidate in shortlist],
            [1],
        )
        quality_record = next(
            item for item in coverage
            if "excluded_by_quality_pool" in item
        )
        self.assertEqual(quality_record["excluded_by_quality_pool"], 2)


if __name__ == "__main__":
    unittest.main()
