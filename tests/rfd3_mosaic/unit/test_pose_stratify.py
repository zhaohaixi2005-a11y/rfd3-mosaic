import unittest

from rfd3_mosaic.pose_stratify import _bin_index, stratify_candidates


def _candidate(seed: int, penalty: float, radius: float, tilt: float):
    return {
        "pose_seed": seed,
        "accepted": True,
        "objective_penalty": penalty,
        "initialization_samples": {
            "primary_seed": {
                "sampled_radius": radius,
                "principal_axis_tilt_deg": tilt,
            }
        },
    }


class PoseStratifyTestCase(unittest.TestCase):
    def test_last_bin_includes_final_edge(self) -> None:
        self.assertEqual(_bin_index(30.0, [20.0, 25.0, 30.0]), 1)

    def test_best_candidate_is_retained_per_occupied_cell(self) -> None:
        ranking = [
            _candidate(1, 1.0, 21.0, 10.0),
            _candidate(2, 1.1, 21.5, 15.0),
            _candidate(3, 1.2, 26.0, 50.0),
            _candidate(4, 1.3, 29.0, 80.0),
        ]

        group_id, shortlist, coverage = stratify_candidates(
            ranking,
            radius_edges=[20.0, 25.0, 30.0],
            tilt_edges=[0.0, 30.0, 60.0, 90.0],
            per_cell=1,
        )

        self.assertEqual(group_id, "primary_seed")
        self.assertEqual(
            [candidate["pose_seed"] for candidate in shortlist],
            [1, 3, 4],
        )
        occupied = [
            item for item in coverage
            if "radius_bin" in item and item["candidate_count"] > 0
        ]
        self.assertEqual(len(occupied), 3)


if __name__ == "__main__":
    unittest.main()
