import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.pose_ensemble import (
    _candidate_summary,
    _joint_sample_overrides,
    _ranking_key,
)


class PoseEnsembleTestCase(unittest.TestCase):
    def test_accepts_candidate_only_when_all_static_gates_pass(self) -> None:
        manifest = {
            "initialization_samples": {"primary_seed": {}},
            "validation": {
                "inter_group_clashes": {
                    "total_hard_clashes": 0,
                    "minimum_inter_group_distance": 4.0,
                },
                "interfaces": {"all_required_satisfied": True},
                "scaffold_link_geometry": {
                    "all_continuous_links_within_maximum_contour": True,
                    "links": [
                        {
                            "endpoint_distance": 22.0,
                            "from_terminal_tangent_to_chord_angle_deg": 35.0,
                            "to_terminal_tangent_to_chord_angle_deg": 40.0,
                            "terminal_tangent_relative_angle_deg": 50.0,
                            "terminal_plane_normal_relative_angle_deg": 20.0,
                            "endpoint_chord_out_of_plane_angle_deg": 15.0,
                            "minimum_endpoint_chord_axis_clearance": 7.0,
                            "minimum_interior_chord_fixed_atom_clearance": 5.0,
                        }
                    ],
                },
                "symmetry_cavities": {
                    "orbits": [
                        {
                            "orbit_id": "primary_orbit",
                            "minimum_axis_clearance": 8.0,
                            "mean_axis_clearance": 12.0,
                            "maximum_axis_extent": 16.0,
                            "radial_thickness": 8.0,
                            "radial_thickness_fraction": 0.5,
                            "axial_span": 20.0,
                            "axial_to_radial_aspect_ratio": 0.625,
                            "shape_covariance_eigenvalues": [
                                4.0,
                                9.0,
                                16.0,
                            ],
                            "shape_sphericity": 0.5,
                        }
                    ]
                },
                "objectives": {
                    "required_failure_count": 0,
                    "total_weighted_penalty": 0.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = _candidate_summary(
                manifest,
                pose_seed=5,
                directory=Path(directory),
            )

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["pose_seed"], 5)
        self.assertEqual(summary["maximum_linker_endpoint_distance"], 22.0)
        self.assertEqual(summary["minimum_axis_clearance"], 8.0)
        self.assertIsNone(
            summary[
                "minimum_axis_clearance_fraction_of_sampled_radius"
            ]
        )
        self.assertEqual(summary["maximum_axial_span"], 20.0)
        self.assertEqual(
            summary["maximum_axial_to_radial_aspect_ratio"], 0.625
        )
        self.assertEqual(summary["minimum_shape_sphericity"], 0.5)
        self.assertIn("primary_orbit", summary["morphology_by_orbit"])
        self.assertEqual(
            summary["maximum_terminal_tangent_to_chord_angle_deg"],
            40.0,
        )
        self.assertEqual(
            summary["minimum_scaffold_chord_axis_clearance"], 7.0
        )
        self.assertEqual(
            summary["minimum_scaffold_chord_fixed_atom_clearance"], 5.0
        )

    def test_normalizes_axis_clearance_for_one_sampled_radius(self) -> None:
        manifest = {
            "initialization_samples": {
                "primary_seed": {"sampled_radius": 40.0}
            },
            "validation": {
                "inter_group_clashes": {
                    "total_hard_clashes": 0,
                    "minimum_inter_group_distance": 4.0,
                },
                "interfaces": {"all_required_satisfied": True},
                "scaffold_link_geometry": {
                    "all_continuous_links_within_maximum_contour": True,
                    "links": [],
                },
                "symmetry_cavities": {
                    "orbits": [
                        {
                            "orbit_id": "primary_orbit",
                            "minimum_axis_clearance": 20.0,
                        }
                    ]
                },
                "objectives": {
                    "required_failure_count": 0,
                    "total_weighted_penalty": 0.0,
                },
            },
        }

        summary = _candidate_summary(
            manifest,
            pose_seed=5,
            directory=Path("candidate"),
        )

        self.assertEqual(
            summary[
                "minimum_axis_clearance_fraction_of_sampled_radius"
            ],
            0.5,
        )

    def test_rejects_clashing_candidate(self) -> None:
        manifest = {
            "validation": {
                "inter_group_clashes": {
                    "total_hard_clashes": 3,
                    "minimum_inter_group_distance": 1.0,
                },
                "interfaces": {"all_required_satisfied": True},
                "scaffold_link_geometry": {
                    "all_continuous_links_within_maximum_contour": True
                },
                "objectives": {
                    "required_failure_count": 0,
                    "total_weighted_penalty": 0.0,
                },
            },
        }
        summary = _candidate_summary(
            manifest,
            pose_seed=6,
            directory=Path("candidate"),
        )

        self.assertFalse(summary["accepted"])

    def test_tie_break_prefers_shorter_worst_linker_span(self) -> None:
        base = {
            "accepted": True,
            "required_objective_failures": 0,
            "hard_clashes": 0,
            "objective_penalty": 0.0,
            "mean_linker_endpoint_distance": 20.0,
            "minimum_inter_group_distance": 5.0,
        }
        compact = {
            **base,
            "pose_seed": 1,
            "maximum_linker_endpoint_distance": 25.0,
        }
        extended = {
            **base,
            "pose_seed": 2,
            "maximum_linker_endpoint_distance": 45.0,
        }

        self.assertLess(_ranking_key(compact), _ranking_key(extended))

    def test_joint_latin_hypercube_covers_each_radius_stratum(self) -> None:
        overrides = _joint_sample_overrides(
            ["primary_seed"],
            sample_count=8,
            random_seed=1000,
        )
        radius_units = [
            item["primary_seed"]["radius_unit"] for item in overrides
        ]
        strata = sorted(int(value * 8) for value in radius_units)

        self.assertEqual(strata, list(range(8)))
        self.assertTrue(
            all(len(item["primary_seed"]["so3_unit"]) == 3 for item in overrides)
        )


if __name__ == "__main__":
    unittest.main()
