import unittest

from pydantic import ValidationError

from rfd3_mosaic.objectives import (
    build_static_metric_map,
    evaluate_objectives,
)
from rfd3_mosaic.schema import ObjectiveSpec


class ObjectiveScoringTestCase(unittest.TestCase):
    def test_required_constraint_failure_precedes_soft_score(self) -> None:
        objectives = {
            "no_clashes": ObjectiveSpec(
                metric="clashes.total_hard_clashes",
                mode="at_most",
                threshold=0.0,
                scale=1.0,
                required=True,
            ),
            "large_cavity": ObjectiveSpec(
                metric="cavities.minimum_central_void_radius",
                mode="maximize",
                scale=10.0,
                weight=0.5,
            ),
        }

        feasible = evaluate_objectives(
            objectives,
            {
                "clashes.total_hard_clashes": 0.0,
                "cavities.minimum_central_void_radius": 5.0,
            },
        )
        infeasible = evaluate_objectives(
            objectives,
            {
                "clashes.total_hard_clashes": 2.0,
                "cavities.minimum_central_void_radius": 100.0,
            },
        )

        self.assertLess(feasible.ranking_key, infeasible.ranking_key)
        self.assertTrue(feasible.all_required_satisfied)
        self.assertEqual(infeasible.required_failure_count, 1)

    def test_target_tolerance_has_zero_penalty_inside_window(self) -> None:
        report = evaluate_objectives(
            {
                "target_distance": ObjectiveSpec(
                    metric="interface.distance",
                    mode="target",
                    target=8.0,
                    tolerance=1.0,
                    scale=2.0,
                    required=True,
                )
            },
            {"interface.distance": 8.5},
        )

        evaluation = report.evaluations[0]
        self.assertEqual(evaluation.penalty, 0.0)
        self.assertTrue(evaluation.satisfied)

    def test_range_penalty_is_normalized_and_squared(self) -> None:
        report = evaluate_objectives(
            {
                "radius": ObjectiveSpec(
                    metric="assembly.radius",
                    mode="range",
                    minimum=20.0,
                    maximum=30.0,
                    scale=5.0,
                )
            },
            {"assembly.radius": 40.0},
        )

        self.assertEqual(report.evaluations[0].penalty, 4.0)

    def test_missing_metric_is_explicit(self) -> None:
        with self.assertRaisesRegex(KeyError, "missing metric"):
            evaluate_objectives(
                {
                    "missing": ObjectiveSpec(
                        metric="not.available",
                        mode="minimize",
                    )
                },
                {},
            )

    def test_directional_objective_cannot_be_required(self) -> None:
        with self.assertRaises(ValidationError):
            ObjectiveSpec(
                metric="assembly.radius",
                mode="maximize",
                required=True,
            )

    def test_mode_rejects_irrelevant_parameters(self) -> None:
        with self.assertRaises(ValidationError):
            ObjectiveSpec(
                metric="assembly.radius",
                mode="at_most",
                threshold=30.0,
                target=25.0,
            )

    def test_static_metric_map_exposes_backend_independent_names(self) -> None:
        metrics = build_static_metric_map(
            clash_report={
                "total_hard_clashes": 2,
                "minimum_inter_group_distance": 1.5,
            },
            interface_report={
                "failed_required_edge_instances": ["edge-1"],
            },
            linker_report={
                "infeasible_link_instances": ["link-1"],
                "links": [
                    {
                        "within_maximum_contour": False,
                        "endpoint_distance": 30.0,
                        "minimum_required_residues_at_3_8A": 7,
                    },
                    {
                        "within_maximum_contour": True,
                        "endpoint_distance": 20.0,
                        "minimum_required_residues_at_3_8A": 5,
                    },
                ],
            },
            cavity_report={
                "orbits": [
                    {
                        "central_void_radius": 5.0,
                        "minimum_axis_clearance": 3.0,
                    },
                    {
                        "central_void_radius": 4.0,
                        "minimum_axis_clearance": 2.0,
                    },
                ]
            },
        )

        self.assertEqual(metrics["clashes.total_hard_clashes"], 2.0)
        self.assertEqual(metrics["linkers.feasible_fraction"], 0.5)
        self.assertEqual(metrics["linkers.mean_endpoint_distance"], 25.0)
        self.assertEqual(metrics["linkers.maximum_endpoint_distance"], 30.0)
        self.assertEqual(
            metrics["linkers.maximum_minimum_required_residues"],
            7.0,
        )
        self.assertEqual(
            metrics["cavities.minimum_central_void_radius"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
