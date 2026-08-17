import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfd3_mosaic.pose_optimizer import (
    PoseEvaluation,
    _evaluation_from_manifest,
    initialize_global_seed_layout,
    optimize_candidate_subset,
    optimize_design_poses,
)
from rfd3_mosaic.schema import UserDesignSpec


def _atom_line(
    serial: int,
    atom_name: str,
    chain: str,
    residue: int,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {atom_name[0]:>2s}\n"
    )


def _write_structure(path: Path) -> None:
    lines: list[str] = []
    serial = 1
    for chain, x_offset in (("A", 5.0), ("B", 20.0), ("C", 35.0)):
        for residue in range(1, 3):
            for atom_name, delta in (
                ("N", -0.5),
                ("CA", 0.0),
                ("C", 0.5),
                ("CB", 1.0),
            ):
                lines.append(
                    _atom_line(
                        serial,
                        atom_name,
                        chain,
                        residue,
                        x_offset + 2.0 * (residue - 1),
                        0.0,
                        delta,
                    )
                )
                serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _evaluation(radius: float, target: float) -> PoseEvaluation:
    distance = abs(radius - target)
    hard_clashes = int(distance > 0.1)
    feasible = hard_clashes == 0
    return PoseEvaluation(
        score=(
            float(not feasible),
            float(hard_clashes),
            0.0,
            0.0,
            float(hard_clashes),
            0.0,
            0.0,
            distance,
            10.0,
            -2.0,
        ),
        feasible=feasible,
        hard_clashes=hard_clashes,
        failed_required_interfaces=(),
        infeasible_links=(),
        blocked_linker_corridors=(),
        required_objective_failures=0,
        linker_contour_excess=0.0,
        maximum_linker_endpoint_distance=10.0,
        minimum_linker_corridor_clearance=3.0,
        minimum_linker_axis_clearance=3.0,
        maximum_terminal_tangent_angle_deg=30.0,
        minimum_inter_group_distance=2.0,
        objective_penalty=distance,
    )


class ContinuousPoseOptimizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "components.pdb"
        _write_structure(self.structure)
        self.design = UserDesignSpec.model_validate(
            {
                "name": "two-component-pose-search",
                "input": self.structure,
                "symmetry": "C3",
                "components": {
                    "alpha": {
                        "selectors": ["A1-2"],
                        "geometry": "rigid",
                    },
                    "beta": {
                        "selectors": ["B1-2"],
                        "geometry": "rigid",
                    },
                },
                "sampling": {"timesteps": 10, "seed": 101},
            }
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_pattern_search_freezes_exact_replayable_component_poses(
        self,
    ) -> None:
        # The source alpha COM lies at radius 6.0 A. One +4 A polling step
        # is therefore the unique feasible solution under this deterministic
        # synthetic compiler objective.
        target = 10.0

        def evaluate(design: UserDesignSpec) -> PoseEvaluation:
            pose = design.sampling.initial_poses["alpha"]
            self.assertEqual(pose.radius.minimum, pose.radius.maximum)
            self.assertEqual(
                pose.axial_offset.minimum,
                pose.axial_offset.maximum,
            )
            return _evaluation(pose.radius.minimum, target)

        with patch(
            "rfd3_mosaic.pose_optimizer.evaluate_design_pose",
            side_effect=evaluate,
        ):
            result = optimize_design_poses(
                self.design,
                levels=1,
                maximum_translation=6.0,
                maximum_rotation_deg=15.0,
                translation_step=4.0,
                rotation_step_deg=5.0,
            )

        self.assertTrue(result.converged)
        self.assertGreater(result.accepted_update_count, 0)
        self.assertEqual(
            result.design.sampling.initial_poses["alpha"].radius.minimum,
            target,
        )
        self.assertEqual(
            result.design.components,
            self.design.components,
        )
        self.assertEqual(
            result.design.interfaces,
            self.design.interfaces,
        )

    def test_hard_constraint_regression_cannot_buy_soft_improvement(
        self,
    ) -> None:
        calls = 0

        def evaluate(design: UserDesignSpec) -> PoseEvaluation:
            nonlocal calls
            calls += 1
            radius = design.sampling.initial_poses["alpha"].radius.minimum
            if calls == 1:
                # Initial state is feasible but has a deliberately poor soft
                # score. Every moved state has a better soft objective and a
                # new hard clash; lexicographic acceptance must reject it.
                return PoseEvaluation(
                    score=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0,
                           10.0, -2.0),
                    feasible=True,
                    hard_clashes=0,
                    failed_required_interfaces=(),
                    infeasible_links=(),
                    blocked_linker_corridors=(),
                    required_objective_failures=0,
                    linker_contour_excess=0.0,
                    maximum_linker_endpoint_distance=10.0,
                    minimum_linker_corridor_clearance=3.0,
                    minimum_linker_axis_clearance=3.0,
                    maximum_terminal_tangent_angle_deg=30.0,
                    minimum_inter_group_distance=2.0,
                    objective_penalty=100.0,
                )
            return PoseEvaluation(
                score=(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                       radius, -3.0),
                feasible=False,
                hard_clashes=1,
                failed_required_interfaces=(),
                infeasible_links=(),
                blocked_linker_corridors=(),
                required_objective_failures=0,
                linker_contour_excess=0.0,
                maximum_linker_endpoint_distance=radius,
                minimum_linker_corridor_clearance=3.0,
                minimum_linker_axis_clearance=3.0,
                maximum_terminal_tangent_angle_deg=30.0,
                minimum_inter_group_distance=3.0,
                objective_penalty=0.0,
            )

        with patch(
            "rfd3_mosaic.pose_optimizer.evaluate_design_pose",
            side_effect=evaluate,
        ):
            result = optimize_design_poses(
                self.design,
                levels=1,
                maximum_translation=4.0,
                maximum_rotation_deg=10.0,
            )

        self.assertTrue(result.final_evaluation.feasible)
        self.assertEqual(result.accepted_update_count, 0)

    def test_connected_seed_pair_can_cross_coordinate_barrier(self) -> None:
        payload = self.design.model_dump(mode="json")
        payload["components"]["gamma"] = {
            "selectors": ["C1-2"],
            "geometry": "rigid",
        }
        payload["connections"] = [
            {
                "id": "alpha_to_beta",
                "from": "alpha.C",
                "to": "beta.N",
                "length": 20,
            }
        ]
        design = UserDesignSpec.model_validate(payload)
        initial_radii: dict[str, float] = {}

        def evaluate(candidate: UserDesignSpec) -> PoseEvaluation:
            radii = {
                component_id: pose.radius.minimum
                for component_id, pose in (
                    candidate.sampling.initial_poses or {}
                ).items()
            }
            if not initial_radii:
                initial_radii.update(radii)
            delta = tuple(
                round(radii[component_id] - initial_radii[component_id], 6)
                for component_id in ("alpha", "beta", "gamma")
            )
            exact = delta == (4.0, -4.0, 0.0)
            initial = delta == (0.0, 0.0, 0.0)
            penalty = 0.0 if exact else (10.0 if initial else 20.0)
            hard_clashes = 0 if exact else 1
            return PoseEvaluation(
                score=(
                    float(not exact),
                    float(hard_clashes),
                    0.0,
                    0.0,
                    float(hard_clashes),
                    0.0,
                    0.0,
                    penalty,
                    10.0,
                    -2.0,
                ),
                feasible=exact,
                hard_clashes=hard_clashes,
                failed_required_interfaces=(),
                infeasible_links=(),
                blocked_linker_corridors=(),
                required_objective_failures=0,
                linker_contour_excess=0.0,
                maximum_linker_endpoint_distance=10.0,
                minimum_linker_corridor_clearance=3.0,
                minimum_linker_axis_clearance=3.0,
                maximum_terminal_tangent_angle_deg=30.0,
                minimum_inter_group_distance=2.0,
                objective_penalty=penalty,
            )

        with patch(
            "rfd3_mosaic.pose_optimizer.evaluate_design_pose",
            side_effect=evaluate,
        ):
            result = optimize_design_poses(
                design,
                levels=1,
                maximum_translation=6.0,
                maximum_rotation_deg=15.0,
                translation_step=4.0,
                rotation_step_deg=5.0,
            )

        self.assertTrue(result.converged)
        accepted = [
            item for item in result.trajectory
            if item["component"] == "__joint__"
        ]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["pattern"], [1.0, -1.0, 0.0])
        self.assertEqual(
            result.metadata()["method"],
            "deterministic_connection_block_pattern_search_v2",
        )

    def test_flexible_linker_chord_obstruction_is_a_soft_routing_signal(
        self,
    ) -> None:
        manifest = {
            "validation": {
                "inter_group_clashes": {
                    "total_hard_clashes": 0,
                    "minimum_inter_group_distance": 6.4,
                },
                "interfaces": {
                    "failed_required_edge_instances": [],
                },
                "scaffold_link_geometry": {
                    "infeasible_link_instances": [],
                    "links": [
                        {
                            "link_instance_id": "polymer_link@orbit[0]",
                            "chain_break": False,
                            "endpoint_distance": 90.0,
                            "minimum_required_residues_at_3_8A": 24,
                            "configured_maximum_length": 45,
                            "minimum_interior_chord_fixed_atom_clearance": (
                                1.0
                            ),
                            "minimum_endpoint_chord_axis_clearance": 0.5,
                            "from_terminal_tangent_to_chord_angle_deg": 30.0,
                            "to_terminal_tangent_to_chord_angle_deg": 40.0,
                        }
                    ],
                },
                "objectives": {
                    "required_failure_count": 0,
                    "total_weighted_penalty": 0.0,
                },
            }
        }

        evaluation = _evaluation_from_manifest(manifest)

        self.assertTrue(evaluation.feasible)
        self.assertEqual(
            evaluation.blocked_linker_corridors,
            ("polymer_link@orbit[0]",),
        )
        self.assertEqual(evaluation.infeasible_links, ())
        self.assertEqual(evaluation.score[0], 0.0)
        self.assertEqual(evaluation.score[6], 1.0)

    def test_single_group_pose_accepts_absent_pair_distance(self) -> None:
        evaluation = _evaluation_from_manifest(
            {
                "validation": {
                    "inter_group_clashes": {
                        "total_hard_clashes": 0,
                        "minimum_inter_group_distance": None,
                    },
                    "interfaces": {
                        "failed_required_edge_instances": [],
                    },
                    "scaffold_link_geometry": {
                        "infeasible_link_instances": [],
                        "links": [],
                    },
                    "objectives": {
                        "required_failure_count": 0,
                        "total_weighted_penalty": 0.0,
                    },
                }
            }
        )

        self.assertTrue(evaluation.feasible)
        self.assertIsNone(evaluation.minimum_inter_group_distance)

    def test_insufficient_linker_contour_remains_a_hard_failure(self) -> None:
        manifest = {
            "validation": {
                "inter_group_clashes": {
                    "total_hard_clashes": 0,
                    "minimum_inter_group_distance": 6.4,
                },
                "interfaces": {
                    "failed_required_edge_instances": [],
                },
                "scaffold_link_geometry": {
                    "infeasible_link_instances": [
                        "polymer_link@orbit[0]"
                    ],
                    "links": [
                        {
                            "link_instance_id": "polymer_link@orbit[0]",
                            "chain_break": False,
                            "endpoint_distance": 190.0,
                            "minimum_required_residues_at_3_8A": 50,
                            "configured_maximum_length": 45,
                            "minimum_interior_chord_fixed_atom_clearance": (
                                3.0
                            ),
                        }
                    ],
                },
                "objectives": {
                    "required_failure_count": 0,
                    "total_weighted_penalty": 0.0,
                },
            }
        }

        evaluation = _evaluation_from_manifest(manifest)

        self.assertFalse(evaluation.feasible)
        self.assertEqual(
            evaluation.infeasible_links,
            ("polymer_link@orbit[0]",),
        )
        self.assertEqual(evaluation.linker_contour_excess, 5.0)

    def test_global_initializer_supports_dihedral_primary_axis(self) -> None:
        dihedral = self.design.model_copy(update={"symmetry": "D3"})

        initialized = initialize_global_seed_layout(
            dihedral,
            sample_index=2,
            sample_count=8,
        )

        self.assertEqual(set(initialized.sampling.initial_poses), {
            "alpha",
            "beta",
        })
        self.assertEqual(initialized.symmetry, "D3")
        self.assertNotEqual(
            initialized.sampling.initial_poses["alpha"].radial_direction,
            initialized.sampling.initial_poses["beta"].radial_direction,
        )

    def test_global_initializer_supports_full_polyhedral_orbits(self) -> None:
        tetrahedral = self.design.model_copy(update={"symmetry": "T"})

        initialized = initialize_global_seed_layout(
            tetrahedral,
            sample_index=3,
            sample_count=8,
        )

        self.assertEqual(set(initialized.sampling.initial_poses), {
            "alpha",
            "beta",
        })
        poses = initialized.sampling.initial_poses
        self.assertNotEqual(
            poses["alpha"].radial_direction,
            poses["beta"].radial_direction,
        )
        self.assertNotEqual(
            poses["alpha"].axial_offset.minimum,
            poses["beta"].axial_offset.minimum,
        )
        self.assertTrue(
            all(pose.radius.minimum > 0.0 for pose in poses.values())
        )

    def test_optimization_shortlist_is_not_a_hard_rejection_filter(
        self,
    ) -> None:
        initialized = initialize_global_seed_layout(
            self.design,
            sample_index=0,
            sample_count=1,
        )

        def evaluate(design: UserDesignSpec) -> PoseEvaluation:
            radius = design.sampling.initial_poses["alpha"].radius.minimum
            return _evaluation(radius, radius)

        with patch(
            "rfd3_mosaic.pose_optimizer.evaluate_design_pose",
            side_effect=evaluate,
        ):
            output = optimize_candidate_subset(
                (
                    ("candidate_a", initialized, {}),
                    ("candidate_b", initialized, {}),
                ),
                top_count=1,
                levels=1,
                maximum_translation=4.0,
                maximum_rotation_deg=10.0,
            )

        outside = output[1][2]
        self.assertEqual(
            outside["pose_optimization"]["reason"],
            "outside_initial_geometry_shortlist",
        )
        self.assertNotIn("preflight_failures", outside)


if __name__ == "__main__":
    unittest.main()
