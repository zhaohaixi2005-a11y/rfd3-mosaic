import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfd3_mosaic.pose_optimizer import (
    PoseEvaluation,
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
    for chain, x_offset in (("A", 5.0), ("B", 20.0)):
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


if __name__ == "__main__":
    unittest.main()
