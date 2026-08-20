import json
import math
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.rfd3_mobility_audit import (
    audit_component_mobility,
    write_mobility_trajectory,
)


class ComponentMobilityAuditTestCase(unittest.TestCase):
    def _write_inputs(
        self,
        root: Path,
        *,
        translation: float = 1.5,
        rotation: float = 4.0,
        active: bool = True,
        subspace: str = "bounded_se3",
        rotation_bound: float | None = 10.0,
        mobile_count: int = 1,
        atomic_joint: bool = True,
        group_action_count: int = 3,
        observed_group_action_count: int | None = None,
        symmetry_id: str = "C3",
        translation_vector: tuple[float, float, float] | None = None,
    ) -> tuple[Path, Path]:
        if observed_group_action_count is None:
            observed_group_action_count = group_action_count
        if translation_vector is None:
            translation_vector = (translation, 0.0, 0.0)
        mobile_orbits = [
            {
                "constraint_orbit_id": f"mobile_orbit_{index}",
                "coupling_group_id": f"mobile_component_{index}",
                "mobility_mode": "orbit_rigid",
                "mobility_subspace": subspace,
                "max_translation": 3.0,
                "max_rotation_deg": rotation_bound,
                "group_transform_ids": list(range(group_action_count)),
            }
            for index in range(mobile_count)
        ]
        compiled = root / "rfd3_input.json"
        compiled.write_text(
            json.dumps(
                {
                    "example": {
                        "extra": {
                            "motif_constraint_orbits": mobile_orbits
                            + [
                                {
                                    "constraint_orbit_id": "fixed_orbit",
                                    "coupling_group_id": "fixed_component",
                                    "mobility_mode": "fixed",
                                },
                            ]
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        result = root / "result.json"
        result.write_text(
            json.dumps(
                {
                    "motif_mobility_diagnostics": {
                        "apply_updates": active,
                        "update_calls": 4 if active else 0,
                        "active_window_calls": 3 if active else 0,
                        "conditioning_refresh_count": 3 if active else 1,
                        "mobile_orbit_count": mobile_count,
                        "symmetry_id": symmetry_id,
                        "symmetry_axis": {
                            "point": [0.0, 0.0, 0.0],
                            "direction": [0.0, 0.0, 1.0],
                            "transform_ids": list(range(group_action_count)),
                        },
                        "runtime_group_action_count": (observed_group_action_count),
                        "proposal_source": "scaffold_boundary",
                        "update_interval": 1,
                        "proposal_schedule": {
                            "total_diffusion_steps": 50,
                            "declared_update_interval": 5,
                            "effective_update_interval": 1,
                            "target_update_count": 24,
                            "scheduled_proposal_count": 50,
                            "scheduled_active_proposal_counts": [20]
                            * mobile_count,
                        },
                        "constraint_runtime": {
                            "schema_version": 1,
                            "state": "finalized",
                            "proposal_source": "scaffold_boundary",
                            "proposal_interval": 1,
                            "conditioning_refresh_count": (3 if active else 1),
                            "phase_counts": {
                                "initialize": 1,
                                "model_prediction": 5,
                                "proposal": 4 if active else 0,
                                "proposal_applied": 2 if active else 0,
                                "state_update": 5,
                                "post_guidance": 0,
                                "finalize": 1,
                            },
                        },
                        "orbits": [
                            {
                                "constraint_orbit_id": (f"mobile_orbit_{index}"),
                                "component_id": f"mobile_component_{index}",
                                "translation_norms": [translation],
                                "translation_vectors": [list(translation_vector)],
                                "template_master_centers": [[10.0, 0.0, 0.0]],
                                "rotation_degrees": [rotation],
                                "group_action_count": (observed_group_action_count),
                                "group_transform_ids": list(
                                    range(observed_group_action_count)
                                ),
                                "schedule": {
                                    "start_fraction": 0.05,
                                    "end_fraction": 0.85,
                                    "response": 0.4,
                                    "max_step_translation": 0.5,
                                    "max_step_rotation_degrees": 2.5,
                                },
                                "effective_pose_prior": {
                                    "translation_scale": 1.0,
                                    "rotation_scale_degrees": 5.0,
                                    "normalization": (
                                        "at_least_one_third_of_hard_bound"
                                    ),
                                },
                            }
                            for index in range(mobile_count)
                        ],
                        "trajectory": (
                            [
                                {
                                    "proposal_source": "scaffold_boundary",
                                    "atomic_joint_acceptance": atomic_joint,
                                    "progress": 0.5,
                                    "window_weight": 1.0,
                                    "accepted": active,
                                    "applied": active,
                                    "joint_energy_delta": -0.5,
                                    "initial_energy": {
                                        "total": 2.0,
                                        "junction": 1.0,
                                        "clash": 1.0,
                                    },
                                    "proposed_energy": {
                                        "total": 1.5,
                                        "junction": 0.75,
                                        "clash": 0.75,
                                    },
                                    "orbit_proposals": [
                                        {
                                            "constraint_orbit_id": (
                                                f"mobile_orbit_{index}"
                                            ),
                                            "component_id": (
                                                f"mobile_component_{index}"
                                            ),
                                            "active": True,
                                            "accepted": active,
                                            "committed": active,
                                            "proposed_delta_translation": [
                                                0.1,
                                                0.0,
                                                0.0,
                                            ],
                                            "proposed_delta_rotation_degrees": (0.5),
                                            "objective": {
                                                "initial": {
                                                    "total": 2.0,
                                                    "junction": 1.0,
                                                    "clash": 1.0,
                                                    "tilt": 0.0,
                                                    "prior": 0.0,
                                                },
                                                "proposed": {
                                                    "total": 1.5,
                                                    "junction": 0.75,
                                                    "clash": 0.75,
                                                    "tilt": 0.0,
                                                    "prior": 0.0,
                                                },
                                                "delta": {
                                                    "total": -0.5,
                                                    "junction": -0.25,
                                                    "clash": -0.25,
                                                    "tilt": 0.0,
                                                    "prior": 0.0,
                                                },
                                            },
                                        }
                                        for index in range(mobile_count)
                                    ],
                                }
                            ]
                            if mobile_count > 1
                            else []
                        ),
                    }
                }
            ),
            encoding="utf-8",
        )
        return compiled, result

    def test_accepts_active_motion_within_declared_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(Path(temporary))
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["summary"]["runtime_active"])
        self.assertEqual(
            report["summary"]["components"][0]["component_id"],
            "mobile_component_0",
        )
        self.assertTrue(report["summary"]["nonzero_motion_observed"])
        self.assertEqual(report["summary"]["applied_proposal_count"], 2)
        component = report["summary"]["components"][0]
        self.assertEqual(component["translation_fraction_of_bound"], 0.5)
        self.assertEqual(component["rotation_fraction_of_bound"], 0.4)
        self.assertEqual(component["scheduled_active_proposal_count"], 20)
        self.assertEqual(component["translation_search_budget_upper_bound"], 3.0)
        self.assertEqual(
            component["rotation_search_budget_upper_bound_degrees"],
            10.0,
        )
        self.assertEqual(
            report["summary"]["proposal_schedule"][
                "effective_update_interval"
            ],
            1,
        )

    def test_zero_motion_is_reported_without_falsely_failing_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                translation=0.0,
                rotation=0.0,
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        self.assertFalse(report["summary"]["nonzero_motion_observed"])
        self.assertEqual(
            report["summary"]["components"][0]["translation_fraction_of_bound"],
            0.0,
        )

    def test_rejects_runtime_motion_beyond_declared_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                translation=3.5,
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        component = report["summary"]["components"][0]
        self.assertFalse(component["passed"])
        self.assertEqual(component["maximum_translation_observed"], 3.5)

    def test_accepts_translation_only_radial_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                rotation=0.0,
                subspace="radial",
                rotation_bound=None,
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        component = report["summary"]["components"][0]
        self.assertEqual(component["mobility_subspace"], "radial")
        self.assertEqual(component["maximum_rotation_deg_allowed"], 0.0)
        self.assertTrue(component["directional_contract_valid"])

    def test_accepts_radial_rotation_without_directional_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                subspace="radial_rotation",
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        component = report["summary"]["components"][0]
        self.assertEqual(component["mobility_subspace"], "radial_rotation")
        self.assertTrue(component["directional_contract_valid"])
        self.assertEqual(
            component["maximum_tangential_translation_leakage"],
            0.0,
        )

    def test_rejects_radial_rotation_with_axial_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                translation=math.sqrt(2.0),
                translation_vector=(1.0, 0.0, 1.0),
                subspace="radial_rotation",
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        component = report["summary"]["components"][0]
        self.assertFalse(component["directional_contract_valid"])
        self.assertEqual(
            component["maximum_axial_translation_observed"],
            1.0,
        )

    def test_radial_axial_rotation_allows_axial_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                translation=math.sqrt(2.0),
                translation_vector=(1.0, 0.0, 1.0),
                subspace="radial_axial_rotation",
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        component = report["summary"]["components"][0]
        self.assertTrue(component["directional_contract_valid"])
        self.assertTrue(component["axial_translation_allowed"])
        self.assertEqual(
            component["maximum_tangential_translation_leakage"],
            0.0,
        )

    def test_rejects_declared_mobility_that_never_became_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                active=False,
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["runtime_active"])

    def test_rejects_missing_constraint_runtime_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(Path(temporary))
            payload = json.loads(result.read_text(encoding="utf-8"))
            del payload["motif_mobility_diagnostics"]["constraint_runtime"]
            result.write_text(json.dumps(payload), encoding="utf-8")

            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["constraint_runtime_valid"])

    def test_requires_atomic_joint_diagnostics_for_multiple_orbits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                mobile_count=2,
                atomic_joint=False,
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["atomic_joint_runtime"])

    def test_accepts_atomic_joint_diagnostics_for_multiple_orbits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                mobile_count=2,
                atomic_joint=True,
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["summary"]["atomic_joint_runtime"])
        self.assertEqual(
            report["summary"]["valid_joint_trajectory_steps"],
            1,
        )

    def test_accepts_complete_d3_group_action_orbits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                mobile_count=2,
                group_action_count=6,
                symmetry_id="D3",
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["summary"]["complete_group_action_orbits"])
        self.assertEqual(
            report["summary"]["runtime_group_action_count"],
            6,
        )
        self.assertTrue(
            all(
                component["runtime_group_action_count"] == 6
                for component in report["summary"]["components"]
            )
        )

    def test_rejects_truncated_d3_group_action_orbit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                mobile_count=2,
                group_action_count=6,
                observed_group_action_count=3,
                symmetry_id="D3",
            )
            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["summary"]["complete_group_action_orbits"])

    def test_rejects_nonfinite_multi_orbit_objective_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compiled, result = self._write_inputs(
                Path(temporary),
                mobile_count=2,
            )
            payload = json.loads(result.read_text(encoding="utf-8"))
            payload["motif_mobility_diagnostics"]["trajectory"][0]["orbit_proposals"][
                0
            ]["objective"]["delta"]["clash"] = float("nan")
            result.write_text(json.dumps(payload), encoding="utf-8")

            report = audit_component_mobility(
                compiled_input=compiled,
                result_json=result,
            )

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["summary"]["valid_joint_trajectory_steps"],
            0,
        )

    def test_writes_strict_standalone_trajectory_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, result = self._write_inputs(root, mobile_count=2)
            payload = json.loads(result.read_text(encoding="utf-8"))
            payload["motif_mobility_diagnostics"]["trajectory"][0][
                "minimum_distance"
            ] = float("inf")
            result.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "mobility_trajectory.json"

            written = write_mobility_trajectory(
                result_json=result,
                output=output,
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(written)
        self.assertEqual(
            artifact["artifact"],
            "rfd3_mosaic.mobility_trajectory",
        )
        self.assertEqual(
            artifact["constraint_runtime"]["phase_counts"]["finalize"],
            1,
        )
        self.assertIsNone(artifact["trajectory"][0]["minimum_distance"])


if __name__ == "__main__":
    unittest.main()
