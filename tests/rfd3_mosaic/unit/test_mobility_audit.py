import json
from pathlib import Path
import tempfile
import unittest

from rfd3_mosaic.rfd3_mobility_audit import audit_component_mobility


class ComponentMobilityAuditTestCase(unittest.TestCase):
    def _write_inputs(
        self,
        root: Path,
        *,
        translation: float = 1.5,
        rotation: float = 4.0,
        active: bool = True,
    ) -> tuple[Path, Path]:
        compiled = root / "rfd3_input.json"
        compiled.write_text(
            json.dumps(
                {
                    "example": {
                        "extra": {
                            "motif_constraint_orbits": [
                                {
                                    "constraint_orbit_id": "mobile_orbit",
                                    "coupling_group_id": "mobile_component",
                                    "mobility_mode": "orbit_rigid",
                                    "max_translation": 3.0,
                                    "max_rotation_deg": 10.0,
                                },
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
                        "conditioning_refresh_count": 3 if active else 0,
                        "mobile_orbit_count": 1,
                        "orbits": [
                            {
                                "translation_norms": [translation],
                                "rotation_degrees": [rotation],
                            }
                        ],
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
            "mobile_component",
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


if __name__ == "__main__":
    unittest.main()
