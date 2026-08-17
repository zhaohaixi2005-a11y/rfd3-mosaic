import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.rfd3_cylindrical_audit import (
    audit_cylindrical_coordinates,
)


class CylindricalAuditTestCase(unittest.TestCase):
    def test_requires_finalized_exact_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiled = root / "input.json"
            result = root / "result.json"
            compiled.write_text(
                json.dumps(
                    {
                        "example": {
                            "extra": {
                                "cylindrical_constraints": [
                                    {
                                        "constraint_id": "radius_lock",
                                        "keep": ["radius"],
                                        "atom_keys": [["A1", "CA"]],
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result.write_text(
                json.dumps(
                    {
                        "constraint_runtime_diagnostics": {
                            "state": "finalized",
                            "cylindrical_projector_active": True,
                            "final_cylindrical_maximum_error": 2.0e-7,
                            "phase_counts": {
                                "initialize": 1,
                                "finalize": 1,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = audit_cylindrical_coordinates(
                compiled_input=compiled,
                result_json=result,
            )

            self.assertTrue(report["passed"])
            self.assertTrue(report["summary"]["runtime_active"])
            self.assertEqual(
                report["summary"]["constraint_ids"],
                ["radius_lock"],
            )

    def test_rejects_missing_runtime_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiled = root / "input.json"
            result = root / "result.json"
            compiled.write_text(
                json.dumps(
                    {
                        "example": {
                            "extra": {
                                "cylindrical_constraints": [
                                    {
                                        "constraint_id": "radius_lock",
                                        "keep": ["radius"],
                                        "atom_keys": [["A1", "CA"]],
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result.write_text("{}", encoding="utf-8")

            report = audit_cylindrical_coordinates(
                compiled_input=compiled,
                result_json=result,
            )

            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
