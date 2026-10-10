import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfd3_mosaic.rfd3_scaffold_audit import (
    _effective_chain_rg_limit,
    _evaluate_assembly_shape_contract,
    main,
)


class ScaffoldAuditPolicyTestCase(unittest.TestCase):
    def test_report_only_cli_retains_full_scaffold_failure_without_route_counts(self):
        from test_central_motif_audit import MMCIF_HEADER

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result.json"
            result.write_text("{}")
            result.with_suffix(".cif").write_text(MMCIF_HEADER +
                "ATOM C CA ALA A 1 1 A 0.0 0.0 0.0 1\n"
                "ATOM C CA ALA A 2 2 A 3.8 0.0 0.0 1\n#\n")
            output = root / "audit.json"
            contract = {"declared": True, "applicable": True, "required": True,
                        "passed": False, "measurement": "explicit_full_scaffold_contract"}
            with patch("sys.argv", ["audit", "--result-json", str(result), "--output", str(output), "--report-only"]), patch(
                "rfd3_mosaic.rfd3_scaffold_audit._audit_final_generated_route_ownership",
                return_value=contract,
            ), contextlib.redirect_stdout(io.StringIO()) as stdout:
                main()
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["summary"]["passed_scaffold_contract"])
            self.assertIn("explicit reference geometry", stdout.getvalue())

    def test_default_limit_is_preserved_for_compact_fixed_geometry(self) -> None:
        self.assertEqual(
            _effective_chain_rg_limit(
                explicit_limit=None,
                fixed_geometry_floor=18.0,
            ),
            25.0,
        )

    def test_fixed_geometry_sets_an_automatic_lower_bound(self) -> None:
        self.assertEqual(
            _effective_chain_rg_limit(
                explicit_limit=None,
                fixed_geometry_floor=25.327,
            ),
            27.327,
        )

    def test_explicit_limit_remains_authoritative(self) -> None:
        self.assertEqual(
            _effective_chain_rg_limit(
                explicit_limit=24.0,
                fixed_geometry_floor=40.0,
            ),
            24.0,
        )

    def test_invalid_limits_fail_closed(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _effective_chain_rg_limit(
                        explicit_limit=None,
                        fixed_geometry_floor=value,
                    )

    def test_final_shape_contract_checks_output_morphology(self) -> None:
        summary = {
            "assembly_spherical_outer_diameter": 72.0,
            "assembly_spherical_inner_diameter": 24.0,
            "assembly_outer_radial_diameter": None,
            "assembly_central_pore_diameter": None,
        }
        shape = {
            "diameter_angstrom": {"minimum": 70.0, "maximum": 80.0},
            "cavity_diameter_angstrom": {
                "minimum": 20.0,
                "maximum": 30.0,
            },
        }

        report = _evaluate_assembly_shape_contract(summary, shape)

        self.assertTrue(report["declared"])
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["checks"]), 2)

    def test_missing_required_morphology_fails_closed(self) -> None:
        report = _evaluate_assembly_shape_contract(
            {},
            {
                "diameter_angstrom": {
                    "minimum": 70.0,
                    "maximum": 80.0,
                }
            },
        )

        self.assertFalse(report["passed"])
        self.assertIsNone(report["checks"][0]["observed"])


if __name__ == "__main__":
    unittest.main()
