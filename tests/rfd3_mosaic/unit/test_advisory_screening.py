import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from rfd3_mosaic.advisory_screening import (
    build_advisory_screening,
    write_advisory_screening,
)
from rfd3_mosaic.schema import UserDesignSpec


class AdvisoryScreeningTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def report(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_interface_quality_is_advisory_not_a_contract_failure(self) -> None:
        interface = self.report(
            "graph_interface_guidance_audit.json",
            {"passed": False, "summary": {"coverage": 0}},
        )
        scaffold = self.report(
            "scaffold_validity_audit.json",
            {
                "passed": False,
                "summary": {
                    "passed_continuity": True,
                    "passed_symmetry": True,
                    "passed_clashes": False,
                    "passed_compactness": True,
                },
            },
        )

        result = build_advisory_screening((interface, scaffold))

        self.assertEqual(result["contract_status"], "met")
        self.assertEqual(
            result["recommendation"],
            "review_advisory_metrics",
        )
        self.assertTrue(result["generated_output_retained"])
        self.assertFalse(result["contract_flags"])
        self.assertEqual(len(result["advisory_flags"]), 2)

    def test_exact_geometry_and_chain_continuity_are_contracts(self) -> None:
        constraint = self.report(
            "constraint_orbit_audit.json",
            {"passed": False},
        )
        scaffold = self.report(
            "scaffold_validity_audit.json",
            {
                "passed": False,
                "summary": {
                    "passed_continuity": False,
                    "passed_symmetry": True,
                    "passed_clashes": True,
                    "passed_compactness": True,
                },
            },
        )

        result = build_advisory_screening((constraint, scaffold))

        self.assertEqual(result["contract_status"], "flagged")
        self.assertEqual(result["recommendation"], "review_contract")
        self.assertEqual(len(result["contract_flags"]), 2)
        self.assertTrue(result["generated_output_retained"])

    def test_peptide_geometry_is_backbone_advice_not_execution_contract(self) -> None:
        scaffold = self.report(
            "scaffold_validity_audit.json",
            {
                "passed": True,
                "summary": {
                    "passed_backbone_atom_completeness": True,
                    "passed_continuity": True,
                    "passed_symmetry": True,
                    "passed_clashes": True,
                    "passed_compactness": True,
                    "passed_peptide_geometry": False,
                },
            },
        )

        result = build_advisory_screening((scaffold,))

        self.assertEqual(result["contract_status"], "met")
        self.assertEqual(result["recommendation"], "review_advisory_metrics")
        self.assertFalse(result["contract_flags"])
        self.assertEqual(
            result["advisory_flags"][0]["code"],
            "advisory.scaffold.passed_peptide_geometry",
        )

    def test_writes_self_describing_advice_without_removing_reports(self) -> None:
        report = self.report("constraint_orbit_audit.json", {"passed": True})
        output = self.root / "screening_advice.json"

        result = write_advisory_screening(
            output,
            (report,),
            protocol="generic_backbone",
        )

        self.assertTrue(output.is_file())
        self.assertTrue(report.is_file())
        self.assertEqual(
            result["recommendation"],
            "recommended_for_next_stage",
        )
        self.assertIn("backbone-only", result["interpretation"])

    def test_schema_defaults_to_non_destructive_advice(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "screening-default",
                "input": "motif.pdb",
                "symmetry": "C3",
            }
        )

        self.assertEqual(design.sampling.screening.mode, "advisory")
        self.assertTrue(design.sampling.screening.retain_all_outputs)

    def test_schema_rejects_destructive_screening(self) -> None:
        with self.assertRaises(ValidationError):
            UserDesignSpec.model_validate(
                {
                    "name": "screening-delete",
                    "input": "motif.pdb",
                    "symmetry": "C3",
                    "sampling": {"screening": {"retain_all_outputs": False}},
                }
            )


if __name__ == "__main__":
    unittest.main()
