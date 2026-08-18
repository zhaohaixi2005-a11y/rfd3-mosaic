import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from rfd3_mosaic.assembly_compiler import CompiledAudit
from rfd3_mosaic.cli import _parser
from rfd3_mosaic.posthoc_audit import audit_existing_run
from rfd3_mosaic.result_auditing import (
    ResultAuditOutcome,
    find_result_json,
    find_result_jsons,
    infer_existing_run_audits,
    run_result_audits,
)


class PosthocAuditTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name) / "12345"
        (self.run / "input").mkdir(parents=True)
        self.input = self.run / "input" / "rfd3_input.json"
        self.result = self.run / "result_model_0.json"
        self.result.write_text("{}\n", encoding="utf-8")
        (self.run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "posthoc-test",
                    "topology": {"kind": "user_design"},
                    "output": {
                        "root": str(self.run.parent),
                        "campaign": "tests",
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_compiled_input(self, extra: dict) -> None:
        self.input.write_text(
            json.dumps({"example": {"extra": extra}}) + "\n",
            encoding="utf-8",
        )

    def test_infers_complete_user_design_audit_set_from_frozen_input(self) -> None:
        self._write_compiled_input(
            {
                "symmetry_multiplicity": 3,
                "motif_constraint_orbits": [
                    {"mobility_mode": "orbit_rigid"}
                ],
                "assembly_interface_relations": [
                    {
                        "required": True,
                        "satisfaction_stage": "output",
                        "target_geometry": {
                            "mode": "geometric_constraints"
                        },
                    }
                ],
            }
        )

        audits = infer_existing_run_audits(
            run_directory=self.run,
            rfd3_input=self.input,
            resolved_config={"topology": {"kind": "user_design"}},
        )

        self.assertEqual(
            [audit.report_name for audit in audits],
            [
                "constraint_orbit_audit.json",
                "assembly_interface_relation_audit.json",
                "graph_interface_guidance_audit.json",
                "component_mobility_audit.json",
            ],
        )

    def test_legacy_interface_audit_uses_frozen_mapping_and_spec(self) -> None:
        self._write_compiled_input({"symmetry_multiplicity": 3})
        mapping = self.run / "input" / "mapping.json"
        mapping.write_text("{}\n", encoding="utf-8")
        specification = self.run / "input" / "assembly_specification.yaml"
        specification.write_text(
            "interface_seed:\n  schema_version: 1\n",
            encoding="utf-8",
        )

        audits = infer_existing_run_audits(
            run_directory=self.run,
            rfd3_input=self.input,
            resolved_config={
                "topology": {"kind": "interface_seed"},
                "project_directory": str(self.run),
            },
        )

        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0].report_name, "seed_integrity_audit.json")
        self.assertIn(
            ("--adapter-mapping", str(mapping)),
            audits[0].input_arguments,
        )
        self.assertIn(
            ("--config", str(specification)),
            audits[0].input_arguments,
        )

    def test_shared_runner_never_invokes_inference(self) -> None:
        self._write_compiled_input({"symmetry_multiplicity": 3})
        commands: list[list[str]] = []

        def fake_run(command: list[str]) -> None:
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            output.write_text('{"passed": true}\n', encoding="utf-8")

        audit = CompiledAudit(
            module="rfd3_mosaic.rfd3_constraint_orbit_audit",
            report_name="constraint_orbit_audit.json",
            input_arguments=(("--compiled-input", str(self.input)),),
        )
        with patch(
            "rfd3_mosaic.result_auditing.write_mobility_trajectory",
            return_value=False,
        ):
            outcome = run_result_audits(
                run_directory=self.run,
                rfd3_input=self.input,
                result_json=self.result,
                semantic_audits=(audit,),
                python="python-test",
                command_runner=fake_run,
            )

        self.assertEqual(len(outcome.reports), 2)
        self.assertTrue(all(path.is_file() for path in outcome.reports))
        rendered = " ".join(" ".join(command) for command in commands)
        self.assertIn("rfd3_constraint_orbit_audit", rendered)
        self.assertIn("rfd3_scaffold_audit", rendered)
        self.assertNotIn("rfd3.run_inference", rendered)

    def test_multi_design_results_are_discovered_and_audited_separately(
        self,
    ) -> None:
        second = self.run / "result_1_model_0.json"
        second.write_text("{}\n", encoding="utf-8")

        results = find_result_jsons(self.run)

        self.assertEqual(results, (second, self.result))
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            find_result_json(self.run)

        self._write_compiled_input({"symmetry_multiplicity": 3})
        commands: list[list[str]] = []

        def fake_run(command: list[str]) -> None:
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            output.write_text('{"passed": true}\n', encoding="utf-8")

        audit = CompiledAudit(
            module="rfd3_mosaic.rfd3_constraint_orbit_audit",
            report_name="constraint_orbit_audit.json",
            input_arguments=(("--compiled-input", str(self.input)),),
        )
        output = self.run / "audits" / "result_1"
        with patch(
            "rfd3_mosaic.result_auditing.write_mobility_trajectory",
            return_value=False,
        ):
            outcome = run_result_audits(
                run_directory=self.run,
                rfd3_input=self.input,
                result_json=second,
                semantic_audits=(audit,),
                output_directory=output,
                python="python-test",
                command_runner=fake_run,
            )

        self.assertTrue(all(path.parent == output for path in outcome.reports))

    def test_successful_reaudit_replaces_failed_worker_verdict(self) -> None:
        self._write_compiled_input(
            {
                "symmetry_multiplicity": 3,
                "motif_constraint_orbits": [],
                "assembly_interface_relations": [],
            }
        )
        (self.run / "experiment_summary.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "experiment": "posthoc-test",
                    "error_type": "ValueError",
                    "error": "old audit could not map chains",
                }
            ),
            encoding="utf-8",
        )
        reports = (
            self.run / "constraint_orbit_audit.json",
            self.run / "scaffold_validity_audit.json",
        )
        for report in reports:
            report.write_text('{"passed": true}\n', encoding="utf-8")
        outcome = ResultAuditOutcome(reports=reports, mobility_trajectory=None)

        with (
            patch(
                "rfd3_mosaic.posthoc_audit.run_result_audits",
                return_value=outcome,
            ),
            patch("rfd3_mosaic.posthoc_audit.gate_result_audits"),
            patch("rfd3_mosaic.posthoc_audit._update_index"),
        ):
            result = audit_existing_run(self.run)

        self.assertTrue(result.passed)
        summary = json.loads(
            (self.run / "experiment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "completed")
        self.assertNotIn("error", summary)
        self.assertFalse(summary["posthoc_audit"]["inference_rerun"])
        self.assertEqual(
            summary["posthoc_audit"]["previous_error"],
            "old audit could not map chains",
        )

    def test_failed_gate_is_persisted_fail_closed(self) -> None:
        self._write_compiled_input(
            {
                "symmetry_multiplicity": 3,
                "motif_constraint_orbits": [],
                "assembly_interface_relations": [],
            }
        )
        reports = (
            self.run / "constraint_orbit_audit.json",
            self.run / "scaffold_validity_audit.json",
        )
        for report in reports:
            report.write_text('{"passed": false}\n', encoding="utf-8")
        outcome = ResultAuditOutcome(reports=reports, mobility_trajectory=None)

        with (
            patch(
                "rfd3_mosaic.posthoc_audit.run_result_audits",
                return_value=outcome,
            ),
            patch(
                "rfd3_mosaic.posthoc_audit.gate_result_audits",
                side_effect=RuntimeError("required audit failed"),
            ),
            patch("rfd3_mosaic.posthoc_audit._update_index"),
        ):
            result = audit_existing_run(self.run)

        self.assertFalse(result.passed)
        summary = json.loads(
            (self.run / "experiment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["error"], "required audit failed")
        self.assertEqual(summary["reports"], [str(path) for path in reports])

    def test_cli_accepts_numeric_audit_target_and_root(self) -> None:
        arguments = _parser().parse_args(
            ["audit", "12345", "--root", str(self.run.parent)]
        )
        self.assertEqual(arguments.command, "audit")
        self.assertEqual(arguments.target, "12345")
        self.assertEqual(arguments.root, self.run.parent)


if __name__ == "__main__":
    unittest.main()
