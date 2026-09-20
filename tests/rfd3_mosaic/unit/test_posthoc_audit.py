import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rfd3_mosaic.assembly_compiler import CompiledAudit
from rfd3_mosaic.cli import _parser
from rfd3_mosaic.posthoc_audit import (
    _materialize_result_compiled_input,
    audit_existing_run,
)
from rfd3_mosaic.result_auditing import (
    ResultAuditOutcome,
    find_result_json,
    find_result_jsons,
    generation_completeness,
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
        self.result.with_suffix(".cif").write_text("data_fixture\n")
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
                "motif_constraint_orbits": [{"mobility_mode": "orbit_rigid"}],
                "assembly_interface_relations": [
                    {
                        "required": True,
                        "satisfaction_stage": "output",
                        "target_geometry": {"mode": "geometric_constraints"},
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

    def test_failed_early_multi_input_run_recovers_one_example_per_result(
        self,
    ) -> None:
        example_zero = "design_00000_pose_00000_rep_000"
        example_one = "design_00001_pose_00000_rep_001"
        self.input.write_text(
            json.dumps(
                {
                    example_zero: {
                        "input": "/frozen/pose.cif",
                        "extra": {"symmetry_multiplicity": 3},
                    },
                    example_one: {
                        "input": "/frozen/pose.cif",
                        "extra": {"symmetry_multiplicity": 3},
                    },
                }
            ),
            encoding="utf-8",
        )
        result = self.run / f"rfd3_input_{example_one}_0_model_0.json"
        result.write_text("{}\n", encoding="utf-8")

        recovered = _materialize_result_compiled_input(
            merged_input=self.input,
            result_json=result,
            run_directory=self.run,
        )

        payload = json.loads(recovered.read_text(encoding="utf-8"))
        self.assertEqual(tuple(payload), (example_one,))
        self.assertTrue(recovered.is_relative_to(self.run / "input"))

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
                    "produced_designs": 1,
                    "contract_met_designs": 0,
                    "contract_flagged_designs": 1,
                    "recommended_designs": 0,
                    "review_designs": 1,
                    "design_results": [
                        {
                            "design_index": 0,
                            "design_id": "result",
                            "result_json": str(self.result),
                            "generated": True,
                            "contract_met": False,
                            "recommendation": "review_contract",
                        }
                    ],
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
            (self.run / "experiment_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "completed")
        self.assertNotIn("error", summary)
        self.assertFalse(summary["posthoc_audit"]["inference_rerun"])
        self.assertEqual(summary["contract_met_designs"], 1)
        self.assertEqual(summary["contract_flagged_designs"], 0)
        self.assertEqual(summary["recommended_designs"], 1)
        self.assertEqual(summary["review_designs"], 0)
        self.assertTrue(summary["design_results"][0]["contract_met"])
        self.assertEqual(
            summary["design_results"][0]["recommendation"],
            "recommended_for_next_stage",
        )
        self.assertEqual(
            summary["posthoc_audit"]["previous_error"],
            "old audit could not map chains",
        )

    def test_flagged_gate_preserves_completed_generated_run(self) -> None:
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
            (self.run / "experiment_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "completed")
        self.assertNotIn("error", summary)
        self.assertEqual(
            summary["posthoc_audit"]["check_flags"],
            ["required audit failed"],
        )
        self.assertTrue(summary["posthoc_audit"]["execution_completed"])
        self.assertIsNone(result.error)
        self.assertEqual(summary["reports"], [str(path) for path in reports])

    def test_cli_accepts_numeric_audit_target_and_root(self) -> None:
        arguments = _parser().parse_args(
            [
                "audit",
                "12345",
                "--root",
                str(self.run.parent),
                "--reuse-reports",
            ]
        )
        self.assertEqual(arguments.command, "audit")
        self.assertEqual(arguments.target, "12345")
        self.assertEqual(arguments.root, self.run.parent)
        self.assertTrue(arguments.reuse_reports)

    def test_reuses_existing_reports_without_running_audit_commands(self) -> None:
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
            report.write_text('{"passed": true}\n', encoding="utf-8")

        with (
            patch("rfd3_mosaic.posthoc_audit.run_result_audits") as runner,
            patch("rfd3_mosaic.posthoc_audit._update_index"),
        ):
            result = audit_existing_run(self.run, reuse_reports=True)

        self.assertTrue(result.passed)
        runner.assert_not_called()
        summary = json.loads(
            (self.run / "experiment_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(summary["posthoc_audit"]["reports_reused"])
        self.assertEqual(summary["contract_met_designs"], 1)

    def test_audit_execution_failure_preserves_completed_run_state(self) -> None:
        self._write_compiled_input(
            {
                "symmetry_multiplicity": 3,
                "motif_constraint_orbits": [],
                "assembly_interface_relations": [],
            }
        )
        original_report = self.run / "scaffold_validity_audit.json"
        original_report.write_text('{"passed": true}\n', encoding="utf-8")
        (self.run / "experiment_summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "experiment": "posthoc-test",
                    "produced_designs": 1,
                    "contract_met_designs": 1,
                    "contract_flagged_designs": 0,
                    "reports": [str(original_report)],
                }
            ),
            encoding="utf-8",
        )

        with (
            patch(
                "rfd3_mosaic.posthoc_audit.run_result_audits",
                side_effect=RuntimeError("audit process was killed"),
            ),
            patch("rfd3_mosaic.posthoc_audit._update_index") as update_index,
        ):
            result = audit_existing_run(self.run)

        self.assertFalse(result.passed)
        self.assertEqual(result.error, "audit process was killed")
        summary = json.loads(
            (self.run / "experiment_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["contract_met_designs"], 1)
        self.assertEqual(summary["reports"], [str(original_report)])
        self.assertNotIn("error", summary)
        self.assertEqual(
            summary["posthoc_audit"]["audit_error"],
            "audit process was killed",
        )
        self.assertEqual(update_index.call_args.kwargs["state"], "completed")
        self.assertIsNone(update_index.call_args.kwargs["error"])

    def _frozen_assignments(self, count: int) -> list[str]:
        assignments = [
            {
                "design_index": i,
                "pose_index": 0,
                "replicate_index": i,
                "example_id": f"design_{i:05d}_pose_00000_rep_{i:03d}",
            }
            for i in range(count)
        ]
        (self.run / "sampling_manifest.json").write_text(
            json.dumps({"assignments": assignments})
        )
        config = yaml.safe_load((self.run / "resolved_config.yaml").read_text())
        config["sampling"] = {"designs": count}
        (self.run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        return [item["example_id"] for item in assignments]

    def test_same_count_wrong_or_duplicate_identity_is_incomplete(self) -> None:
        names = self._frozen_assignments(2)
        config = {"sampling": {"designs": 2}}
        for results in (
            (self.run / f"input_{names[0]}_0_model_0.json", self.result),
            (
                self.run / f"input_{names[0]}_0_model_0.json",
                self.run / f"other_{names[0]}_0_model_0.json",
            ),
        ):
            report = generation_completeness(
                self.run, results, resolved_config=config, previous_summary={}
            )
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_example_ids"], [names[1]])
        results = tuple(self.run / f"input_{name}_0_model_0.json" for name in names)
        for result in results:
            result.write_text("{}")
            result.with_suffix(".cif").write_text("data_fixture\n")
        report = generation_completeness(
            self.run, results, resolved_config=config, previous_summary={}
        )
        self.assertTrue(report["complete"])
        self.assertTrue(report["identity_verified"])

    def test_failed_generation_ledger_cannot_be_overridden_by_existing_files(
        self,
    ) -> None:
        names = self._frozen_assignments(2)
        results = tuple(self.run / f"input_{name}_0_model_0.json" for name in names)
        (self.run / "design_outcomes.json").write_text(
            json.dumps(
                {
                    "expected_example_ids": names,
                    "designs": {
                        names[0]: {"generation_status": "generated"},
                        names[1]: {"generation_status": "failed"},
                    },
                }
            )
        )
        report = generation_completeness(
            self.run,
            results,
            resolved_config={"sampling": {"designs": 2}},
            previous_summary={},
        )
        self.assertFalse(report["complete"])
        self.assertEqual(report["incomplete_outcome_ids"], [names[1]])

    def test_successful_partial_reaudit_preserves_generation_failure(self) -> None:
        self._write_compiled_input({"symmetry_multiplicity": 3})
        names = self._frozen_assignments(3)
        self.result.rename(self.run / f"input_{names[0]}_0_model_0.json")
        (self.run / "experiment_summary.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error": "inference stopped",
                    "requested_designs": 3,
                }
            )
        )
        for name in ("constraint_orbit_audit.json", "scaffold_validity_audit.json"):
            (self.run / name).write_text('{"passed": true}\n')
        with patch("rfd3_mosaic.posthoc_audit._update_index"):
            result = audit_existing_run(self.run, reuse_reports=True)
        summary = json.loads((self.run / "experiment_summary.json").read_text())
        self.assertFalse(result.passed)
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["execution_completed"])
        self.assertEqual(summary["error"], "inference stopped")
        self.assertTrue(summary["posthoc_audit"]["execution_completed"])
        self.assertEqual(
            summary["generation_completeness"]["missing_example_ids"], names[1:]
        )

    def test_complete_frozen_cohort_recovers_only_audit_failure(self) -> None:
        self._write_compiled_input({"symmetry_multiplicity": 3})
        names = self._frozen_assignments(1)
        self.result.rename(self.run / f"input_{names[0]}_0_model_0.json")
        self.result.with_suffix(".cif").rename(self.run / f"input_{names[0]}_0_model_0.cif")
        (self.run / "experiment_summary.json").write_text(
            json.dumps(
                {
                    "status": "partial",
                    "error_type": "RuntimeError",
                    "error": "audit interrupted",
                    "requested_designs": 1,
                }
            )
        )
        (self.run / "design_outcomes.json").write_text(
            json.dumps(
                {
                    "expected_example_ids": names,
                    "designs": {
                        names[0]: {
                            "generation_status": "generated",
                            "audit_status": "failed",
                            "audit_error": "audit interrupted",
                        }
                    },
                    "status": "completed",
                }
            )
        )
        for name in ("constraint_orbit_audit.json", "scaffold_validity_audit.json"):
            (self.run / name).write_text('{"passed": true}\n')
        with patch("rfd3_mosaic.posthoc_audit._update_index"):
            result = audit_existing_run(self.run, reuse_reports=True)
        summary = json.loads((self.run / "experiment_summary.json").read_text())
        self.assertTrue(result.passed)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["execution_completed"])
        self.assertEqual(summary["failure_history"][0]["error"], "audit interrupted")
        self.assertNotIn("error", summary)
        ledger = json.loads((self.run / "design_outcomes.json").read_text())
        self.assertEqual(ledger["status"], "completed")
        self.assertEqual(ledger["designs"][names[0]]["audit_status"], "completed")
        self.assertEqual(
            ledger["designs"][names[0]]["audit_failure_history"], ["audit interrupted"]
        )

    def test_current_missing_structure_cannot_reuse_cached_pass_as_completed(self):
        self._write_compiled_input({"symmetry_multiplicity": 3})
        self.result.with_suffix(".cif").unlink()
        (self.run / "experiment_summary.json").write_text('{"status":"completed"}')
        for name in ("constraint_orbit_audit.json", "scaffold_validity_audit.json"):
            (self.run / name).write_text('{"passed": true}')
        with patch("rfd3_mosaic.posthoc_audit._update_index"):
            outcome = audit_existing_run(self.run, reuse_reports=True)
        summary = json.loads((self.run / "experiment_summary.json").read_text())
        self.assertFalse(outcome.passed)
        self.assertEqual(summary["status"], "partial")
        self.assertIn(self.result.name, summary["generation_completeness"]["artifact_errors"])

    def test_generation_process_failure_cannot_be_erased_by_complete_file_set(
        self,
    ) -> None:
        names = self._frozen_assignments(1)
        ledger = {
            "expected_example_ids": names,
            "status": "completed",
            "designs": {names[0]: {"generation_status": "generated"}},
        }
        for extra in (
            {"inference_error": "worker exit 1"},
            {"ledger_error": "invalid native receipt"},
            {"status": "running"},
            {"status": None},
        ):
            (self.run / "design_outcomes.json").write_text(
                json.dumps({**ledger, **extra})
            )
            report = generation_completeness(
                self.run,
                (self.run / f"input_{names[0]}_0_model_0.json",),
                resolved_config={"sampling": {"designs": 1}},
                previous_summary={},
            )
            self.assertFalse(report["complete"])
            self.assertTrue(report["ledger_errors"])

    def test_reaudit_refreshes_explanation_payload_and_evidence_hash(self) -> None:
        self._write_compiled_input({"symmetry_multiplicity": 3})
        reports = [
            self.run / name
            for name in ("constraint_orbit_audit.json", "scaffold_validity_audit.json")
        ]
        for report in reports:
            report.write_text('{"passed": true}\n')
        with patch("rfd3_mosaic.posthoc_audit._update_index"):
            audit_existing_run(self.run, reuse_reports=True)
            reports[1].write_text(
                '{"passed": false, "summary": {"passed_continuity": false}}\n'
            )
            audit_existing_run(self.run, reuse_reports=True)
        explanation = json.loads((self.run / "decision_explanation.json").read_text())
        self.assertEqual(explanation["screening"]["contract_status"], "flagged")
        snapshot = next(
            item for item in explanation["audits"] if item["path"] == reports[1].name
        )
        self.assertEqual(
            snapshot["sha256"], hashlib.sha256(reports[1].read_bytes()).hexdigest()
        )
        self.assertFalse(snapshot["payload"]["summary"]["passed_continuity"])
        self.assertIn("`flagged`", (self.run / "decision_explanation.md").read_text())


if __name__ == "__main__":
    unittest.main()
