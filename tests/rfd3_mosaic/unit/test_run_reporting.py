import json
import tempfile
import unittest
from pathlib import Path

import yaml

from rfd3_mosaic.run_reporting import (
    RunReference,
    collect_run_status,
    format_status_text,
    resolve_run_reference,
    write_report,
)


class RunReportingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _completed_run(self, job_id: str = "12345") -> Path:
        run = self.root / "campaign" / "design" / job_id
        run.mkdir(parents=True)
        constraint = run / "constraint_orbit_audit.json"
        scaffold = run / "scaffold_validity_audit.json"
        self._write_json(
            constraint,
            {"passed": True, "summary": {"joint_orbit_rmsd": 1e-5}},
        )
        self._write_json(
            scaffold,
            {"passed": True, "summary": {"ca_clash_count": 0}},
        )
        self._write_json(
            run / "rfd3_prevalidation.json",
            {"status": "passed"},
        )
        structure = run / "result_model_0.cif.gz"
        structure.write_bytes(b"structure")
        self._write_json(
            run / "experiment_summary.json",
            {
                "status": "completed",
                "experiment": "design",
                "reports": [str(constraint), str(scaffold)],
                "result_json": str(run / "result_model_0.json"),
            },
        )
        return run

    def test_completed_run_requires_every_declared_audit(self) -> None:
        run = self._completed_run()

        status = collect_run_status(
            RunReference(job_id="12345", run_directory=run),
            include_scheduler=False,
        )

        self.assertEqual(status["state"], "completed")
        self.assertTrue(status["passed"])
        self.assertEqual(len(status["audits"]), 3)
        self.assertEqual(len(status["artifacts"]["structures"]), 1)
        self.assertIn("result:     GENERATED", format_status_text(status))

        self._write_json(
            run / "scaffold_validity_audit.json",
            {"passed": False, "summary": {"ca_clash_count": 3}},
        )
        failed = collect_run_status(
            RunReference(job_id="12345", run_directory=run),
            include_scheduler=False,
        )
        self.assertFalse(failed["passed"])

    def test_relocated_run_resolves_frozen_absolute_audit_paths(self) -> None:
        source = self._completed_run("24681")
        target = self.root / "2026-08-19" / "design" / "24681"
        target.parent.mkdir(parents=True)
        source.rename(target)

        status = collect_run_status(
            RunReference(job_id="24681", run_directory=target),
            include_scheduler=False,
        )

        self.assertTrue(status["passed"])
        self.assertTrue(
            all(
                Path(audit["path"]).is_relative_to(target) for audit in status["audits"]
            )
        )

    def test_failed_worker_reports_failure_without_scheduler(self) -> None:
        run = self.root / "campaign" / "design" / "999"
        run.mkdir(parents=True)
        self._write_json(
            run / "experiment_summary.json",
            {
                "status": "failed",
                "experiment": "design",
                "error_type": "ValueError",
                "error": "semantic gate failed",
            },
        )

        status = collect_run_status(
            RunReference(job_id="999", run_directory=run),
            include_scheduler=False,
        )

        self.assertEqual(status["state"], "failed")
        self.assertFalse(status["passed"])
        self.assertIn("semantic gate failed", format_status_text(status))

    def test_multi_design_run_reports_partial_scientific_yield(self) -> None:
        run = self._completed_run("13579")
        summary_path = run / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "requested_designs": 4,
                "produced_designs": 4,
                "accepted_designs": 3,
                "rejected_designs": 1,
            }
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        status = collect_run_status(
            RunReference(job_id="13579", run_directory=run),
            include_scheduler=False,
        )

        text = format_status_text(status)
        self.assertIn("result:     GENERATED", text)
        self.assertIn("execution:   COMPLETED", text)
        self.assertIn(
            "generated=4 contract_met=3 contract_flagged=1",
            text,
        )

    def test_completed_rejected_run_is_not_reported_as_execution_failure(self) -> None:
        run = self._completed_run("13580")
        summary_path = run / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "requested_designs": 1,
                "produced_designs": 1,
                "accepted_designs": 0,
                "rejected_designs": 1,
            }
        )
        self._write_json(summary_path, summary)

        status = collect_run_status(
            RunReference(job_id="13580", run_directory=run),
            include_scheduler=False,
        )

        text = format_status_text(status)
        self.assertEqual(status["state"], "completed")
        self.assertIn("result:     GENERATED", text)
        self.assertIn("execution:   COMPLETED", text)

    def test_numeric_job_id_resolves_completed_run(self) -> None:
        run = self._completed_run("24680")

        reference = resolve_run_reference("24680", root=self.root)

        self.assertEqual(reference.job_id, "24680")
        self.assertEqual(reference.run_directory, run.resolve())

    def test_pending_submission_receipt_resolves_without_run_directory(self) -> None:
        submission = self.root / "runs" / "campaign" / "_submissions" / "x"
        submission.mkdir(parents=True)
        self._write_json(
            submission / "submission.json",
            {"job_id": "777", "sbatch_output": "Submitted batch job 777"},
        )
        (submission / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "pending-design",
                    "output": {
                        "root": str(self.root / "runs"),
                        "campaign": "campaign",
                    },
                }
            ),
            encoding="utf-8",
        )

        reference = resolve_run_reference(submission)
        status = collect_run_status(reference, include_scheduler=False)

        self.assertEqual(reference.job_id, "777")
        self.assertIsNone(reference.run_directory)
        self.assertEqual(status["state"], "submitted")
        self.assertIsNone(status["passed"])

    def test_report_writes_html_and_canonical_json(self) -> None:
        run = self._completed_run("13579")
        status = collect_run_status(
            RunReference(job_id="13579", run_directory=run),
            include_scheduler=False,
        )
        status["experiment"] = "design <unsafe>"

        output = write_report(status)

        self.assertEqual(output, run / "mosaic_report.html")
        html = output.read_text(encoding="utf-8")
        self.assertIn("design &lt;unsafe&gt;", html)
        self.assertIn("joint_orbit_rmsd", html)
        assessment = output.with_suffix(".txt")
        self.assertTrue(assessment.is_file())
        self.assertIn("result:     GENERATED", assessment.read_text())
        payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"])

    def test_status_exposes_final_graph_packing_metrics(self) -> None:
        run = self._completed_run("97531")
        guidance = run / "graph_interface_guidance_audit.json"
        self._write_json(
            guidance,
            {
                "passed": True,
                "summary": {
                    "final_packing_metrics": {
                        "energy": 1.25,
                        "orientation": 0.1,
                        "shape": 0.2,
                        "minimum_edge_distance": 4.0,
                    }
                },
            },
        )
        summary_path = run / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reports"].append(str(guidance))
        self._write_json(summary_path, summary)

        status = collect_run_status(
            RunReference(job_id="97531", run_directory=run),
            include_scheduler=False,
        )
        text = format_status_text(status)

        self.assertIn("packing energy=1.25", text)
        self.assertIn("orientation=0.1", text)

    def test_status_exposes_core_quality_and_actual_mobility(self) -> None:
        run = self._completed_run("97532")
        core = run / "scaffold_core_guidance_audit.json"
        mobility = run / "component_mobility_audit.json"
        self._write_json(
            core,
            {
                "passed": True,
                "summary": {
                    "scientific_quality_satisfied": False,
                    "final_metrics": {
                        "mean_normalized_rg": 2.9,
                        "mean_tertiary_support_fraction": 0.4,
                        "long_range_contacts": 0.3,
                    },
                },
            },
        )
        self._write_json(
            mobility,
            {
                "passed": True,
                "summary": {
                    "applied_proposal_count": 7,
                    "nonzero_motion_observed": True,
                    "components": [
                        {
                            "translation_fraction_of_bound": 0.25,
                            "rotation_fraction_of_bound": 0.5,
                        }
                    ],
                },
            },
        )
        summary_path = run / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reports"].extend([str(core), str(mobility)])
        self._write_json(summary_path, summary)

        status = collect_run_status(
            RunReference(job_id="97532", run_directory=run),
            include_scheduler=False,
        )
        text = format_status_text(status)

        self.assertIn("normalized_rg=2.9", text)
        self.assertIn("declared_target_met=False", text)
        self.assertIn(
            "internal monomer-core controller references were not reached",
            text,
        )
        self.assertIn("mobility applied=7 moved=True", text)
        self.assertIn("rotation_bound_used=0.5", text)

    def test_copied_run_resolves_absolute_report_paths_by_name(self) -> None:
        run = self._completed_run("86420")
        summary_path = run / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reports"] = [
            "/unavailable/original/constraint_orbit_audit.json",
            "/unavailable/original/scaffold_validity_audit.json",
        ]
        self._write_json(summary_path, summary)

        status = collect_run_status(
            RunReference(job_id="86420", run_directory=run),
            include_scheduler=False,
        )

        self.assertTrue(status["passed"])
        self.assertTrue(
            all(Path(item["path"]).parent == run for item in status["audits"])
        )

    def test_status_identifies_frozen_input_and_design_task(self) -> None:
        run = self._completed_run("11223")
        (run / "input").mkdir()
        source = run / "input" / "presymmetrized_input.cif"
        source.write_text("data_test\n", encoding="utf-8")
        self._write_json(
            run / "input" / "rfd3_input.json",
            {
                "example": {
                    "input": "presymmetrized_input.cif",
                    "symmetry": {"id": "C3"},
                    "extra": {
                        "symmetry_multiplicity": 3,
                        "motif_constraint_orbits": [{}, {}],
                        "assembly_interface_relations": [
                            {"satisfaction_stage": "input"}
                        ],
                    },
                }
            },
        )
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "topology": {"kind": "user_design"},
                    "sampling": {"timesteps": 50, "seed": 7},
                    "resources": {"profile_name": "p100"},
                }
            ),
            encoding="utf-8",
        )

        status = collect_run_status(
            RunReference(job_id="11223", run_directory=run),
            include_scheduler=False,
        )

        self.assertEqual(status["design"]["task"], "preserve_supplied_geometry")
        self.assertEqual(status["design"]["symmetry"], "C3")
        self.assertEqual(status["design"]["fixed_component_count"], 2)
        text = format_status_text(status)
        self.assertIn("task:       preserve_supplied_geometry", text)
        self.assertIn("symmetry:   C3 x3", text)
        self.assertIn(str(source.resolve()), text)

    def test_status_traces_compiled_input_back_to_original_structure(self) -> None:
        run = self._completed_run("11224")
        (run / "input").mkdir()
        compiled = run / "input" / "presymmetrized_input.cif"
        compiled.write_text("data_test\n", encoding="utf-8")
        original = run / "inputs" / "supplied_seed.pdb"
        original.parent.mkdir()
        original.write_text("END\n", encoding="utf-8")
        self._write_json(
            run / "input" / "rfd3_input.json",
            {
                "example": {
                    "input": "presymmetrized_input.cif",
                    "symmetry": {"id": "C3"},
                    "extra": {
                        "symmetry_multiplicity": 3,
                        "motif_constraint_orbits": [{}, {}],
                        "assembly_interface_relations": [
                            {
                                "source_interface_id": "interface_alpha",
                                "satisfaction_stage": "input",
                            },
                            {
                                "source_interface_id": "interface_alpha",
                                "satisfaction_stage": "input",
                            },
                            {
                                "source_interface_id": "interface_beta",
                                "satisfaction_stage": "input",
                            },
                        ],
                    },
                }
            },
        )
        self._write_json(
            run / "input" / "manifest.json",
            {"inputs": [{"path": str(original), "sha256": "abc123"}]},
        )
        (run / "input" / "assembly_specification.yaml").write_text(
            yaml.safe_dump(
                {
                    "assembly": {
                        "fragments": {
                            "left": {"selection": "A/186-189/*"},
                            "right": {"selection": "B/238-240/*"},
                        },
                        "ports": {
                            "left_port": {"fragments": ["left"]},
                            "right_port": {"fragments": ["right"]},
                        },
                        "interfaces": {
                            "interface_alpha": {
                                "left_port": "left_port",
                                "right_port": "right_port",
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        status = collect_run_status(
            RunReference(job_id="11224", run_directory=run),
            include_scheduler=False,
        )

        design = status["design"]
        self.assertEqual(design["structure_input"], str(original))
        self.assertEqual(design["compiled_input"], str(compiled.resolve()))
        self.assertEqual(
            design["interface_instances"],
            {"interface_alpha": 2, "interface_beta": 1},
        )
        self.assertEqual(
            design["interface_selections"],
            {"interface_alpha": ["A/186-189/*", "B/238-240/*"]},
        )
        text = format_status_text(status)
        self.assertIn(f"input:      {original}", text)
        self.assertIn(f"RFD3 input:  {compiled.resolve()}", text)
        self.assertIn("not a generated design", text)
        self.assertIn("interfaces: interface_alpha x2, interface_beta x1", text)
        self.assertIn("- interface_alpha: A/186-189/* + B/238-240/*", text)
        report = write_report(status)
        html = report.read_text(encoding="utf-8")
        self.assertIn("Design provenance", html)
        self.assertIn("supplied_seed.pdb", html)
        self.assertIn("A/186-189/*", html)

    def test_status_recovers_context_from_multi_example_pose_artifacts(self) -> None:
        run = self._completed_run("11225")
        input_directory = run / "input"
        input_directory.mkdir()
        original = run / "inputs" / "supplied_seed.pdb"
        original.parent.mkdir()
        original.write_text("END\n", encoding="utf-8")
        examples = {}
        for index in range(2):
            pose_directory = input_directory / f"pose_{index:05d}"
            pose_directory.mkdir()
            compiled = pose_directory / "presymmetrized_input.cif"
            compiled.write_text("data_test\n", encoding="utf-8")
            self._write_json(
                pose_directory / "manifest.json",
                {"inputs": [{"path": str(original), "sha256": "abc123"}]},
            )
            examples[f"design_{index:05d}"] = {
                "input": f"pose_{index:05d}/presymmetrized_input.cif",
                "symmetry": {"id": "C3"},
                "extra": {
                    "symmetry_multiplicity": 3,
                    "motif_constraint_orbits": [{"id": "fixed"}],
                    "assembly_interface_relations": [{"satisfaction_stage": "output"}],
                },
            }
        self._write_json(input_directory / "rfd3_input.json", examples)
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "topology": {"kind": "user_design"},
                    "sampling": {"timesteps": 50, "seed": 73000, "designs": 2},
                }
            ),
            encoding="utf-8",
        )

        status = collect_run_status(
            RunReference(job_id="11225", run_directory=run),
            include_scheduler=False,
        )

        design = status["design"]
        self.assertEqual(design["compiled_example_count"], 2)
        self.assertEqual(design["task"], "create_symmetric_interface")
        self.assertEqual(design["symmetry"], "C3")
        self.assertEqual(design["symmetry_multiplicity"], 3)
        self.assertEqual(design["fixed_component_count"], 1)
        self.assertEqual(design["structure_input"], str(original))
        self.assertIn("pose_00000", design["compiled_input"])


if __name__ == "__main__":
    unittest.main()
