from pathlib import Path
import tempfile
import unittest

import yaml

from rfd3_mosaic.cli import _parser, _write_quick_experiment
from rfd3_mosaic.experiment import (
    build_execution_plan,
    render_submission,
    resolve_experiment,
)


class ExperimentConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.template_structure = self.root / "source.cif"
        self.template_structure.write_text("data_test\n", encoding="utf-8")
        self.template_input = self.root / "template.json"
        self.template_input.write_text(
            '{"source": {"input": "source.cif", "symmetry": {"id": "C3"}}}\n',
            encoding="utf-8",
        )
        self.profile = self.root / "profile.yaml"
        self.profile.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "test-gpu",
                    "slurm": {
                        "partition": "test-gpu",
                        "gres": "gpu:1",
                        "cpus": 4,
                        "memory": "16G",
                        "walltime": "00:30:00",
                    },
                    "setup_commands": ["source /test/environment.sh"],
                    "checkpoint": "/checkpoints/rfd3.ckpt",
                    "foundry_checkpoint_dirs": "/checkpoints",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_experiment(self, payload) -> Path:
        path = self.root / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def test_resolves_central_motif_exact_preset(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "central-test",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                    "n_terminal_length": 20,
                    "c_terminal_length": 25,
                },
                "sampling": {"timesteps": 50, "seed": 7},
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs", "campaign": "test"},
            }
        )

        resolved = resolve_experiment(config)

        self.assertEqual(resolved.payload["topology"]["kind"], "central_motif")
        self.assertEqual(resolved.payload["sampling"]["preset"], "exact_mosaic")
        sampler = resolved.payload["sampling"]["sampler"]
        self.assertFalse(sampler["allow_realignment"])
        self.assertTrue(sampler["preserve_fixed_motif_during_symmetry"])
        self.assertEqual(sampler["symmetry_state_mode"], "orbit_average")
        self.assertEqual(resolved.payload["resources"]["slurm"]["cpus"], 4)
        provenance = resolved.payload["provenance"]
        self.assertIn("repository", provenance)
        self.assertIn("commit", provenance["repository"])
        compatibility = provenance["foundry_compatibility"]
        self.assertEqual(
            compatibility["manifest"]["engine_id"],
            "mosaic-rfd3",
        )
        self.assertEqual(len(compatibility["sha256"]), 64)

    def test_rendered_sbatch_delegates_to_worker(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "render-test",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                },
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs", "campaign": "test"},
            }
        )
        resolved = resolve_experiment(config)

        script = render_submission(resolved, output_directory=self.root / "rendered")
        text = script.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=test-gpu", text)
        self.assertIn("python -m rfd3_mosaic.experiment_worker", text)
        self.assertNotIn("python -m rfd3.run_inference", text)
        self.assertTrue((script.parent / "resolved_config.yaml").is_file())
        self.assertTrue((script.parent / "provenance.json").is_file())

    def test_builds_a_read_only_user_auditable_plan(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "plan-test",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                    "n_terminal_length": 20,
                    "c_terminal_length": 25,
                },
                "sampling": {"timesteps": 50, "seed": 7},
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs", "campaign": "test"},
            }
        )

        plan = build_execution_plan(resolve_experiment(config))

        self.assertEqual(plan["design"]["topology"], "central_motif")
        constraint = plan["design"]["effective_constraints"][0]
        self.assertEqual(constraint["operator"], "fixed_xyz")
        self.assertEqual(constraint["selector"], "B1-31")
        self.assertEqual(
            constraint["orbit_scope"],
            "complete_symmetry_orbit",
        )
        self.assertEqual(plan["sampling"]["timesteps"], 50)
        self.assertEqual(plan["execution"]["profile"], "test-gpu")
        self.assertEqual(plan["software"]["compatibility_id"], "mosaic-rfd3")

    def test_plan_command_supports_machine_readable_output(self) -> None:
        arguments = _parser().parse_args(
            ["plan", "design.yaml", "--format", "json"]
        )

        self.assertEqual(arguments.command, "plan")
        self.assertEqual(arguments.format, "json")

    def test_resolves_official_rfd3_control_preset(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "official-control",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                },
                "sampling": {
                    "preset": "official_rfd3",
                    "timesteps": 200,
                    "seed": 102,
                },
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs", "campaign": "official-a"},
            }
        )

        resolved = resolve_experiment(config)
        sampler = resolved.payload["sampling"]["sampler"]

        self.assertEqual(
            sampler["fixed_motif_finalization_mode"],
            "official_reinsert_then_project",
        )
        self.assertFalse(sampler["preserve_fixed_motif_during_symmetry"])
        self.assertFalse(sampler["require_motif_constraint_groups"])
        self.assertEqual(sampler["symmetry_state_mode"], "legacy_asu")
        self.assertEqual(sampler["symmetry_noise_mode"], "independent")

    def test_unknown_fields_fail_closed(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "bad-test",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                    "mystery": True,
                },
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs"},
            }
        )

        with self.assertRaisesRegex(ValueError, "Unknown central_motif"):
            resolve_experiment(config)

    def test_boolean_strings_are_rejected(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "bad-boolean",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                },
                "sampling": {"low_memory_mode": "false"},
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs"},
            }
        )

        with self.assertRaisesRegex(ValueError, "must be true or false"):
            resolve_experiment(config)

    def test_invalid_declared_checkpoint_digest_is_rejected(self) -> None:
        profile = yaml.safe_load(self.profile.read_text(encoding="utf-8"))
        profile["checkpoint_sha256"] = "not-a-digest"
        self.profile.write_text(
            yaml.safe_dump(profile, sort_keys=False),
            encoding="utf-8",
        )
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "bad-checkpoint-digest",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                },
                "resources": {"profile": self.profile.name},
                "output": {"root": "runs"},
            }
        )

        with self.assertRaisesRegex(ValueError, "64-character SHA256"):
            resolve_experiment(config)

    def test_profile_override_accepts_a_path(self) -> None:
        config = self._write_experiment(
            {
                "schema_version": 1,
                "name": "profile-path",
                "topology": {
                    "kind": "central_motif",
                    "template_input": self.template_input.name,
                    "fixed_selector": "B1-31",
                },
                "resources": {"profile": "unused-profile"},
                "output": {"root": "runs"},
            }
        )

        resolved = resolve_experiment(config, profile_override=self.profile)

        self.assertEqual(resolved.profile_path, self.profile.resolve())

    def test_quick_central_command_writes_internal_experiment(self) -> None:
        arguments = _parser().parse_args(
            [
                "central",
                "--input",
                str(self.template_input),
                "--motif",
                "B1-31",
                "--n-length",
                "20",
                "--c-length",
                "25",
                "--output",
                str(self.root / "runs"),
                "--profile",
                str(self.profile),
                "--preset",
                "official_rfd3",
                "--dry-run",
            ]
        )

        path = _write_quick_experiment(arguments)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["topology"]["kind"], "central_motif")
        self.assertEqual(payload["topology"]["fixed_selector"], "B1-31")
        self.assertEqual(payload["topology"]["n_terminal_length"], 20)
        self.assertEqual(payload["topology"]["c_terminal_length"], 25)
        self.assertEqual(payload["sampling"]["preset"], "official_rfd3")


if __name__ == "__main__":
    unittest.main()
