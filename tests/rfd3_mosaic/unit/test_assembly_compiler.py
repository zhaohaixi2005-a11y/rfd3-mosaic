import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rfd3_mosaic.assembly_frontends import AssemblyCompilationRequest
from rfd3_mosaic.assembly_compiler import (
    CompiledAudit,
    compile_experiment_assembly,
)


class AssemblyCompilerTestCase(unittest.TestCase):
    def test_central_frontend_returns_common_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected = output / "rfd3_input.json"
            mapping = output / "mapping.json"
            with patch(
                "rfd3_mosaic.assembly_compiler.lower_experiment_topology",
                return_value=AssemblyCompilationRequest(
                    specification_path=output / "assembly.yaml",
                    example_id="central-c3",
                    semantic_audit="central_motif",
                    audit_metadata={"probe_fixed_selector": "A1-31"},
                ),
            ) as lower, patch(
                "rfd3_mosaic.assembly_compiler.compile_assembly_rfd3_input",
                return_value=SimpleNamespace(
                    input_path=expected,
                    mapping_path=mapping,
                ),
            ) as compile_native:
                artifact = compile_experiment_assembly(
                    {"kind": "central_motif"},
                    output,
                    project_directory=".",
                    experiment_name="central-c3",
                )

        self.assertEqual(artifact.input_path, expected)
        self.assertEqual(artifact.example_id, "central-c3")
        self.assertEqual(len(artifact.semantic_audits), 1)
        self.assertEqual(
            artifact.semantic_audits[0].module,
            "rfd3_mosaic.rfd3_central_motif_audit",
        )
        self.assertEqual(
            artifact.semantic_audits[0].report_name,
            "central_motif_audit.json",
        )
        lower.assert_called_once()
        compile_native.assert_called_once()

    def test_interface_frontend_returns_common_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected_input = output / "rfd3_input.json"
            expected_mapping = output / "mapping.json"
            adapter_outputs = SimpleNamespace(
                input_path=expected_input,
                mapping_path=expected_mapping,
            )
            with patch(
                "rfd3_mosaic.assembly_compiler.lower_experiment_topology",
                return_value=AssemblyCompilationRequest(
                    specification_path=Path("assembly.yaml"),
                    example_id="interface-c5",
                    pose_seed=42,
                    linker_length=80,
                    semantic_audit="interface_seed",
                ),
            ), patch(
                "rfd3_mosaic.assembly_compiler.compile_assembly_rfd3_input",
                return_value=adapter_outputs,
            ) as compile_interface:
                artifact = compile_experiment_assembly(
                    {
                        "kind": "interface_seed",
                        "config": "assembly.yaml",
                        "example_id": "interface-c5",
                        "pose_seed": 42,
                        "linker_length": 80,
                    },
                    output,
                    project_directory="/project",
                    experiment_name="ignored-for-interface",
                )

        self.assertEqual(artifact.input_path, expected_input)
        self.assertEqual(len(artifact.semantic_audits), 1)
        self.assertEqual(
            artifact.semantic_audits[0].module,
            "rfd3_mosaic.rfd3_seed_audit",
        )
        self.assertIn(
            ("--adapter-mapping", str(expected_mapping)),
            artifact.semantic_audits[0].input_arguments,
        )
        compile_interface.assert_called_once()

    def test_both_frontends_invoke_the_same_native_compiler(self) -> None:
        requests = (
            AssemblyCompilationRequest(
                specification_path=Path("central.yaml"),
                example_id="central",
                semantic_audit="central_motif",
            ),
            AssemblyCompilationRequest(
                specification_path=Path("interface.yaml"),
                example_id="interface",
                semantic_audit="interface_seed",
            ),
        )
        outputs = SimpleNamespace(
            input_path=Path("rfd3_input.json"),
            mapping_path=Path("mapping.json"),
        )
        with patch(
            "rfd3_mosaic.assembly_compiler.lower_experiment_topology",
            side_effect=requests,
        ), patch(
            "rfd3_mosaic.assembly_compiler.compile_assembly_rfd3_input",
            return_value=outputs,
        ) as native:
            compile_experiment_assembly(
                {"kind": "central_motif"},
                "/tmp/central",
                project_directory=".",
                experiment_name="central",
            )
            compile_experiment_assembly(
                {"kind": "interface_seed"},
                "/tmp/interface",
                project_directory=".",
                experiment_name="interface",
            )

        self.assertEqual(native.call_count, 2)

    def test_compiled_audit_builds_a_complete_generic_command(self) -> None:
        audit = CompiledAudit(
            module="example.audit",
            report_name="audit.json",
            input_arguments=(("--input", "/tmp/input.json"),),
        )

        self.assertEqual(
            audit.command(
                python="/env/bin/python",
                result_json="/run/result.json",
                output_report="/run/audit.json",
            ),
            [
                "/env/bin/python",
                "-m",
                "example.audit",
                "--input",
                "/tmp/input.json",
                "--result-json",
                "/run/result.json",
                "--output",
                "/run/audit.json",
                "--report-only",
            ],
        )

    def test_unknown_legacy_frontend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported topology"):
            compile_experiment_assembly(
                {"kind": "unknown"},
                "/tmp/output",
                project_directory=".",
                experiment_name="invalid",
            )


if __name__ == "__main__":
    unittest.main()
