import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from biotite.structure import AtomArray
from rfd3.inference.parsing import InputSelection

from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.output import compile_rfd3_input
from rfd3_mosaic.rfd3_prevalidate import (
    _audit_runtime_transform_matrices,
    _expected_multiplicity,
    _validate_report,
    prevalidate_rfd3_input,
)
from rfd3_mosaic.schema import SimpleCageIntentSpec
from rfd3_mosaic.simple_resolver import (
    enumerate_simple_design_candidates,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


class RFD3PrevalidationLogicTestCase(unittest.TestCase):
    def test_empty_selection_accepts_extended_mmcif_chain_ids(self) -> None:
        atoms = AtomArray(2)
        atoms.chain_id = np.asarray(["AA", "AA"])
        atoms.res_id = np.asarray([1, 1])
        atoms.res_name = np.asarray(["ALA", "ALA"])
        atoms.atom_name = np.asarray(["N", "CA"])

        selection = InputSelection.from_any(False, atoms)

        self.assertIsNotNone(selection)
        self.assertEqual(selection.data, {})
        self.assertEqual(selection.tokens, {})
        self.assertFalse(selection.mask.any())

    def test_cross_copy_multi_seed_path_builds_runtime_constraint_groups(
        self,
    ) -> None:
        """Exercise the runtime token binding for a real C3 seam path.

        This is deliberately stronger than adapter compilation.  Candidate
        000007 contains two supplied binary interface seeds and one polymer
        link that crosses from the master copy to the previous C3 copy.  The
        emitted contig therefore fixes physical fragments from more than one
        symmetry copy.  ``AddSymmetryFeats`` must bind every motif-constraint
        member to those actual fixed runtime atoms; canonicalizing the seam
        fragment back to copy zero leaves an empty member and is rejected by
        :func:`prevalidate_rfd3_input`.
        """

        structure = (
            REPOSITORY_ROOT
            / "examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb"
        )
        intent = SimpleCageIntentSpec.model_validate(
            {
                "name": "cross-copy-runtime-constraint-regression",
                "input": structure,
                "goal": {
                    "architecture": "ring",
                    "composition": "auto",
                    "symmetry": ["C3"],
                    "subunits": {"minimum": 6, "maximum": 6},
                },
                "interface_seeds": {
                    "interface_alpha": {
                        "participants": ["A", "B"],
                        "selectors": {
                            "A": "A/186-189/*",
                            "B": "B/238-240/*",
                        },
                        "use": {"exact": 3},
                    },
                    "interface_beta": {
                        "participants": ["A", "B"],
                        "selectors": {
                            "A": "A/191-192/*",
                            "B": "B/234-235/*",
                        },
                        "use": {"exact": 3},
                    },
                },
                "generation": {
                    "length": {"minimum": 10, "maximum": 30}
                },
            }
        )
        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            timesteps=50,
        )
        candidate = next(
            item
            for item in candidates
            if item.candidate_id == "candidate_000007"
        )
        self.assertEqual(candidate.connection_orbit_offset, -1)
        self.assertEqual(
            sum(offset != 0 for _, _, offset in candidate.polymer_links),
            1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assembly_path = root / "assembly.yaml"
            lowered = lower_user_design(candidate.design)
            assembly_path.write_text(
                yaml.safe_dump(
                    {
                        "assembly": lowered.specification.model_dump(
                            mode="json"
                        )
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs = compile_rfd3_input(
                assembly_path,
                root / "adapter",
                base_directory=structure.parent,
                example_id="cross_copy_runtime_constraint_regression",
            )
            emitted = json.loads(
                outputs.input_path.read_text(encoding="utf-8")
            )[outputs.example_id]
            report = prevalidate_rfd3_input(outputs.input_path)

        self.assertIn("/0", outputs.contig)
        declared_groups = emitted["extra"]["motif_constraint_groups"]
        self.assertEqual(len(declared_groups), 6)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["resolved_motif_constraint_group_count"],
            len(declared_groups),
        )
        self.assertEqual(
            report["motif_constraint_group_covered_atom_count"],
            report["fixed_coordinate_atom_count"],
        )

    def test_compiled_lhd101_uses_one_materialized_linker_build(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            outputs = compile_rfd3_input(
                LHD101_CONFIG,
                output_directory,
                base_directory=REPOSITORY_ROOT,
            )
            import torch

            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            report = prevalidate_rfd3_input(outputs.input_path)
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_equal(np.random.get_state(), numpy_state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            emitted = json.loads(
                outputs.input_path.read_text(encoding="utf-8")
            )[outputs.example_id]

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["atom_array_validated"])
        self.assertTrue(report["sampler_compatibility_validated"])
        self.assertTrue(report["runtime_features_validated"])
        self.assertFalse(report["checkpoint_compatibility_validated"])
        self.assertFalse(report["model_forward_validated"])
        self.assertEqual(
            report["runtime_configuration"]["configuration_source"],
            "shipped_source_configuration",
        )
        self.assertGreater(
            report["runtime_feature_audit"]["padded_atom_count"], report["atom_count"]
        )
        self.assertIn("ref_pos", report["runtime_feature_audit"]["tensor_shapes"])
        self.assertEqual(set(report["residues_per_chain"].values()), {146})
        sampled_contig = report["rfd3_metadata"]["extra"][
            "sampled_contig"
        ]
        self.assertIn("85P", sampled_contig.split(","))
        self.assertEqual(
            emitted["extra"]["materialized_linker_length"],
            85,
        )

    def test_preflight_rejects_incompatible_sampler_and_records_actual_scope(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs = compile_rfd3_input(
                LHD101_CONFIG, root, base_directory=REPOSITORY_ROOT
            )
            report_path = root / "negative.json"
            with self.assertRaisesRegex(ValueError, "sampler is not set to symmetry"):
                prevalidate_rfd3_input(
                    outputs.input_path,
                    inference_sampler={"kind": "default"},
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text())
        self.assertTrue(report["atom_array_validated"])
        self.assertFalse(report["sampler_compatibility_validated"])
        self.assertFalse(report["runtime_features_validated"])
        self.assertNotIn("native_inference_features", report["validation_scope"])
        self.assertEqual(report["status"], "failed")

    def test_preflight_executes_supplied_transform_instead_of_source_defaults(self):
        from rfd3_mosaic.rfd3_runtime_preflight import source_inference_training_config

        config = source_inference_training_config()
        # This is valid input geometry but an impossible feature configuration.
        # It fails inside native padding, beyond the old atom-array-only check.
        config.datasets.global_transform_args.central_atom = "DOES_NOT_EXIST"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outputs = compile_rfd3_input(
                LHD101_CONFIG, root, base_directory=REPOSITORY_ROOT
            )
            report_path = root / "negative.json"
            with self.assertRaisesRegex(ValueError, "Native inference features"):
                prevalidate_rfd3_input(
                    outputs.input_path,
                    resolved_training_config=config,
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text())
        self.assertTrue(report["atom_array_validated"])
        self.assertTrue(report["sampler_compatibility_validated"])
        self.assertFalse(report["runtime_features_validated"])
        self.assertFalse(report["runtime_feature_audit"]["passed"])
        self.assertEqual(
            report["runtime_configuration"]["configuration_source"],
            "provided_training_configuration",
        )
        self.assertEqual(report["status"], "failed")

    def test_shared_inference_transform_resolution_keeps_interpolations_and_overrides(self):
        from omegaconf import OmegaConf

        from foundry.inference_engines.base import resolve_inference_transform_config

        config = OmegaConf.create({
            "model": {"sigma": 16},
            "datasets": {"val": {
                "first": {"dataset": {"transform": {
                    "sigma_data": "${model.sigma}",
                    "diffusion_batch_size": 32,
                    "nested": {"left": 1, "right": 2},
                }}},
                "unused": {"dataset": {"transform": {"sigma_data": 99}}},
            }},
        })
        key, transform = resolve_inference_transform_config(
            config, {"diffusion_batch_size": 1, "nested": {"right": 3}}
        )
        self.assertEqual(key, "first")
        self.assertEqual(OmegaConf.to_container(transform, resolve=True), {
            "sigma_data": 16,
            "diffusion_batch_size": 1,
            "nested": {"left": 1, "right": 3},
        })
        self.assertEqual(config.datasets.val.first.dataset.transform.diffusion_batch_size, 32)
        self.assertEqual(config.datasets.val.first.dataset.transform.nested.right, 2)

    def test_reads_cyclic_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("C3"), 3)
        self.assertEqual(_expected_multiplicity("c12"), 12)

    def test_reads_dihedral_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("D2"), 4)
        self.assertEqual(_expected_multiplicity("d5"), 10)

    def test_reads_polyhedral_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("T"), 12)
        self.assertEqual(_expected_multiplicity("o"), 24)
        self.assertEqual(_expected_multiplicity("I"), 60)

    def test_rejects_unsupported_symmetry(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cn/Dn/T/O/I"):
            _expected_multiplicity("T3")

    def test_accepts_consistent_constructed_atom_array_report(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 3,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {"A": 120, "B": 120, "C": 120},
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
            "declared_motif_constraint_group_count": 3,
            "resolved_motif_constraint_group_count": 3,
            "motif_constraint_group_covered_atom_count": 100,
            "symmetry_transform_matrix_audit": {"passed": True},
            "fixed_target_symmetry_audit": {"passed": True},
            "declared_motif_constraint_orbit_count": 1,
            "resolved_motif_constraint_orbit_count": 1,
        }
        self.assertEqual(_validate_report(report), [])

    def test_mosaic_adapter_cannot_omit_groups_or_orbits(self) -> None:
        report = {
            "compiler": "rfd3_mosaic.static_adapter",
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 3,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {"A": 120, "B": 120, "C": 120},
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
            "declared_motif_constraint_group_count": 0,
            "declared_motif_constraint_orbit_count": 0,
            "symmetry_transform_matrix_audit": {"passed": True},
            "fixed_target_symmetry_audit": {"passed": True},
        }

        failures = _validate_report(report)

        self.assertIn(
            "Mosaic adapter input must declare motif constraint groups",
            failures,
        )
        self.assertIn(
            "Mosaic adapter input must declare motif constraint orbits",
            failures,
        )

    def test_reports_incomplete_symmetry_expansion(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 2,
            "symmetry_transform_ids": [0, 1],
            "residues_per_chain": {"A": 120, "B": 119},
            "motif_atom_count": 0,
            "fixed_coordinate_atom_count": 0,
            "fixed_sequence_atom_count": 0,
            "asu_atom_count": 0,
            "symmetry_ids": ["C2"],
            "expected_symmetry_id": "C3",
        }
        self.assertGreaterEqual(len(_validate_report(report)), 6)

    def test_accepts_two_chain_asu_repeated_by_symmetry(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 2,
            "chain_count": 6,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {
                "A": 31,
                "B": 30,
                "C": 31,
                "D": 30,
                "E": 31,
                "F": 30,
            },
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
        }
        self.assertEqual(_validate_report(report), [])

    def test_ligand_chain_is_not_multiplied_as_a_protein_path(self) -> None:
        report = {
            "expected_multiplicity": 5,
            "expected_asu_chain_count": 2,
            "expected_asu_polymer_chain_count": 1,
            "expected_ligand_selector_count": 1,
            "chain_count": 6,
            "protein_chain_count": 5,
            "nonprotein_chain_count": 1,
            "symmetry_transform_ids": [0, 1, 2, 3, 4],
            "residues_per_chain": {
                "A": 260,
                "B": 260,
                "C": 260,
                "D": 260,
                "E": 260,
                "L": 1,
            },
            "protein_residues_per_chain": {
                "A": 260,
                "B": 260,
                "C": 260,
                "D": 260,
                "E": 260,
            },
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C5"],
            "expected_symmetry_id": "C5",
        }

        self.assertEqual(_validate_report(report), [])

    def test_rejects_partially_fixed_interface_seed(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 3,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {"A": 120, "B": 120, "C": 120},
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 60,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
        }
        failures = _validate_report(report)
        self.assertTrue(
            any("not every motif atom has fixed coordinates" in failure
                for failure in failures)
        )

    def test_rejects_incomplete_constraint_group_coverage(self) -> None:
        report = {
            "expected_multiplicity": 3,
            "expected_asu_chain_count": 1,
            "chain_count": 3,
            "symmetry_transform_ids": [0, 1, 2],
            "residues_per_chain": {"A": 120, "B": 120, "C": 120},
            "motif_atom_count": 100,
            "fixed_coordinate_atom_count": 100,
            "fixed_sequence_atom_count": 100,
            "asu_atom_count": 500,
            "symmetry_ids": ["C3"],
            "expected_symmetry_id": "C3",
            "declared_motif_constraint_group_count": 3,
            "resolved_motif_constraint_group_count": 2,
            "motif_constraint_group_covered_atom_count": 80,
        }

        failures = _validate_report(report)

        self.assertIn(
            "not every declared motif constraint group was resolved",
            failures,
        )
        self.assertIn(
            "motif constraint groups do not cover every fixed atom",
            failures,
        )

    @staticmethod
    def _z_transform(angle_degrees: float) -> np.ndarray:
        angle = np.deg2rad(angle_degrees)
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return matrix

    def test_transform_audit_resolves_permuted_c3_runtime_ids(
        self,
    ) -> None:
        registry = {
            "C3:e": self._z_transform(0.0),
            "C3:r1": self._z_transform(120.0),
            "C3:r2": self._z_transform(240.0),
        }
        runtime = {
            0: registry["C3:e"],
            1: registry["C3:r2"],
            2: registry["C3:r1"],
        }

        audit = _audit_runtime_transform_matrices(
            runtime,
            registry,
            ["C3:e", "C3:r2", "C3:r1"],
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["runtime_to_registry"],
            {"0": "C3:e", "1": "C3:r2", "2": "C3:r1"},
        )

    def test_transform_audit_rejects_wrong_declared_direction(
        self,
    ) -> None:
        registry = {
            "C3:e": self._z_transform(0.0),
            "C3:r1": self._z_transform(120.0),
            "C3:r2": self._z_transform(240.0),
        }
        runtime = {
            0: registry["C3:e"],
            1: registry["C3:r2"],
            2: registry["C3:r1"],
        }

        audit = _audit_runtime_transform_matrices(
            runtime,
            registry,
            ["C3:e", "C3:r1", "C3:r2"],
        )

        self.assertFalse(audit["passed"])
        self.assertTrue(
            any(
                "runtime transform 1" in failure
                for failure in audit["failures"]
            )
        )

    def test_transform_audit_accepts_sparse_declared_registry_subset(
        self,
    ) -> None:
        registry = {
            "C4:e": self._z_transform(0.0),
            "C4:r1": self._z_transform(90.0),
            "C4:r2": self._z_transform(180.0),
            "C4:r3": self._z_transform(270.0),
        }
        runtime = {
            0: registry["C4:e"],
            2: registry["C4:r2"],
        }

        strict = _audit_runtime_transform_matrices(
            runtime,
            registry,
            ["C4:e", "C4:r1", "C4:r2", "C4:r3"],
        )
        quotient = _audit_runtime_transform_matrices(
            runtime,
            registry,
            ["C4:e", "C4:r1", "C4:r2", "C4:r3"],
            allow_declared_subset=True,
        )

        self.assertFalse(strict["passed"])
        self.assertTrue(quotient["passed"])
        self.assertEqual(
            quotient["runtime_to_registry"],
            {"0": "C4:e", "2": "C4:r2"},
        )
        self.assertEqual(
            quotient["missing_registry_transform_ids"],
            ["C4:r1", "C4:r3"],
        )
        self.assertEqual(
            quotient["coverage_contract"],
            "declared_registry_subset",
        )

    def test_transform_audit_rejects_subset_id_outside_registry(
        self,
    ) -> None:
        identity = np.eye(4)

        audit = _audit_runtime_transform_matrices(
            {0: identity, 4: identity},
            {
                "C4:e": identity,
                "C4:r1": self._z_transform(90.0),
                "C4:r2": self._z_transform(180.0),
                "C4:r3": self._z_transform(270.0),
            },
            ["C4:e", "C4:r1", "C4:r2", "C4:r3"],
            allow_declared_subset=True,
        )

        self.assertFalse(audit["passed"])
        self.assertTrue(
            any(
                "outside the declared registry" in failure
                for failure in audit["failures"]
            )
        )

    def test_transform_audit_rejects_improper_runtime_frame(
        self,
    ) -> None:
        identity = np.eye(4)
        reflection = np.eye(4)
        reflection[0, 0] = -1.0

        audit = _audit_runtime_transform_matrices(
            {0: identity, 1: reflection},
            {"C2:e": identity, "C2:r1": self._z_transform(180.0)},
            ["C2:e", "C2:r1"],
        )

        self.assertFalse(audit["passed"])
        self.assertTrue(
            any(
                "proper rotation" in failure
                for failure in audit["failures"]
            )
        )

    def test_transform_audit_rejects_malformed_registry_frame(
        self,
    ) -> None:
        identity = np.eye(4)
        scaled = np.eye(4)
        scaled[0, 0] = 2.0

        audit = _audit_runtime_transform_matrices(
            {0: identity, 1: self._z_transform(180.0)},
            {"C2:e": identity, "C2:r1": scaled},
            ["C2:e", "C2:r1"],
        )

        self.assertFalse(audit["passed"])
        self.assertTrue(
            any(
                "registry transform 'C2:r1' is not orthogonal" in failure
                for failure in audit["failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
