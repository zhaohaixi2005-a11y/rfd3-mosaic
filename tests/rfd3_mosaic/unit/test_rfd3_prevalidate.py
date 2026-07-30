import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.output import compile_rfd3_input
from rfd3_mosaic.rfd3_prevalidate import (
    _audit_runtime_transform_matrices,
    _expected_multiplicity,
    _validate_report,
    prevalidate_rfd3_input,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


class RFD3PrevalidationLogicTestCase(unittest.TestCase):
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
            report = prevalidate_rfd3_input(outputs.input_path)
            emitted = json.loads(
                outputs.input_path.read_text(encoding="utf-8")
            )[outputs.example_id]

        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["residues_per_chain"].values()), {146})
        sampled_contig = report["rfd3_metadata"]["extra"][
            "sampled_contig"
        ]
        self.assertIn("85P", sampled_contig.split(","))
        self.assertEqual(
            emitted["extra"]["materialized_linker_length"],
            85,
        )

    def test_reads_cyclic_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("C3"), 3)
        self.assertEqual(_expected_multiplicity("c12"), 12)

    def test_reads_dihedral_multiplicity(self) -> None:
        self.assertEqual(_expected_multiplicity("D2"), 4)
        self.assertEqual(_expected_multiplicity("d5"), 10)

    def test_rejects_unsupported_symmetry(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cn/Dn"):
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
