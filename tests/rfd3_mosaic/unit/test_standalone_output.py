import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from rfd3_mosaic.output import compile_standalone
from rfd3_mosaic.output.standalone import _classify_symmetry_pair


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


class StandaloneOutputTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.outputs = compile_standalone(
            LHD101_CONFIG,
            self.output_directory,
            base_directory=REPOSITORY_ROOT,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_lhd101_c3_output_counts(self) -> None:
        self.assertEqual(self.outputs.atom_count, 1488)
        self.assertEqual(self.outputs.residue_count, 183)
        self.assertEqual(self.outputs.chain_count, 6)

    def test_all_artifacts_are_written(self) -> None:
        self.assertTrue(self.outputs.structure_path.is_file())
        self.assertTrue(self.outputs.mapping_path.is_file())
        self.assertTrue(self.outputs.manifest_path.is_file())

    def test_cif_contains_six_chains_and_all_atoms(self) -> None:
        atom_rows = [
            line.split()
            for line in self.outputs.structure_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.startswith(("ATOM ", "HETATM "))
        ]

        self.assertEqual(len(atom_rows), 1488)
        self.assertEqual({row[6] for row in atom_rows}, set("ABCDEF"))

    def test_mapping_covers_every_atom(self) -> None:
        payload = json.loads(
            self.outputs.mapping_path.read_text(encoding="utf-8")
        )

        self.assertEqual(len(payload["atom_mappings"]), 1488)
        self.assertEqual(len(payload["fragment_ranges"]), 6)
        indices = [
            record["compiled"]["atom_index"]
            for record in payload["atom_mappings"]
        ]
        self.assertEqual(indices, list(range(1488)))

    def test_manifest_is_explicit_about_unbuilt_linkers(self) -> None:
        manifest = json.loads(
            self.outputs.manifest_path.read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["counts"]["scaffold_link_instances"], 3)
        self.assertTrue(
            any(
                "not generated" in limitation
                for limitation in manifest["limitations"]
            )
        )

    def test_master_seed_center_matches_sampled_radial_distance(self) -> None:
        atom_rows = [
            line.split()
            for line in self.outputs.structure_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.startswith(("ATOM ", "HETATM "))
        ]
        first_group_coordinates = np.asarray(
            [
                [float(row[10]), float(row[11]), float(row[12])]
                for row in atom_rows
                if row[6] in {"A", "B"}
            ]
        )
        center = first_group_coordinates.mean(axis=0)
        manifest = json.loads(
            self.outputs.manifest_path.read_text(encoding="utf-8")
        )
        sampled_radius = manifest["initialization_samples"][
            "primary_seed"
        ]["sampled_radius"]

        self.assertGreaterEqual(sampled_radius, 20.0)
        self.assertLessEqual(sampled_radius, 30.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(center[:2])),
            sampled_radius,
            places=2,
        )
        self.assertAlmostEqual(float(center[2]), 0.0, places=2)

    def test_manifest_reports_no_hard_inter_group_clashes(self) -> None:
        manifest = json.loads(
            self.outputs.manifest_path.read_text(encoding="utf-8")
        )
        report = manifest["validation"]["inter_group_clashes"]

        self.assertEqual(report["total_hard_clashes"], 0)
        self.assertGreater(report["minimum_inter_group_distance"], 2.0)
        self.assertEqual(
            report["categories"]["cyclic_intra"]["group_pair_count"],
            3,
        )

    def test_manifest_reports_scaffold_endpoint_feasibility(self) -> None:
        manifest = json.loads(
            self.outputs.manifest_path.read_text(encoding="utf-8")
        )
        report = manifest["validation"]["scaffold_link_geometry"]

        self.assertTrue(
            report["all_continuous_links_within_maximum_contour"]
        )
        self.assertEqual(len(report["links"]), 3)
        for link in report["links"]:
            self.assertEqual(link["from_anchor"], "C")
            self.assertEqual(link["to_anchor"], "N")
            self.assertGreater(link["endpoint_distance"], 0.0)
            self.assertTrue(link["within_maximum_contour"])

    def test_manifest_reports_symmetry_axis_and_central_clearance(self) -> None:
        manifest = json.loads(
            self.outputs.manifest_path.read_text(encoding="utf-8")
        )
        report = manifest["validation"]["symmetry_cavities"]

        self.assertEqual(len(report["orbits"]), 1)
        orbit = report["orbits"][0]
        self.assertEqual(orbit["orbit_id"], "primary_orbit")
        self.assertEqual(orbit["symmetry_type"], "cyclic")
        self.assertEqual(orbit["copy_count"], 3)
        self.assertGreater(orbit["central_void_radius"], 0.0)
        self.assertGreaterEqual(orbit["minimum_axis_clearance"], 0.0)

    def test_dihedral_pair_classes_distinguish_cosets(self) -> None:
        self.assertEqual(
            _classify_symmetry_pair(
                "assembly",
                "D3:e",
                "assembly",
                "D3:r2",
            ),
            "dihedral_intra_coset",
        )
        self.assertEqual(
            _classify_symmetry_pair(
                "assembly",
                "D3:r1",
                "assembly",
                "D3:s1",
            ),
            "dihedral_inter_coset",
        )

    def test_all_required_reference_interfaces_are_satisfied(self) -> None:
        manifest = json.loads(
            self.outputs.manifest_path.read_text(encoding="utf-8")
        )
        report = manifest["validation"]["interfaces"]

        self.assertTrue(report["all_required_satisfied"])
        self.assertEqual(len(report["edges"]), 3)
        for edge in report["edges"]:
            self.assertTrue(edge["satisfied"])
            self.assertLess(edge["translation_error"], 1e-6)
            self.assertLess(edge["rotation_error_deg"], 1e-5)
            self.assertGreater(edge["heavy_atom_contacts_below_4_5A"], 0)

    def test_compiler_rejects_unplaced_overlapping_c3_copies(self) -> None:
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        payload["interface_seed"]["initialization"] = {}
        invalid_config = self.output_directory / "overlapping.yaml"
        invalid_config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "rejected severe inter-group clashes",
        ):
            compile_standalone(
                invalid_config,
                self.output_directory / "invalid-output",
                base_directory=REPOSITORY_ROOT,
            )

    def test_relaxed_compilation_scores_infeasible_pose(self) -> None:
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        payload["interface_seed"]["initialization"] = {}
        payload["interface_seed"]["objectives"] = {
            "no_hard_clashes": {
                "metric": "clashes.total_hard_clashes",
                "mode": "at_most",
                "threshold": 0.0,
                "scale": 1.0,
                "required": True,
            }
        }
        invalid_config = self.output_directory / "scored-overlap.yaml"
        invalid_config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        outputs = compile_standalone(
            invalid_config,
            self.output_directory / "scored-overlap-output",
            base_directory=REPOSITORY_ROOT,
            strict_validation=False,
        )
        manifest = json.loads(
            outputs.manifest_path.read_text(encoding="utf-8")
        )
        validation = manifest["validation"]

        self.assertFalse(validation["strict_validation"])
        self.assertGreater(
            validation["inter_group_clashes"]["total_hard_clashes"],
            0,
        )
        self.assertEqual(
            validation["objectives"]["required_failure_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
