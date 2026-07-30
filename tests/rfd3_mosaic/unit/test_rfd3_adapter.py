import json
import tempfile
import unittest
from pathlib import Path

import yaml

from rfd3_mosaic.output import compile_rfd3_input, compile_standalone


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)
LHD101_D2_DRYRUN_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/dihedral/lhd101_d2_dryrun.yaml"
)
LHD101_D3_DRYRUN_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/dihedral/lhd101_d3_dryrun.yaml"
)
LHD101_CYCLIC_CONFIGS = {
    order: (
        REPOSITORY_ROOT
        / f"configs/rfd3_mosaic/cyclic/lhd101_c{order}.yaml"
    )
    for order in (5, 6, 7)
}


class RFD3AdapterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.outputs = compile_rfd3_input(
            LHD101_CONFIG,
            self.output_directory,
            base_directory=REPOSITORY_ROOT,
        )
        self.payload = json.loads(
            self.outputs.input_path.read_text(encoding="utf-8")
        )[self.outputs.example_id]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_uses_cross_copy_asu_scaffold_contig(self) -> None:
        self.assertEqual(
            self.payload["contig"],
            "B1-31,85-85,C1-30",
        )
        extra = self.payload["extra"]
        self.assertEqual(
            extra["configured_linker_length_range"],
            [70, 100],
        )
        self.assertEqual(extra["materialized_linker_length"], 85)
        self.assertEqual(
            extra["linker_length_policy"],
            "configured_range_midpoint",
        )
        self.assertTrue(extra["contig_linker_is_deterministic"])
        contour = extra["materialized_linker_contour_preflight"]
        self.assertTrue(contour["passed"])
        self.assertEqual(contour["status"], "passed")
        self.assertEqual(
            len(contour["evaluated_link_instances"]),
            3,
        )
        self.assertTrue(
            all(
                item["materialized_linker_length"] == 85
                and item["passed"]
                for item in contour["evaluated_link_instances"]
            )
        )

    def test_accepts_an_explicit_linker_length_inside_configured_range(
        self,
    ) -> None:
        outputs = compile_rfd3_input(
            LHD101_CONFIG,
            self.output_directory / "explicit-linker",
            base_directory=REPOSITORY_ROOT,
            linker_length=92,
        )
        emitted = json.loads(
            outputs.input_path.read_text(encoding="utf-8")
        )[outputs.example_id]

        self.assertEqual(emitted["contig"], "B1-31,92-92,C1-30")
        self.assertEqual(
            emitted["extra"]["materialized_linker_length"],
            92,
        )
        self.assertEqual(
            emitted["extra"]["linker_length_policy"],
            "explicit",
        )

    def test_rejects_linker_length_outside_configured_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "inside the configured range",
        ):
            compile_rfd3_input(
                LHD101_CONFIG,
                self.output_directory / "invalid-linker",
                base_directory=REPOSITORY_ROOT,
                linker_length=101,
            )

    def test_uses_native_c3_symmetry(self) -> None:
        self.assertEqual(
            self.payload["symmetry"],
            {"id": "C3", "is_symmetric_motif": True},
        )

    def test_fixes_all_motif_atoms_and_preserves_sequence(self) -> None:
        self.assertEqual(
            self.payload["select_fixed_atoms"],
            {"B1-31": "ALL", "C1-30": "ALL"},
        )
        self.assertFalse(self.payload["redesign_motif_sidechains"])

    def test_structure_path_is_portable_relative_to_json(self) -> None:
        self.assertEqual(
            self.payload["input"],
            "presymmetrized_input.cif",
        )
        self.assertTrue(self.outputs.structure_path.is_file())

    def test_records_asu_copy_relation(self) -> None:
        extra = self.payload["extra"]
        self.assertEqual(extra["asu_source_copy_index"], 0)
        self.assertEqual(extra["asu_target_copy_index"], 1)

    def test_embeds_registry_matrices_for_runtime_prevalidation(
        self,
    ) -> None:
        extra = self.payload["extra"]
        self.assertEqual(
            list(extra["registry_transform_matrices"]),
            extra["registry_transform_order"],
        )
        for matrix in extra["registry_transform_matrices"].values():
            self.assertEqual(len(matrix), 4)
            self.assertTrue(all(len(row) == 4 for row in matrix))

    def test_emits_one_static_master_constraint_orbit(self) -> None:
        orbits = self.payload["extra"]["motif_constraint_orbits"]

        self.assertEqual(len(orbits), 1)
        orbit = orbits[0]
        self.assertEqual(orbit["mobility_mode"], "fixed")
        self.assertEqual(orbit["group_transform_ids"], [0, 1, 2])
        self.assertEqual(
            orbit["group_registry_transform_ids"],
            ["C3:e", "C3:r1", "C3:r2"],
        )
        self.assertEqual(
            orbit["master_group_id"],
            orbit["group_ids"][0],
        )

    def test_emits_bounded_orbit_rigid_metadata_when_requested(
        self,
    ) -> None:
        payload = yaml.safe_load(
            LHD101_CONFIG.read_text(encoding="utf-8")
        )
        payload["interface_seed"]["interfaces"]["ring_interface"][
            "mobility"
        ] = {
            "mode": "orbit_rigid",
            "bounds": {
                "max_translation": 2.0,
                "max_rotation_deg": 10.0,
            },
        }
        config = self.output_directory / "lhd101_c3_mobile.yaml"
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "mobile",
            base_directory=REPOSITORY_ROOT,
        )
        emitted = json.loads(
            outputs.input_path.read_text(encoding="utf-8")
        )[outputs.example_id]
        orbit = emitted["extra"]["motif_constraint_orbits"][0]

        self.assertEqual(orbit["mobility_mode"], "orbit_rigid")
        self.assertEqual(orbit["max_translation"], 2.0)
        self.assertEqual(orbit["max_rotation_deg"], 10.0)

    def test_emits_complete_cross_chain_runtime_constraint_groups(
        self,
    ) -> None:
        groups = self.payload["extra"]["motif_constraint_groups"]

        self.assertEqual(len(groups), 3)
        resolved_pairs = []
        for group in groups:
            self.assertEqual(group["orbit_id"], "primary_orbit")
            self.assertEqual(
                {member["role"] for member in group["members"]},
                {"left", "right"},
            )
            self.assertEqual(len(group["members"]), 2)
            resolved_pairs.append(
                {
                    (
                        tuple(member["src_components"]),
                        member["sym_transform_id"],
                    )
                    for member in group["members"]
                }
            )
        self.assertEqual(
            resolved_pairs,
            [
                {
                    (tuple(f"B{i}" for i in range(1, 32)), 0),
                    (tuple(f"C{i}" for i in range(1, 31)), 2),
                },
                {
                    (tuple(f"B{i}" for i in range(1, 32)), 1),
                    (tuple(f"C{i}" for i in range(1, 31)), 0),
                },
                {
                    (tuple(f"B{i}" for i in range(1, 32)), 2),
                    (tuple(f"C{i}" for i in range(1, 31)), 1),
                },
            ],
        )

    def test_rebuilds_exact_joint_sample_from_candidate_manifest(self) -> None:
        candidate_directory = self.output_directory / "candidate"
        candidate = compile_standalone(
            LHD101_CONFIG,
            candidate_directory,
            base_directory=REPOSITORY_ROOT,
            random_seed=2101,
            sample_overrides={
                "primary_seed": {
                    "radius_unit": 0.8,
                    "axial_offset_unit": 0.5,
                    "so3_unit": [0.2, 0.4, 0.6],
                }
            },
        )
        rebuilt = compile_rfd3_input(
            LHD101_CONFIG,
            self.output_directory / "rebuilt-candidate",
            base_directory=REPOSITORY_ROOT,
            pose_candidate_manifest=candidate.manifest_path,
        )

        self.assertEqual(
            candidate.structure_path.read_bytes(),
            rebuilt.structure_path.read_bytes(),
        )
        emitted = json.loads(
            rebuilt.input_path.read_text(encoding="utf-8")
        )[rebuilt.example_id]
        self.assertEqual(emitted["extra"]["pose_source"], "candidate_manifest")
        self.assertEqual(
            emitted["extra"]["pose_candidate_structure_sha256"],
            emitted["extra"]["adapter_structure_sha256"],
        )

    def test_emits_native_d2_input_from_dihedral_config(self) -> None:
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        transform_set = payload["interface_seed"]["symmetry"][
            "transform_sets"
        ]["ring_c3"]
        transform_set.update(
            {
                "type": "dihedral",
                "order": 2,
                "secondary_axis": [1.0, 0.0, 0.0],
            }
        )
        payload["interface_seed"]["initialization"]["primary_seed"][
            "placement"
        ]["axial_offset"] = {"mean": 40.0, "range": 0.0}
        config = self.output_directory / "lhd101_d2.yaml"
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "d2",
            base_directory=REPOSITORY_ROOT,
            example_id="lhd101_d2_interface_seed",
        )
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            outputs.example_id
        ]

        self.assertEqual(
            emitted["symmetry"],
            {"id": "D2", "is_symmetric_motif": True},
        )
        self.assertEqual(emitted["extra"]["symmetry_multiplicity"], 4)
        self.assertEqual(
            emitted["extra"]["mosaic_transform_order"],
            ["D2:e", "D2:r1", "D2:s0", "D2:s1"],
        )
        self.assertEqual(emitted["extra"]["registry_preflight"], "passed")

    def test_chain_break_emits_independent_asu_chains_without_linker(self) -> None:
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        link = payload["interface_seed"]["scaffold_links"]["protomer"]
        link["chain_break"] = True
        link["length"] = {"minimum": 0, "maximum": 0}
        link["copy_relation"] = {"orbit_offset": 0}
        config = self.output_directory / "lhd101_c3_no_linker.yaml"
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "no-linker",
            base_directory=REPOSITORY_ROOT,
            example_id="lhd101_c3_no_linker",
        )
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            outputs.example_id
        ]

        self.assertEqual(emitted["contig"], "B1-31,/0,A1-30")
        self.assertEqual(
            emitted["extra"]["scaffold_mode"],
            "independent_chains",
        )
        self.assertEqual(emitted["extra"]["asu_chain_count"], 2)
        self.assertIsNone(
            emitted["extra"]["materialized_linker_length"]
        )
        self.assertEqual(
            emitted["extra"]["linker_length_policy"],
            "not_applicable",
        )
        self.assertEqual(
            emitted["extra"]["materialized_linker_contour_preflight"][
                "status"
            ],
            "not_applicable",
        )

    def test_tracked_dihedral_dryruns_compile_to_native_inputs(self) -> None:
        cases = (
            (
                LHD101_D2_DRYRUN_CONFIG,
                "D2",
                4,
                ["D2:e", "D2:r1", "D2:s0", "D2:s1"],
            ),
            (
                LHD101_D3_DRYRUN_CONFIG,
                "D3",
                6,
                [
                    "D3:e",
                    "D3:r1",
                    "D3:r2",
                    "D3:s0",
                    "D3:s1",
                    "D3:s2",
                ],
            ),
        )
        for config, symmetry_id, multiplicity, transform_order in cases:
            with self.subTest(symmetry_id=symmetry_id):
                outputs = compile_rfd3_input(
                    config,
                    self.output_directory / f"tracked-{symmetry_id.lower()}",
                    base_directory=REPOSITORY_ROOT,
                    example_id=f"tracked_{symmetry_id.lower()}_dryrun",
                )
                emitted = json.loads(
                    outputs.input_path.read_text(encoding="utf-8")
                )[outputs.example_id]
                self.assertEqual(emitted["symmetry"]["id"], symmetry_id)
                self.assertEqual(
                    emitted["extra"]["symmetry_multiplicity"],
                    multiplicity,
                )
                self.assertEqual(
                    emitted["extra"]["registry_transform_order"],
                    transform_order,
                )

    def test_c5_c6_c7_compile_to_native_cyclic_inputs(self) -> None:
        for order, config in LHD101_CYCLIC_CONFIGS.items():
            with self.subTest(order=order):
                outputs = compile_rfd3_input(
                    config,
                    self.output_directory / f"tracked-c{order}",
                    base_directory=REPOSITORY_ROOT,
                    example_id=f"lhd101_c{order}_interface_seed",
                )
                emitted = json.loads(
                    outputs.input_path.read_text(encoding="utf-8")
                )[outputs.example_id]
                extra = emitted["extra"]

                self.assertEqual(emitted["symmetry"]["id"], f"C{order}")
                self.assertEqual(
                    extra["symmetry_multiplicity"],
                    order,
                )
                self.assertEqual(
                    extra["registry_transform_order"],
                    [
                        f"C{order}:e",
                        *[
                            f"C{order}:r{copy_index}"
                            for copy_index in range(1, order)
                        ],
                    ],
                )
                self.assertEqual(
                    len(
                        extra["materialized_linker_contour_preflight"][
                            "evaluated_link_instances"
                        ]
                    ),
                    order,
                )


if __name__ == "__main__":
    unittest.main()
