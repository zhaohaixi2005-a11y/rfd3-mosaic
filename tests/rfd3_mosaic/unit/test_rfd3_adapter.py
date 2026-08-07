import json
import tempfile
import unittest
from pathlib import Path

import yaml

from rfd3_mosaic.output import compile_rfd3_input, compile_standalone
from rfd3_mosaic.output.rfd3_adapter import (
    _native_symmetry_id_and_multiplicity,
    _selector_source_components,
)
from rfd3_mosaic.schema.specs import (
    SymmetryTransformSetSpec,
    SymmetryType,
)


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
LHD101_D3_TWO_ORBIT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/dihedral/"
    "lhd101_d3_two_orbit_engineering.yaml"
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

    def test_high_order_native_symmetry_is_not_capped_at_ten_copies(
        self,
    ) -> None:
        cases = (
            (SymmetryType.CYCLIC, 12, "C12", 12),
            (SymmetryType.DIHEDRAL, 6, "D6", 12),
        )
        for symmetry_type, order, symmetry_id, multiplicity in cases:
            with self.subTest(symmetry_id=symmetry_id):
                specification = SymmetryTransformSetSpec(
                    type=symmetry_type,
                    order=order,
                    secondary_axis=(1.0, 0.0, 0.0)
                    if symmetry_type == SymmetryType.DIHEDRAL
                    else None,
                )
                self.assertEqual(
                    _native_symmetry_id_and_multiplicity(specification),
                    (symmetry_id, multiplicity),
                )

    def test_polyhedral_symmetry_uses_complete_declared_multiplicity(
        self,
    ) -> None:
        cases = (
            (SymmetryType.TETRAHEDRAL, 12, "T"),
            (SymmetryType.OCTAHEDRAL, 24, "O"),
            (SymmetryType.ICOSAHEDRAL, 60, "I"),
        )
        for symmetry_type, order, symmetry_id in cases:
            with self.subTest(symmetry_id=symmetry_id):
                specification = SymmetryTransformSetSpec(
                    type=symmetry_type,
                    order=order,
                )
                self.assertEqual(
                    _native_symmetry_id_and_multiplicity(specification),
                    (symmetry_id, order),
                )

    def test_compiles_central_terminal_extensions_through_native_ir(
        self,
    ) -> None:
        source = self.output_directory / "central_motif.pdb"
        source.write_text(
            "".join(
                (
                    "ATOM      1  N   ALA A   1       9.000   0.000   0.000"
                    "  1.00 20.00           N  \n",
                    "ATOM      2  CA  ALA A   1      10.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      3  C   ALA A   1      11.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      4  N   GLY A   2      12.000   0.000   0.000"
                    "  1.00 20.00           N  \n",
                    "ATOM      5  CA  GLY A   2      13.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      6  C   GLY A   2      14.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "END\n",
                )
            ),
            encoding="utf-8",
        )
        config = self.output_directory / "central.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "assembly": {
                        "schema_version": 2,
                        "mode": "constraint_assembly",
                        "fragments": {
                            "motif": {
                                "source": str(source),
                                "selection": "A/1-2/*",
                                "entity_type": "protein",
                                "role": "functional_motif",
                                "fixed_atoms": "all",
                            }
                        },
                        "motion_groups": {
                            "motif_group": {
                                "members": ["motif"],
                                "mode": "fixed",
                            }
                        },
                        "symmetry": {
                            "transform_sets": {
                                "ring": {"type": "cyclic", "order": 3}
                            },
                            "orbits": {
                                "motif_orbit": {
                                    "transform_set": "ring",
                                    "master_groups": ["motif_group"],
                                }
                            },
                        },
                        "generated_segments": {
                            "n_flank": {
                                "anchor": {
                                    "fragment": "motif",
                                    "terminus": "N",
                                },
                                "length": {"minimum": 5, "maximum": 5},
                            },
                            "c_flank": {
                                "anchor": {
                                    "fragment": "motif",
                                    "terminus": "C",
                                },
                                "length": {"minimum": 7, "maximum": 7},
                            },
                        },
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "central-output",
            example_id="central-c3",
        )
        emitted = json.loads(outputs.input_path.read_text())["central-c3"]

        self.assertEqual(emitted["contig"], "5-5,A1-2,7-7")
        self.assertEqual(emitted["select_fixed_atoms"], {"A1-2": "ALL"})
        self.assertEqual(
            emitted["extra"]["scaffold_mode"],
            "terminal_extensions",
        )
        self.assertEqual(
            {
                group["constraint_kind"]
                for group in emitted["extra"][
                    "motif_constraint_groups"
                ]
            },
            {"fixed_motif"},
        )
        self.assertEqual(
            emitted["extra"]["motif_constraint_orbits"][0][
                "group_transform_ids"
            ],
            [0, 1, 2],
        )
        self.assertTrue(emitted["symmetry"]["use_declared_frames"])

    def test_compiles_tetrahedral_terminal_design_with_declared_frames(
        self,
    ) -> None:
        source = self.output_directory / "tetrahedral_motif.pdb"
        source.write_text(
            "".join(
                (
                    "ATOM      1  N   ALA A   1      59.000  20.000  10.000"
                    "  1.00 20.00           N  \n",
                    "ATOM      2  CA  ALA A   1      60.000  20.000  10.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      3  C   ALA A   1      61.000  20.000  10.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      4  N   GLY A   2      62.000  20.000  10.000"
                    "  1.00 20.00           N  \n",
                    "ATOM      5  CA  GLY A   2      63.000  20.000  10.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      6  C   GLY A   2      64.000  20.000  10.000"
                    "  1.00 20.00           C  \n",
                    "END\n",
                )
            ),
            encoding="utf-8",
        )
        config = self.output_directory / "tetrahedral.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "assembly": {
                        "schema_version": 2,
                        "mode": "constraint_assembly",
                        "fragments": {
                            "motif": {
                                "source": str(source),
                                "selection": "A/1-2/*",
                                "entity_type": "protein",
                                "role": "functional_motif",
                                "fixed_atoms": "all",
                            }
                        },
                        "motion_groups": {
                            "motif_group": {
                                "members": ["motif"],
                                "mode": "fixed",
                            }
                        },
                        "symmetry": {
                            "transform_sets": {
                                "cage": {
                                    "type": "tetrahedral",
                                    "order": 12,
                                }
                            },
                            "orbits": {
                                "motif_orbit": {
                                    "transform_set": "cage",
                                    "master_groups": ["motif_group"],
                                }
                            },
                        },
                        "generated_segments": {
                            "n_flank": {
                                "anchor": {
                                    "fragment": "motif",
                                    "terminus": "N",
                                },
                                "length": {"minimum": 5, "maximum": 5},
                            },
                        },
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "tetrahedral-output",
            example_id="tetrahedral-terminal",
        )
        emitted = json.loads(outputs.input_path.read_text())[
            "tetrahedral-terminal"
        ]

        self.assertEqual(emitted["symmetry"]["id"], "T")
        self.assertTrue(emitted["symmetry"]["use_declared_frames"])
        self.assertEqual(
            len(emitted["symmetry"]["declared_transform_order"]),
            12,
        )
        self.assertEqual(
            len(emitted["symmetry"]["declared_transform_matrices"]),
            12,
        )
        self.assertEqual(
            emitted["extra"]["motif_constraint_orbits"][0][
                "group_transform_ids"
            ],
            list(range(12)),
        )

    def test_public_between_path_emits_joint_fixed_constraint_orbit(
        self,
    ) -> None:
        source = self.output_directory / "public_fixed.pdb"
        source.write_text(
            "".join(
                (
                    "ATOM      1  N   ALA A   1       9.000   0.000   0.000"
                    "  1.00 20.00           N  \n",
                    "ATOM      2  CA  ALA A   1      10.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      3  C   ALA A   1      11.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      4  N   GLY A   2      20.000   0.000   0.000"
                    "  1.00 20.00           N  \n",
                    "ATOM      5  CA  GLY A   2      21.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "ATOM      6  C   GLY A   2      22.000   0.000   0.000"
                    "  1.00 20.00           C  \n",
                    "END\n",
                )
            ),
            encoding="utf-8",
        )
        config = self.output_directory / "public_fixed.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "assembly": {
                        "schema_version": 2,
                        "mode": "constraint_assembly",
                        "fragments": {
                            "left": {
                                "source": str(source),
                                "selection": "A/1-1/*",
                                "entity_type": "protein",
                                "role": "functional_motif",
                                "fixed_atoms": "all",
                            },
                            "right": {
                                "source": str(source),
                                "selection": "A/2-2/*",
                                "entity_type": "protein",
                                "role": "functional_motif",
                                "fixed_atoms": "all",
                            },
                        },
                        "motion_groups": {
                            "motif_group": {
                                "members": ["left", "right"],
                                "mode": "fixed",
                            }
                        },
                        "symmetry": {
                            "transform_sets": {
                                "ring": {"type": "cyclic", "order": 3}
                            },
                            "orbits": {
                                "motif_orbit": {
                                    "transform_set": "ring",
                                    "master_groups": ["motif_group"],
                                }
                            },
                        },
                        "generated_segments": {
                            "middle": {
                                "from_endpoint": {
                                    "fragment": "left",
                                    "terminus": "C",
                                },
                                "to_endpoint": {
                                    "fragment": "right",
                                    "terminus": "N",
                                },
                                "length": {"minimum": 5, "maximum": 5},
                            }
                        },
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "public-fixed-output",
            example_id="public-fixed-c3",
        )
        emitted = json.loads(outputs.input_path.read_text())["public-fixed-c3"]
        groups = emitted["extra"]["motif_constraint_groups"]
        orbit = emitted["extra"]["motif_constraint_orbits"][0]

        self.assertEqual(len(groups), 3)
        self.assertEqual(
            [group["members"][0]["sym_transform_id"] for group in groups],
            [0, 1, 2],
        )
        for group in groups:
            self.assertEqual(group["constraint_kind"], "fixed_motif")
            self.assertEqual(
                {member["source_fragment_id"] for member in group["members"]},
                {"left", "right"},
            )
            self.assertEqual(
                {member["role"] for member in group["members"]},
                {"motif"},
            )
        self.assertEqual(orbit["group_transform_ids"], [0, 1, 2])
        self.assertEqual(orbit["mobility_mode"], "fixed")
        self.assertTrue(emitted["symmetry"]["use_declared_frames"])

        independent = yaml.safe_load(config.read_text(encoding="utf-8"))
        assembly = independent["assembly"]
        assembly["motion_groups"] = {
            "left_component": {
                "members": ["left"],
                "mode": "fixed",
            },
            "right_component": {
                "members": ["right"],
                "mode": "fixed",
            },
        }
        assembly["symmetry"]["orbits"]["motif_orbit"][
            "master_groups"
        ] = ["left_component", "right_component"]
        assembly["symmetry"]["orbits"]["motif_orbit"][
            "component_mobility"
        ] = {
            "left_component": {
                "mode": "orbit_rigid",
                "bounds": {
                    "max_translation": 3.0,
                    "max_rotation_deg": 10.0,
                },
                "subspace": "bounded_se3",
                "proposal": "denoiser_fit",
            }
        }
        assembly["constraint_group_strategy"] = "motion_groups"
        diagnostic_frame = {
            "method": "precomputed",
            "transform": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        assembly["ports"] = {
            "left_port": {
                "group": "left_component",
                "fragments": ["left"],
                "atoms": "heavy",
                "frame": diagnostic_frame,
            },
            "right_port": {
                "group": "right_component",
                "fragments": ["right"],
                "atoms": "heavy",
                "frame": diagnostic_frame,
            },
        }
        assembly["interfaces"] = {
            "diagnostic_relation": {
                "left_port": "left_port",
                "right_port": "right_port",
                "copy_relation": {"orbit_offset": 0},
                "required": False,
                "target_geometry": {
                    "mode": "geometric_constraints",
                    "contacts": {
                        "min_heavy_atom_contacts": 0,
                        "cutoff": 4.5,
                    },
                },
            }
        }
        independent_path = self.output_directory / "public_independent.yaml"
        independent_path.write_text(
            yaml.safe_dump(independent, sort_keys=False),
            encoding="utf-8",
        )
        independent_outputs = compile_rfd3_input(
            independent_path,
            self.output_directory / "public-independent-output",
            example_id="public-independent-c3",
        )
        independent_emitted = json.loads(
            independent_outputs.input_path.read_text()
        )["public-independent-c3"]
        independent_groups = independent_emitted["extra"][
            "motif_constraint_groups"
        ]
        independent_orbits = independent_emitted["extra"][
            "motif_constraint_orbits"
        ]
        relation_plan = independent_emitted["extra"][
            "assembly_interface_relations"
        ]

        self.assertEqual(len(independent_groups), 6)
        self.assertEqual(len(independent_orbits), 2)
        self.assertEqual(
            {orbit["coupling_group_id"] for orbit in independent_orbits},
            {"left_component", "right_component"},
        )
        self.assertEqual(len(relation_plan), 3)
        self.assertEqual(
            {edge["source_interface_id"] for edge in relation_plan},
            {"diagnostic_relation"},
        )
        self.assertEqual(
            {edge["source_copy_index"] for edge in relation_plan},
            {0, 1, 2},
        )
        self.assertEqual(
            {edge["reference_basis"] for edge in relation_plan},
            {"declared_target_geometry"},
        )
        source_components_by_group = {
            orbit["coupling_group_id"]: orbit["source_components"]
            for orbit in independent_orbits
        }
        for edge in relation_plan:
            observed_left = [
                component
                for selector in edge["left_source_components"]
                for component in _selector_source_components(selector)
            ]
            observed_right = [
                component
                for selector in edge["right_source_components"]
                for component in _selector_source_components(selector)
            ]
            self.assertEqual(
                observed_left,
                source_components_by_group["left_component"],
            )
            self.assertEqual(
                observed_right,
                source_components_by_group["right_component"],
            )
        self.assertTrue(
            all(
                len(group["members"]) == 1
                and group["geometry_lock"] == "joint_rigid"
                and group["constraint_kind"] == "fixed_motif"
                for group in independent_groups
            )
        )
        mobility_by_component = {
            orbit["coupling_group_id"]: orbit["mobility_mode"]
            for orbit in independent_orbits
        }
        self.assertEqual(
            mobility_by_component,
            {
                "left_component": "orbit_rigid",
                "right_component": "fixed",
            },
        )

    def test_c12_compiles_to_a_native_input(self) -> None:
        config = yaml.safe_load(
            LHD101_CYCLIC_CONFIGS[5].read_text(encoding="utf-8")
        )
        interface_seed = config["interface_seed"]
        transform_set = next(
            iter(interface_seed["symmetry"]["transform_sets"].values())
        )
        transform_set["order"] = 12

        # Preserve the configured C5 adjacent-copy chord length while
        # increasing the ring order.
        interface_seed["initialization"]["primary_seed"]["placement"][
            "radius"
        ]["mean"] = 83.68

        config_path = self.output_directory / "lhd101_c12.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        outputs = compile_rfd3_input(
            config_path,
            self.output_directory / "tracked-c12",
            base_directory=REPOSITORY_ROOT,
            example_id="lhd101_c12_interface_seed",
        )
        emitted = json.loads(
            outputs.input_path.read_text(encoding="utf-8")
        )[outputs.example_id]

        self.assertEqual(emitted["symmetry"]["id"], "C12")
        self.assertEqual(emitted["extra"]["symmetry_multiplicity"], 12)
        self.assertEqual(
            len(emitted["extra"]["registry_transform_order"]),
            12,
        )
        self.assertEqual(
            len(
                emitted["extra"]["materialized_linker_contour_preflight"][
                    "evaluated_link_instances"
                ]
            ),
            12,
        )

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
        self.assertIsNone(orbit["mobility_subspace"])
        self.assertIsNone(orbit["mobility_proposal"])
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
            "schedule": {
                "start_fraction": 0.05,
                "end_fraction": 0.75,
                "response": 0.2,
                "max_step_translation": 0.25,
                "max_step_rotation_deg": 1.0,
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
        self.assertEqual(orbit["mobility_subspace"], "bounded_se3")
        self.assertEqual(orbit["mobility_proposal"], "denoiser_fit")
        self.assertEqual(
            orbit["mobility_schedule"]["max_step_rotation_deg"],
            1.0,
        )

    def test_emits_complete_cross_chain_runtime_constraint_groups(
        self,
    ) -> None:
        groups = self.payload["extra"]["motif_constraint_groups"]

        self.assertEqual(len(groups), 3)
        resolved_pairs = []
        for group in groups:
            self.assertEqual(group["constraint_kind"], "interface")
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

    def test_d3_two_orbit_engineering_input_emits_two_asu_segments(
        self,
    ) -> None:
        outputs = compile_rfd3_input(
            LHD101_D3_TWO_ORBIT_CONFIG,
            self.output_directory / "d3-two-orbit",
            base_directory=REPOSITORY_ROOT,
            example_id="lhd101_d3_two_orbit_engineering",
        )
        emitted = json.loads(
            outputs.input_path.read_text(encoding="utf-8")
        )[outputs.example_id]
        extra = emitted["extra"]

        self.assertEqual(emitted["symmetry"]["id"], "D3")
        self.assertTrue(emitted["symmetry"]["use_declared_frames"])
        self.assertEqual(extra["symmetry_multiplicity"], 6)
        self.assertEqual(extra["asu_chain_count"], 2)
        self.assertEqual(
            extra["scaffold_mode"],
            "multiple_asu_scaffold_segments",
        )
        self.assertEqual(emitted["contig"].count("85-85"), 2)
        self.assertEqual(emitted["contig"].count("/0"), 1)
        self.assertEqual(len(emitted["select_fixed_atoms"]), 4)
        self.assertIsNone(extra["asu_scaffold_link_instance"])
        self.assertEqual(len(extra["asu_scaffold_link_instances"]), 2)
        self.assertEqual(len(extra["asu_scaffold_segments"]), 2)
        self.assertEqual(len(extra["motif_constraint_orbits"]), 2)
        self.assertEqual(len(extra["motif_constraint_groups"]), 12)
        self.assertTrue(
            extra["materialized_linker_contour_preflight"]["passed"]
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
