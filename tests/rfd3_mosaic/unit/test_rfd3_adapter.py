import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.output import compile_rfd3_input, compile_standalone
from rfd3_mosaic.output.rfd3_adapter import (
    _compile_asu_scaffold_segments,
    _native_symmetry_id_and_multiplicity,
    _selector_source_components,
)
from rfd3_mosaic.rfd3_prevalidate import prevalidate_rfd3_input
from rfd3_mosaic.schema.specs import (
    SymmetryTransformSetSpec,
    SymmetryType,
)
from rfd3_mosaic.topology.component_incidence import (
    enumerate_binary_interface_incidence_plans,
)
from rfd3_mosaic.topology.stabilizer_cosets import (
    stabilizer_coset_hypotheses,
)
from rfd3_mosaic.topology.symmetry_connectivity import finite_symmetry_spec

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = REPOSITORY_ROOT / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
LHD101_D2_DRYRUN_CONFIG = (
    REPOSITORY_ROOT / "configs/rfd3_mosaic/dihedral/lhd101_d2_dryrun.yaml"
)
LHD101_D3_DRYRUN_CONFIG = (
    REPOSITORY_ROOT / "configs/rfd3_mosaic/dihedral/lhd101_d3_dryrun.yaml"
)
LHD101_D3_TWO_ORBIT_CONFIG = (
    REPOSITORY_ROOT / "configs/rfd3_mosaic/dihedral/"
    "lhd101_d3_two_orbit_engineering.yaml"
)
LHD101_CYCLIC_CONFIGS = {
    order: (REPOSITORY_ROOT / f"configs/rfd3_mosaic/cyclic/lhd101_c{order}.yaml")
    for order in (5, 6, 7)
}


class OrderedASUScaffoldPathTestCase(unittest.TestCase):
    """Protect arbitrary-length single-input fixed-fragment paths."""

    @staticmethod
    def _link(
        link_id: str,
        source: str,
        target: str,
        *,
        minimum_length: int = 5,
        maximum_length: int = 5,
        tie_group: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=link_id,
            source_id=link_id,
            from_fragment_instance_id=source,
            to_fragment_instance_id=target,
            chain_break=False,
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            tie_group=tie_group,
            copy_index=0,
            target_copy_index=0,
            orbit_id="motif_orbit",
        )

    @staticmethod
    def _contour_report(*_args, materialized_length, **_kwargs):
        return {
            "status": "passed",
            "passed": True,
            "materialized_linker_length": materialized_length,
            "evaluated_link_instances": [],
        }

    @staticmethod
    def _extension(
        extension_id: str,
        anchor: str,
        terminus: str,
        *,
        minimum_length: int,
        maximum_length: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=extension_id,
            source_id=extension_id,
            anchor_fragment_instance_id=anchor,
            anchor_terminus=SimpleNamespace(value=terminus),
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            tie_group=None,
            orbit_id="motif_orbit",
            copy_index=0,
        )

    def _compile(self, links, *, terminal_extensions=(), orphan_fragments=()):
        selectors = {
            "fragment_a": "A1-2",
            "fragment_b": "B1-2",
            "fragment_c": "C1-2",
            "fragment_d": "D1-2",
        }
        with (
            patch(
                "rfd3_mosaic.output.rfd3_adapter._fragment_selector",
                side_effect=lambda _mapping, fragment_id: selectors[fragment_id],
            ),
            patch(
                "rfd3_mosaic.output.rfd3_adapter."
                "_materialized_linker_contour_preflight",
                side_effect=self._contour_report,
            ),
            patch(
                "rfd3_mosaic.output.rfd3_adapter."
                "_minimum_materialized_linker_length",
                return_value=0,
            ),
        ):
            return _compile_asu_scaffold_segments(
                links,
                terminal_extensions=terminal_extensions,
                orphan_fragments=orphan_fragments,
                mapping={},
                manifest_path=Path("unused-manifest.json"),
                linker_length=None,
            )

    def test_automatic_length_covers_worst_symmetry_instance(self) -> None:
        link = self._link(
            "link_ab",
            "fragment_a",
            "fragment_b",
            minimum_length=5,
            maximum_length=10,
        )
        selectors = {
            "fragment_a": "A1-2",
            "fragment_b": "B1-2",
        }
        with (
            patch(
                "rfd3_mosaic.output.rfd3_adapter._fragment_selector",
                side_effect=lambda _mapping, fragment_id: selectors[fragment_id],
            ),
            patch(
                "rfd3_mosaic.output.rfd3_adapter."
                "_minimum_materialized_linker_length",
                return_value=9,
            ),
            patch(
                "rfd3_mosaic.output.rfd3_adapter."
                "_materialized_linker_contour_preflight",
                side_effect=self._contour_report,
            ),
        ):
            segments = _compile_asu_scaffold_segments(
                (link,),
                mapping={},
                manifest_path=Path("unused-manifest.json"),
                linker_length=None,
            )

        entry = segments[0].links[0]
        self.assertEqual(entry.materialized_linker_length, 9)
        self.assertEqual(
            entry.linker_length_policy,
            "configured_range_contour_sufficient",
        )

    def test_tied_links_materialize_one_common_length(self) -> None:
        links = (
            self._link(
                "link_ab",
                "fragment_a",
                "fragment_b",
                minimum_length=5,
                maximum_length=12,
                tie_group="unit_length",
            ),
            self._link(
                "link_cd",
                "fragment_c",
                "fragment_d",
                minimum_length=7,
                maximum_length=10,
                tie_group="unit_length",
            ),
        )
        selectors = {
            "fragment_a": "A1-2",
            "fragment_b": "B1-2",
            "fragment_c": "C1-2",
            "fragment_d": "D1-2",
        }
        requirements = {"link_ab": 8, "link_cd": 9}
        with (
            patch(
                "rfd3_mosaic.output.rfd3_adapter._fragment_selector",
                side_effect=lambda _mapping, fragment_id: selectors[fragment_id],
            ),
            patch(
                "rfd3_mosaic.output.rfd3_adapter."
                "_minimum_materialized_linker_length",
                side_effect=lambda _path, *, source_link_id: requirements[
                    source_link_id
                ],
            ),
            patch(
                "rfd3_mosaic.output.rfd3_adapter."
                "_materialized_linker_contour_preflight",
                side_effect=self._contour_report,
            ),
        ):
            segments = _compile_asu_scaffold_segments(
                links,
                mapping={},
                manifest_path=Path("unused-manifest.json"),
                linker_length=None,
            )

        lengths = {
            link.materialized_linker_length
            for segment in segments
            for link in segment.links
        }
        policies = {
            link.linker_length_policy for segment in segments for link in segment.links
        }
        self.assertEqual(lengths, {9})
        self.assertEqual(policies, {"tie_group_contour_sufficient"})

    def test_ordered_path_materializes_intermediate_seed_once(self) -> None:
        segments = self._compile(
            (
                self._link(
                    "link_bc",
                    "fragment_b",
                    "fragment_c",
                    minimum_length=7,
                    maximum_length=7,
                ),
                self._link("link_ab", "fragment_a", "fragment_b"),
            )
        )

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(
            segment.fragment_instance_ids,
            ("fragment_a", "fragment_b", "fragment_c"),
        )
        self.assertEqual(
            segment.contig_chains,
            ("A1-2,5-5,B1-2,7-7,C1-2",),
        )
        self.assertEqual(segment.contig_chains[0].count("B1-2"), 1)
        self.assertEqual(
            [entry.link.id for entry in segment.links],
            ["link_ab", "link_bc"],
        )

    def test_ordered_path_length_is_data_driven(self) -> None:
        segments = self._compile(
            (
                self._link("link_ab", "fragment_a", "fragment_b"),
                self._link("link_bc", "fragment_b", "fragment_c"),
                self._link("link_cd", "fragment_c", "fragment_d"),
            )
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(
            segments[0].fragment_instance_ids,
            (
                "fragment_a",
                "fragment_b",
                "fragment_c",
                "fragment_d",
            ),
        )
        self.assertEqual(len(segments[0].links), 3)

    def test_ordered_path_accepts_terminal_extensions_at_both_ends(
        self,
    ) -> None:
        segments = self._compile(
            (self._link("link_ab", "fragment_a", "fragment_b"),),
            terminal_extensions=(
                self._extension(
                    "n_flank",
                    "fragment_a",
                    "N",
                    minimum_length=3,
                    maximum_length=3,
                ),
                self._extension(
                    "c_flank",
                    "fragment_b",
                    "C",
                    minimum_length=4,
                    maximum_length=4,
                ),
            ),
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(
            segments[0].contig_chains,
            ("3-3,A1-2,5-5,B1-2,4-4",),
        )
        self.assertEqual(
            [extension.id for extension in segments[0].terminal_extensions],
            ["c_flank", "n_flank"],
        )

    def test_ordered_path_rejects_extension_on_occupied_internal_end(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "terminus of an internal fixed fragment",
        ):
            self._compile(
                (
                    self._link("link_ab", "fragment_a", "fragment_b"),
                    self._link("link_bc", "fragment_b", "fragment_c"),
                ),
                terminal_extensions=(
                    self._extension(
                        "invalid_n_flank",
                        "fragment_b",
                        "N",
                        minimum_length=3,
                        maximum_length=3,
                    ),
                ),
            )

    def test_fixed_component_fragment_without_link_is_an_independent_path(
        self,
    ) -> None:
        segments = self._compile(
            (self._link("link_ab", "fragment_a", "fragment_b"),),
            orphan_fragments=(
                SimpleNamespace(
                    id="fragment_c",
                    source_id="fragment_c",
                    orbit_id="motif_orbit",
                    copy_index=0,
                ),
            ),
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(
            {segment.contig_chains for segment in segments},
            {("A1-2,5-5,B1-2",), ("C1-2",)},
        )

    def test_ordered_path_rejects_chain_branching(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "two C-terminal outgoing links",
        ):
            self._compile(
                (
                    self._link("link_ab", "fragment_a", "fragment_b"),
                    self._link("link_ac", "fragment_a", "fragment_c"),
                )
            )

    def test_ordered_path_rejects_closed_cycles(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "closed cycle"):
            self._compile(
                (
                    self._link("link_ab", "fragment_a", "fragment_b"),
                    self._link("link_ba", "fragment_b", "fragment_a"),
                )
            )


class RFD3AdapterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.outputs = compile_rfd3_input(
            LHD101_CONFIG,
            self.output_directory,
            base_directory=REPOSITORY_ROOT,
        )
        self.payload = json.loads(self.outputs.input_path.read_text(encoding="utf-8"))[
            self.outputs.example_id
        ]

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
                            "transform_sets": {"ring": {"type": "cyclic", "order": 3}},
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
                for group in emitted["extra"]["motif_constraint_groups"]
            },
            {"fixed_motif"},
        )
        self.assertEqual(
            emitted["extra"]["motif_constraint_orbits"][0]["group_transform_ids"],
            [0, 1, 2],
        )
        self.assertTrue(emitted["symmetry"]["use_declared_frames"])

    def test_native_path_mixes_scaffold_link_and_terminal_extensions(
        self,
    ) -> None:
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        assembly = payload["interface_seed"]
        assembly["scaffold_links"]["protomer"]["copy_relation"] = {"orbit_offset": 0}
        assembly["generated_segments"] = {
            "n_flank": {
                "anchor": {"fragment": "right", "terminus": "N"},
                "length": {"minimum": 3, "maximum": 3},
            },
            "c_flank": {
                "anchor": {"fragment": "left", "terminus": "C"},
                "length": {"minimum": 4, "maximum": 4},
            },
        }
        config = self.output_directory / "mixed-polymer-path.yaml"
        config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "mixed-polymer-path-output",
            base_directory=REPOSITORY_ROOT,
            example_id="mixed-polymer-path-c3",
            linker_length=80,
        )
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            "mixed-polymer-path-c3"
        ]

        self.assertEqual(
            emitted["contig"],
            "3-3,B1-31,80-80,A1-30,4-4",
        )
        self.assertEqual(
            emitted["extra"]["scaffold_mode"],
            "linker_with_terminal_extensions",
        )
        self.assertEqual(
            len(emitted["extra"]["asu_terminal_extensions"]),
            2,
        )
        self.assertEqual(
            emitted["extra"]["asu_scaffold_segments"][0][
                "terminal_extension_instance_ids"
            ],
            ["c_flank@primary_orbit[0]", "n_flank@primary_orbit[0]"],
        )

    def test_compiles_finite_quotient_orbit_into_physical_frames(
        self,
    ) -> None:
        source = self.output_directory / "quotient_motif.pdb"
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
        action = stabilizer_coset_hypotheses("C4", 2)[0]
        config = self.output_directory / "quotient.yaml"
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
                            "transform_sets": {"ring": {"type": "cyclic", "order": 4}},
                            "orbits": {
                                "motif_orbit": {
                                    "transform_set": "ring",
                                    "master_groups": ["motif_group"],
                                    "finite_action": {
                                        "coset_representative_ids": list(
                                            action.coset_representative_ids
                                        ),
                                        "stabilizer_transform_ids": list(
                                            action.stabilizer_transform_ids
                                        ),
                                        "transform_to_coset_representative": dict(
                                            action.transform_to_coset_representative
                                        ),
                                    },
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
            self.output_directory / "quotient-output",
            example_id="quotient-c4-c2",
        )
        emitted = json.loads(outputs.input_path.read_text())["quotient-c4-c2"]

        self.assertEqual(emitted["symmetry"]["id"], "C4")
        self.assertTrue(emitted["symmetry"]["declared_action_is_quotient"])
        self.assertEqual(
            emitted["symmetry"]["declared_transform_order"],
            list(action.coset_representative_ids),
        )
        self.assertEqual(emitted["ori_token"], [0.0, 0.0, 0.0])
        self.assertEqual(emitted["extra"]["symmetry_multiplicity"], 2)
        self.assertEqual(
            emitted["extra"]["full_symmetry_multiplicity"],
            4,
        )
        self.assertEqual(
            emitted["extra"]["symmetry_action_kind"],
            "stabilizer_quotient",
        )
        self.assertEqual(
            emitted["extra"]["motif_constraint_orbits"][0]["group_transform_ids"],
            [0, 1],
        )
        report = prevalidate_rfd3_input(outputs.input_path)
        self.assertEqual(report["expected_multiplicity"], 2)
        self.assertEqual(report["full_symmetry_multiplicity"], 4)
        self.assertEqual(report["symmetry_transform_ids"], [0, 1])

    def test_compiles_mixed_t_c2_c3_component_orbits_for_rfd3(
        self,
    ) -> None:
        plan = next(
            item
            for item in enumerate_binary_interface_incidence_plans(
                symmetry="T",
                interface_id="natural_interface",
                left_participant="c2_component",
                right_participant="c3_component",
            )
            if (item.left.valency, item.right.valency) == (2, 3)
        )
        registry = build_transform_registry(finite_symmetry_spec("T"))
        source = self.output_directory / "mixed_t_c2_c3.pdb"
        lines = []
        serial = 1

        def emit_component(chains, stabilizer_ids, center):
            nonlocal serial
            for chain, transform_id in zip(
                chains,
                stabilizer_ids,
                strict=True,
            ):
                matrix = registry.transform(transform_id)
                for residue in range(1, 7):
                    base_ca = np.asarray(
                        (
                            center[0] + 0.25 * residue,
                            center[1] + 0.15 * residue,
                            center[2] + 1.45 * residue,
                        )
                    )
                    for atom_name, offset, element in (
                        ("N", (-0.55, 0.05, -0.45), "N"),
                        ("CA", (0.0, 0.0, 0.0), "C"),
                        ("C", (0.55, -0.05, 0.45), "C"),
                        ("O", (0.80, -0.10, 0.75), "O"),
                    ):
                        coordinate = base_ca + np.asarray(offset)
                        coordinate = coordinate @ matrix[:3, :3].T + matrix[:3, 3]
                        lines.append(
                            f"ATOM  {serial:5d} {atom_name:>4s} ALA "
                            f"{chain:1s}{residue:4d}    "
                            f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}"
                            f"{coordinate[2]:8.3f}{1.0:6.2f}{20.0:6.2f}"
                            f"          {element:>2s}\n"
                        )
                        serial += 1

        emit_component(
            ("A", "B"),
            plan.left.action.stabilizer_transform_ids,
            (22.0, 7.0, 3.0),
        )
        emit_component(
            ("C", "D", "E"),
            plan.right.action.stabilizer_transform_ids,
            (4.0, 36.0, 11.0),
        )
        lines.append("END\n")
        source.write_text("".join(lines), encoding="utf-8")

        def action_payload(participant):
            action = participant.action
            return {
                "coset_representative_ids": list(action.coset_representative_ids),
                "stabilizer_transform_ids": list(action.stabilizer_transform_ids),
                "transform_to_coset_representative": dict(
                    action.transform_to_coset_representative
                ),
            }

        fragments = {}
        groups = {
            "c2_component": {"members": [], "mode": "fixed"},
            "c3_component": {"members": [], "mode": "fixed"},
        }
        links = {}
        component_fragments = {"c2_component": [], "c3_component": []}
        for component_id, chains in (
            ("c2_component", ("A", "B")),
            ("c3_component", ("C", "D", "E")),
        ):
            for chain in chains:
                chain_fragments = []
                for label, selection in (
                    ("n", f"{chain}/1-2/*"),
                    ("c", f"{chain}/5-6/*"),
                ):
                    fragment_id = f"{component_id}_{chain}_{label}"
                    fragments[fragment_id] = {
                        "source": str(source),
                        "selection": selection,
                        "entity_type": "protein",
                        "role": "interface_motif",
                        "fixed_atoms": "all",
                    }
                    groups[component_id]["members"].append(fragment_id)
                    component_fragments[component_id].append(fragment_id)
                    chain_fragments.append(fragment_id)
                links[f"path_{chain}"] = {
                    "from_endpoint": {
                        "fragment": chain_fragments[0],
                        "terminus": "C",
                    },
                    "to_endpoint": {
                        "fragment": chain_fragments[1],
                        "terminus": "N",
                    },
                    "length": {"minimum": 2, "maximum": 2},
                    "chain_break": False,
                    "copy_relation": {"orbit_offset": 0},
                }

        config = self.output_directory / "mixed-t-c2-c3.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "assembly": {
                        "schema_version": 2,
                        "mode": "constraint_assembly",
                        "constraint_group_strategy": "interface_edges",
                        "fragments": fragments,
                        "motion_groups": groups,
                        "ports": {
                            "c2_port": {
                                "group": "c2_component",
                                "fragments": component_fragments["c2_component"],
                                "atoms": "heavy",
                                "frame": {"method": "reference_interface_pca"},
                            },
                            "c3_port": {
                                "group": "c3_component",
                                "fragments": component_fragments["c3_component"],
                                "atoms": "heavy",
                                "frame": {"method": "reference_interface_pca"},
                            },
                        },
                        "symmetry": {
                            "transform_sets": {
                                "cage": {
                                    "type": "tetrahedral",
                                    "order": 12,
                                }
                            },
                            "orbits": {
                                "c2_orbit": {
                                    "transform_set": "cage",
                                    "master_groups": ["c2_component"],
                                    "finite_action": action_payload(plan.left),
                                },
                                "c3_orbit": {
                                    "transform_set": "cage",
                                    "master_groups": ["c3_component"],
                                    "finite_action": action_payload(plan.right),
                                },
                            },
                        },
                        "interfaces": {
                            "natural_interface": {
                                "left_port": "c2_port",
                                "right_port": "c3_port",
                                "copy_relation": {"orbit_offset": 0},
                                "required": True,
                                "target_geometry": {
                                    "mode": "reference_transform",
                                    "from_reference_seed": True,
                                    "translation_tolerance": 2.0,
                                    "rotation_tolerance_deg": 10.0,
                                },
                            }
                        },
                        "scaffold_links": links,
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        outputs = compile_rfd3_input(
            config,
            self.output_directory / "mixed-t-c2-c3-output",
            example_id="mixed-t-c2-c3",
        )
        emitted = json.loads(outputs.input_path.read_text())["mixed-t-c2-c3"]
        extra = emitted["extra"]
        self.assertEqual(extra["symmetry_action_kind"], "mixed_stabilizer_quotients")
        self.assertEqual(len(extra["preexpanded_chain_layout"]), 24)
        self.assertEqual(len(extra["motif_constraint_groups"]), 12)
        self.assertEqual(len(extra["assembly_interface_relations"]), 12)
        self.assertEqual(emitted["ori_token"], [0.0, 0.0, 0.0])
        report = prevalidate_rfd3_input(outputs.input_path)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["chain_count"], 24)
        self.assertEqual(report["expected_multiplicity"], 12)
        matrix_audit = report["symmetry_transform_matrix_audit"]
        self.assertEqual(
            matrix_audit["coverage_contract"],
            "declared_registry_subset",
        )

        quotient_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        quotient_payload["assembly"]["symmetry"]["orbits"]["c3_orbit"][
            "finite_action"
        ] = action_payload(plan.left)
        quotient_assembly = quotient_payload["assembly"]
        for fragment_id in (
            "c3_component_D_n",
            "c3_component_D_c",
            "c3_component_E_n",
            "c3_component_E_c",
        ):
            quotient_assembly["fragments"].pop(fragment_id)
            quotient_assembly["motion_groups"]["c3_component"][
                "members"
            ].remove(fragment_id)
            quotient_assembly["ports"]["c3_port"]["fragments"].remove(
                fragment_id
            )
        quotient_assembly["scaffold_links"].pop("path_D")
        quotient_assembly["scaffold_links"].pop("path_E")
        quotient_config = self.output_directory / "mixed-t-c2-c2.yaml"
        quotient_config.write_text(
            yaml.safe_dump(quotient_payload, sort_keys=False),
            encoding="utf-8",
        )
        quotient_outputs = compile_rfd3_input(
            quotient_config,
            self.output_directory / "mixed-t-c2-c2-output",
            example_id="mixed-t-c2-c2",
        )
        quotient_emitted = json.loads(
            quotient_outputs.input_path.read_text()
        )["mixed-t-c2-c2"]
        quotient_extra = quotient_emitted["extra"]
        self.assertEqual(
            len(quotient_extra["assembly_interface_relations"]),
            6,
        )
        self.assertEqual(
            {
                relation["edge_stabilizer_order"]
                for relation in quotient_extra[
                    "assembly_interface_relations"
                ]
            },
            {2},
        )
        self.assertEqual(
            len(quotient_extra["motif_constraint_groups"]),
            6,
        )
        quotient_report = prevalidate_rfd3_input(
            quotient_outputs.input_path
        )
        self.assertEqual(quotient_report["status"], "passed")
        self.assertEqual(
            matrix_audit["runtime_transform_count"],
            len(
                {
                    int(record["transform_index"])
                    for record in extra["preexpanded_chain_layout"]
                }
            ),
        )

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
        emitted = json.loads(outputs.input_path.read_text())["tetrahedral-terminal"]

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
            emitted["extra"]["motif_constraint_orbits"][0]["group_transform_ids"],
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
                            "transform_sets": {"ring": {"type": "cyclic", "order": 3}},
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
        assembly["symmetry"]["orbits"]["motif_orbit"]["master_groups"] = [
            "left_component",
            "right_component",
        ]
        assembly["symmetry"]["orbits"]["motif_orbit"]["component_mobility"] = {
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
        independent_emitted = json.loads(independent_outputs.input_path.read_text())[
            "public-independent-c3"
        ]
        independent_groups = independent_emitted["extra"]["motif_constraint_groups"]
        independent_orbits = independent_emitted["extra"]["motif_constraint_orbits"]
        relation_plan = independent_emitted["extra"]["assembly_interface_relations"]

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

    def test_single_input_three_fragment_path_is_emitted_once(self) -> None:
        source = self.output_directory / "three_fragment_seed.pdb"
        atom_lines = []
        serial = 1
        for residue_number, residue_name, origin in (
            (1, "ALA", 9.0),
            (2, "GLY", 19.0),
            (3, "SER", 29.0),
        ):
            for atom_name, offset, element in (
                ("N", 0.0, "N"),
                ("CA", 1.0, "C"),
                ("C", 2.0, "C"),
            ):
                atom_lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} "
                    f"A{residue_number:4d}    {origin + offset:8.3f}"
                    "   0.000   0.000  1.00 20.00           "
                    f"{element:>2s}  \n"
                )
                serial += 1
        atom_lines.append("END\n")
        source.write_text("".join(atom_lines), encoding="utf-8")

        config = self.output_directory / "three_fragment_seed.yaml"
        fragments = {
            name: {
                "source": str(source),
                "selection": f"A/{residue_number}-{residue_number}/*",
                "entity_type": "protein",
                "role": "functional_motif",
                "fixed_atoms": "all",
            }
            for name, residue_number in (
                ("seed_a", 1),
                ("seed_b", 2),
                ("seed_c", 3),
            )
        }
        config.write_text(
            yaml.safe_dump(
                {
                    "assembly": {
                        "schema_version": 2,
                        "mode": "constraint_assembly",
                        "fragments": fragments,
                        "motion_groups": {
                            "seed_geometry": {
                                "members": ["seed_a", "seed_b", "seed_c"],
                                "mode": "fixed",
                            }
                        },
                        "symmetry": {
                            "transform_sets": {"ring": {"type": "cyclic", "order": 3}},
                            "orbits": {
                                "motif_orbit": {
                                    "transform_set": "ring",
                                    "master_groups": ["seed_geometry"],
                                }
                            },
                        },
                        "generated_segments": {
                            "link_ab": {
                                "from_endpoint": {
                                    "fragment": "seed_a",
                                    "terminus": "C",
                                },
                                "to_endpoint": {
                                    "fragment": "seed_b",
                                    "terminus": "N",
                                },
                                "length": {"minimum": 5, "maximum": 5},
                            },
                            "link_bc": {
                                "from_endpoint": {
                                    "fragment": "seed_b",
                                    "terminus": "C",
                                },
                                "to_endpoint": {
                                    "fragment": "seed_c",
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
            self.output_directory / "three-fragment-output",
            example_id="three-fragment-c3",
        )
        emitted = json.loads(outputs.input_path.read_text())["three-fragment-c3"]
        extra = emitted["extra"]

        self.assertEqual(extra["scaffold_mode"], "ordered_asu_scaffold_path")
        self.assertEqual(extra["asu_chain_count"], 1)
        self.assertEqual(len(extra["asu_scaffold_segments"]), 1)
        self.assertEqual(
            extra["asu_scaffold_segments"][0]["source_link_ids"],
            ["link_ab", "link_bc"],
        )
        path_selectors = extra["asu_scaffold_segments"][0]["path_selectors"]
        self.assertEqual(len(path_selectors), 3)
        self.assertEqual(emitted["contig"].count(path_selectors[1]), 1)
        self.assertEqual(len(emitted["select_fixed_atoms"]), 3)

    def test_c12_compiles_to_a_native_input(self) -> None:
        config = yaml.safe_load(LHD101_CYCLIC_CONFIGS[5].read_text(encoding="utf-8"))
        interface_seed = config["interface_seed"]
        transform_set = next(
            iter(interface_seed["symmetry"]["transform_sets"].values())
        )
        transform_set["order"] = 12

        # Preserve the configured C5 adjacent-copy chord length while
        # increasing the ring order.
        interface_seed["initialization"]["primary_seed"]["placement"]["radius"][
            "mean"
        ] = 83.68

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
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            outputs.example_id
        ]

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
                item["materialized_linker_length"] == 85 and item["passed"]
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
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            outputs.example_id
        ]

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
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        payload["interface_seed"]["interfaces"]["ring_interface"]["mobility"] = {
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
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            outputs.example_id
        ]
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
        emitted = json.loads(rebuilt.input_path.read_text(encoding="utf-8"))[
            rebuilt.example_id
        ]
        self.assertEqual(emitted["extra"]["pose_source"], "candidate_manifest")
        self.assertEqual(
            emitted["extra"]["pose_candidate_structure_sha256"],
            emitted["extra"]["adapter_structure_sha256"],
        )

    def test_records_realized_compiler_pose_separately_from_diffusion(
        self,
    ) -> None:
        compiled = compile_rfd3_input(
            LHD101_CONFIG,
            self.output_directory / "pose-provenance",
            base_directory=REPOSITORY_ROOT,
            pose_seed=10063,
        )
        emitted = json.loads(
            compiled.input_path.read_text(encoding="utf-8")
        )[compiled.example_id]
        extra = emitted["extra"]

        self.assertEqual(extra["pose_source"], "compiler_initialization")
        self.assertEqual(extra["pose_seed"], 10063)
        self.assertIn("primary_seed", extra["initialization_samples"])
        sample = extra["initialization_samples"]["primary_seed"]
        self.assertGreaterEqual(sample["sampled_radius"], 20.0)
        self.assertLessEqual(sample["sampled_radius"], 30.0)
        self.assertIsNotNone(sample["quaternion_xyzw"])

    def test_emits_native_d2_input_from_dihedral_config(self) -> None:
        payload = yaml.safe_load(LHD101_CONFIG.read_text(encoding="utf-8"))
        transform_set = payload["interface_seed"]["symmetry"]["transform_sets"][
            "ring_c3"
        ]
        transform_set.update(
            {
                "type": "dihedral",
                "order": 2,
                "secondary_axis": [1.0, 0.0, 0.0],
            }
        )
        payload["interface_seed"]["initialization"]["primary_seed"]["placement"][
            "axial_offset"
        ] = {"mean": 40.0, "range": 0.0}
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
        self.assertIsNone(emitted["extra"]["materialized_linker_length"])
        self.assertEqual(
            emitted["extra"]["linker_length_policy"],
            "not_applicable",
        )
        self.assertEqual(
            emitted["extra"]["materialized_linker_contour_preflight"]["status"],
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
                emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
                    outputs.example_id
                ]
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
        emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
            outputs.example_id
        ]
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
        self.assertTrue(extra["materialized_linker_contour_preflight"]["passed"])

    def test_c5_c6_c7_compile_to_native_cyclic_inputs(self) -> None:
        for order, config in LHD101_CYCLIC_CONFIGS.items():
            with self.subTest(order=order):
                outputs = compile_rfd3_input(
                    config,
                    self.output_directory / f"tracked-c{order}",
                    base_directory=REPOSITORY_ROOT,
                    example_id=f"lhd101_c{order}_interface_seed",
                )
                emitted = json.loads(outputs.input_path.read_text(encoding="utf-8"))[
                    outputs.example_id
                ]
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
                        *[f"C{order}:r{copy_index}" for copy_index in range(1, order)],
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
