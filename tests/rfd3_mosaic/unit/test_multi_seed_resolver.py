import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rfd3_mosaic.compile import expand_symmetry_instances
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.schema import (
    SimpleCageIntentSpec,
    UserDesignSpec,
    UserDesignTask,
    load_simple_cage_intent,
)
from rfd3_mosaic.simple_architecture import symmetry_group_action_count
from rfd3_mosaic.simple_resolver import (
    enumerate_simple_design_candidates,
    resolve_simple_intent,
)
from rfd3_mosaic.topology.component_incidence import (
    enumerate_binary_interface_incidence_plans,
)
from rfd3_mosaic.topology.symmetry_connectivity import finite_symmetry_spec


def _atom_line(
    serial: int,
    atom_name: str,
    chain: str,
    residue: int,
    x: float,
    y: float,
    z: float,
) -> str:
    element = atom_name[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2s}\n"
    )


def _write_two_seed_structure(
    path: Path,
    *,
    disconnect_second_seed: bool = False,
    missing_backbone_anchor: bool = False,
    include_third_seed: bool = False,
) -> None:
    locations = {
        "A": (0.0, 0.0),
        "B": (0.0, 3.0),
        "C": (40.0, 0.0),
        "D": (40.0, 20.0 if disconnect_second_seed else 3.0),
    }
    if include_third_seed:
        locations.update({"E": (80.0, 0.0), "F": (80.0, 3.0)})
    lines: list[str] = []
    serial = 1
    for chain, (x_offset, y) in locations.items():
        for residue in range(1, 5):
            atom_layout = (
                ("N", -0.5),
                ("CA", 0.0),
                ("C", 0.5),
                ("CB", 1.0),
            )
            for atom_name, z in atom_layout:
                if (
                    missing_backbone_anchor
                    and chain == "A"
                    and residue == 1
                    and atom_name == "N"
                ):
                    continue
                lines.append(
                    _atom_line(
                        serial,
                        atom_name,
                        chain,
                        residue,
                        x_offset + 2.0 * (residue - 1),
                        y,
                        z,
                    )
                )
                serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _write_polyhedral_c2_cn_interface_seed(
    path: Path,
    *,
    symmetry: str,
    right_valency: int,
) -> None:
    right_id = f"c{right_valency}"
    plan = next(
        item
        for item in enumerate_binary_interface_incidence_plans(
            symmetry=symmetry,
            interface_id="natural_interface",
            left_participant="c2",
            right_participant=right_id,
        )
        if (item.left.valency, item.right.valency) == (2, right_valency)
    )
    registry = build_transform_registry(finite_symmetry_spec(symmetry))
    lines: list[str] = []
    serial = 1
    right_chains = tuple(
        chr(ord("C") + index) for index in range(right_valency)
    )
    for chains, stabilizer_ids, interface_offset in (
        (
            ("A", "B"),
            plan.left.action.stabilizer_transform_ids,
            np.asarray((0.0, 0.0, 0.0)),
        ),
        (
            right_chains,
            plan.right.action.stabilizer_transform_ids,
            np.asarray((0.0, 3.5, 2.0)),
        ),
    ):
        for chain, transform_id in zip(chains, stabilizer_ids, strict=True):
            transform = registry.transform(transform_id)
            for residue in range(1, 7):
                base = np.asarray(
                    (160.0 + 0.3 * residue, 65.0, 1.4 * residue),
                    dtype=np.float64,
                ) + interface_offset
                for atom_name, offset in (
                    ("N", (-0.45, 0.0, -0.35)),
                    ("CA", (0.0, 0.0, 0.0)),
                    ("C", (0.45, 0.0, 0.35)),
                    ("O", (0.70, 0.0, 0.55)),
                ):
                    coordinate = base + np.asarray(offset)
                    coordinate = (
                        coordinate @ transform[:3, :3].T
                        + transform[:3, 3]
                    )
                    lines.append(
                        _atom_line(
                            serial,
                            atom_name,
                            chain,
                            residue,
                            *coordinate,
                        )
                    )
                    serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _write_t_c2_c3_interface_seed(path: Path) -> None:
    _write_polyhedral_c2_cn_interface_seed(
        path,
        symmetry="T",
        right_valency=3,
    )


class MultiSeedSimpleResolverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "two_interface_seeds.pdb"
        _write_two_seed_structure(self.structure)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _intent(
        self,
        *,
        structure: Path | None = None,
    ) -> SimpleCageIntentSpec:
        residue_range = "1-4"
        return SimpleCageIntentSpec.model_validate(
            {
                "name": "ordinary-two-interface-ring",
                "input": structure or self.structure,
                "goal": {
                    "architecture": "ring",
                    "composition": "auto",
                    "symmetry": ["C3"],
                },
                "interface_seeds": {
                    "interface_alpha": {
                        "participants": ["A", "B"],
                        "selectors": {
                            "A": f"A/{residue_range}/*",
                            "B": f"B/{residue_range}/*",
                        },
                        "use": "auto",
                        "geometry": "preserve_exact",
                    },
                    "interface_beta": {
                        "participants": ["C", "D"],
                        "selectors": {
                            "C": f"C/{residue_range}/*",
                            "D": f"D/{residue_range}/*",
                        },
                        "use": "auto",
                        "geometry": "preserve_exact",
                    },
                },
                "generation": {
                    "length": {"minimum": 20, "maximum": 60}
                },
            }
        )

    @staticmethod
    def _signature(candidate) -> tuple[object, ...]:
        return (
            candidate.candidate_id,
            candidate.symmetry,
            candidate.topology_id,
            repr(candidate.connection_order),
            tuple(candidate.unresolved_variables),
            candidate.design.model_dump(mode="json"),
        )

    def test_two_seed_topology_enumeration_is_deterministic(self) -> None:
        intent = self._intent()

        first = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9400,
        )
        second = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9400,
        )

        self.assertTrue(first)
        for candidate in first:
            metadata = candidate.metadata()
            self.assertEqual(
                set(metadata["supplied_interface_ids"]),
                {"interface_alpha", "interface_beta"},
            )
            self.assertEqual(metadata["invented_interface_count"], 0)
            self.assertEqual(
                {
                    interface.hyperedge_id
                    for interface in candidate.design.interfaces
                },
                {"interface_alpha", "interface_beta"},
            )

        # 2 pure path covers x 2 chemical directions x 2 possible closing
        # seams x 2 C3 winding directions.
        self.assertEqual(len(first), 16)
        self.assertEqual(
            tuple(map(self._signature, first)),
            tuple(map(self._signature, second)),
        )
        self.assertEqual(
            len({candidate.candidate_id for candidate in first}),
            len(first),
        )
        self.assertEqual(
            len({candidate.topology_id for candidate in first}),
            len(first),
        )

    def test_multi_seed_interfaces_preserve_multi_helix_participants(
        self,
    ) -> None:
        payload = self._intent().model_dump(mode="json")
        for seed in payload["interface_seeds"].values():
            seed["selectors"] = {
                participant: (
                    f"{participant}/1-2/*,{participant}/3-4/*"
                )
                for participant in seed["participants"]
            }
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9400,
        )

        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(len(candidate.design.components), 2)
        self.assertTrue(
            all(
                len(component.selectors) == 4
                for component in candidate.design.components.values()
            )
        )
        self.assertTrue(
            all(
                len(port.selectors) == 2
                for port in candidate.design.ports.values()
            )
        )
        # Four intra-participant helix links plus the two cross-seed links.
        self.assertEqual(len(candidate.design.connections), 6)
        self.assertEqual(
            len({
                interface.hyperedge_id
                for interface in candidate.design.interfaces
            }),
            2,
        )

    def test_user_declared_polymer_paths_replace_topology_enumeration(
        self,
    ) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["polymer_connections"] = [
            {
                "id": "unit_one",
                "from": "interface_alpha.A",
                "to": "interface_beta.D",
            },
            {
                "id": "unit_two",
                "from": "interface_beta.C",
                "to": "interface_alpha.B",
            },
        ]
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9400,
        )

        # The user fixes the two covalent paths.  Mosaic only assigns which
        # one carries the C3 seam and its +/- winding; it must not enumerate
        # alternative participant pairings or reverse the declared chains.
        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            self.assertEqual(
                candidate.resolution_frontend,
                "user_declared_polymer_paths_v1",
            )
            self.assertEqual(
                {
                    (source, target)
                    for source, target, _ in candidate.polymer_links
                },
                {
                    (
                        "interface_alpha:side_0",
                        "interface_beta:side_1",
                    ),
                    (
                        "interface_beta:side_0",
                        "interface_alpha:side_1",
                    ),
                },
            )
            self.assertEqual(candidate.polymer_units_per_copy, 2)
            self.assertTrue(candidate.topology_search_complete)

    def test_declared_c3_relation_eliminates_seam_enumeration(self) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["polymer_connections"] = [
            {
                "id": "fixed_seam",
                "from": "interface_alpha.A",
                "to": "interface_beta.D",
                "copy_relation": {"orbit_offset": 1},
            },
            {
                "id": "same_copy_link",
                "from": "interface_beta.C",
                "to": "interface_alpha.B",
            },
        ]
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9400,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(
            {connection.id for connection in candidate.design.connections},
            {"fixed_seam", "same_copy_link"},
        )
        relations = {
            connection.id: connection.copy_relation.orbit_offset
            for connection in candidate.design.connections
        }
        self.assertEqual(relations, {
            "fixed_seam": 1,
            "same_copy_link": 0,
        })
        self.assertEqual(candidate.connection_orbit_offset, 1)

    def test_declared_t_generators_freeze_one_topology(self) -> None:
        structure = self.root / "three_interface_seeds.pdb"
        _write_two_seed_structure(structure, include_third_seed=True)
        payload = self._intent(structure=structure).model_dump(mode="json")
        payload["goal"] = {
            "architecture": "cage",
            "composition": "auto",
            "symmetry": ["T"],
            "subunits": {"minimum": 24, "maximum": 24},
        }
        for seed in payload["interface_seeds"].values():
            seed["use"] = {"exact": 12}
        payload["interface_seeds"]["interface_gamma"] = {
            "participants": ["E", "F"],
            "selectors": {"E": "E/1-4/*", "F": "F/1-4/*"},
            "use": {"exact": 12},
            "geometry": "preserve_exact",
        }
        payload["polymer_connections"] = [
            {
                "id": "t_generator_one",
                "from": "interface_alpha.A",
                "to": "interface_beta.D",
                "copy_relation": {"transform": "T:g01"},
            },
            {
                "id": "t_generator_two",
                "from": "interface_beta.D",
                "to": "interface_gamma.E",
                "copy_relation": {"transform": "T:g03"},
            },
            {
                "id": "same_copy_one",
                "from": "interface_gamma.F",
                "to": "interface_beta.C",
            },
            {
                "id": "same_copy_two",
                "from": "interface_beta.C",
                "to": "interface_alpha.B",
            },
        ]
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("T",),
            seed_start=9530,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(
            {
                connection.id: connection.copy_relation.transform
                for connection in candidate.design.connections
                if connection.copy_relation.transform is not None
            },
            {
                "t_generator_one": "T:g01",
                "t_generator_two": "T:g03",
            },
        )
        self.assertEqual(candidate.physical_polymer_unit_count, 24)

    def test_real_three_seed_t_strict_replay_is_executable(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        config = (
            repository
            / "experiments/"
            "lrz_simple_three_seed_t_user_connections_v100_50step_intent.yaml"
        )
        intent = load_simple_cage_intent(config)

        report = resolve_simple_intent(
            intent,
            self.root / "real-three-seed-t-resolution",
            source_path=config,
            symmetry_ids=("T",),
            pose_samples=1,
            seed_start=953,
            timesteps=50,
            top_count=1,
            optimize_poses=True,
            pose_optimize_top=1,
            pose_optimization_levels=1,
        )

        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(
            report["accepted_count"],
            1,
            report["ranking"],
        )
        self.assertEqual(
            report["selected_count"],
            1,
            report["ranking"],
        )
        self.assertEqual(report["replay_failure_count"], 0)
        selected = report["ranking"][0]
        self.assertTrue(selected["replay_validated"])
        self.assertTrue(selected["rfd3_adapter_validated"])
        self.assertTrue(selected["rfd3_adapter_prevalidated"])
        self.assertEqual(selected["physical_polymer_unit_count"], 24)
        self.assertEqual(selected["invented_interface_count"], 0)

    def test_user_declared_paths_must_assign_every_participant(self) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["polymer_connections"] = [
            {
                "from": "interface_alpha.A",
                "to": "interface_beta.D",
            }
        ]
        with self.assertRaisesRegex(
            ValueError,
            "must assign every supplied interface participant",
        ):
            SimpleCageIntentSpec.model_validate(payload)

    def test_three_user_supplied_interfaces_keep_declared_connections(
        self,
    ) -> None:
        structure = self.root / "three_interface_seeds.pdb"
        _write_two_seed_structure(structure, include_third_seed=True)
        payload = self._intent(structure=structure).model_dump(mode="json")
        payload["goal"]["subunits"] = {"minimum": 9, "maximum": 9}
        payload["interface_seeds"]["interface_gamma"] = {
            "participants": ["E", "F"],
            "selectors": {"E": "E/1-4/*", "F": "F/1-4/*"},
            "use": {"exact": 3},
            "geometry": "preserve_exact",
        }
        payload["polymer_connections"] = [
            {
                "from": "interface_alpha.A",
                "to": "interface_beta.D",
            },
            {
                "from": "interface_beta.C",
                "to": "interface_gamma.F",
            },
            {
                "from": "interface_gamma.E",
                "to": "interface_alpha.B",
            },
        ]
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9510,
        )

        # Three possible C3 seam links x two winding directions. No path
        # pairing or chemical-direction alternatives are invented.
        self.assertEqual(len(candidates), 6)
        expected_links = {
            ("interface_alpha:side_0", "interface_beta:side_1"),
            ("interface_beta:side_0", "interface_gamma:side_1"),
            ("interface_gamma:side_0", "interface_alpha:side_1"),
        }
        for candidate in candidates:
            self.assertEqual(
                candidate.resolution_frontend,
                "user_declared_polymer_paths_v1",
            )
            self.assertEqual(
                {
                    (source, target)
                    for source, target, _ in candidate.polymer_links
                },
                expected_links,
            )
            self.assertEqual(candidate.polymer_units_per_copy, 3)
            self.assertEqual(candidate.physical_polymer_unit_count, 9)
            self.assertEqual(
                set(candidate.supplied_interface_ids),
                {
                    "interface_alpha",
                    "interface_beta",
                    "interface_gamma",
                },
            )

    def test_three_supplied_interfaces_form_two_three_face_t_units(
        self,
    ) -> None:
        structure = self.root / "three_interface_seeds.pdb"
        _write_two_seed_structure(structure, include_third_seed=True)
        payload = self._intent(structure=structure).model_dump(mode="json")
        payload["goal"] = {
            "architecture": "cage",
            "composition": "auto",
            "symmetry": ["T"],
            "subunits": {"minimum": 24, "maximum": 24},
        }
        for seed in payload["interface_seeds"].values():
            seed["use"] = {"exact": 12}
        payload["interface_seeds"]["interface_gamma"] = {
            "participants": ["E", "F"],
            "selectors": {"E": "E/1-4/*", "F": "F/1-4/*"},
            "use": {"exact": 12},
            "geometry": "preserve_exact",
        }
        # Two user-authoritative three-face protein units.  Mosaic may assign
        # the independent T generators, but may not alter these paths or
        # invent another non-covalent interface identity.
        payload["polymer_connections"] = [
            {"from": "interface_alpha.A", "to": "interface_beta.D"},
            {"from": "interface_beta.D", "to": "interface_gamma.E"},
            {"from": "interface_gamma.F", "to": "interface_beta.C"},
            {"from": "interface_beta.C", "to": "interface_alpha.B"},
        ]
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("T",),
            seed_start=9530,
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.symmetry, "T")
            self.assertEqual(
                candidate.resolution_frontend,
                "user_declared_polymer_paths_v1",
            )
            self.assertEqual(candidate.polymer_units_per_copy, 2)
            self.assertEqual(candidate.physical_polymer_unit_count, 24)
            self.assertEqual(len(candidate.design.connections), 4)
            self.assertEqual(
                set(candidate.supplied_interface_ids),
                {
                    "interface_alpha",
                    "interface_beta",
                    "interface_gamma",
                },
            )
            self.assertEqual(
                candidate.metadata()["invented_interface_count"],
                0,
            )
            self.assertEqual(
                candidate.expanded_topology_status,
                "valid_interface_unit_graph",
            )

    def test_one_supplied_t_interface_resolves_c2_c3_component_incidence(
        self,
    ) -> None:
        source = self.root / "t_c2_c3_interface.pdb"
        _write_t_c2_c3_interface_seed(source)
        intent = SimpleCageIntentSpec.model_validate({
            "name": "ordinary-t-c2-c3-interface",
            "input": source,
            "goal": {
                "architecture": "cage",
                "composition": "auto",
                "symmetry": ["T"],
            },
            "interface_seeds": {
                "natural_interface": {
                    "participants": ["c2", "c3"],
                    "selectors": {
                        "c2": "A1-2,A5-6,B1-2,B5-6",
                        "c3": "C1-2,C5-6,D1-2,D5-6,E1-2,E5-6",
                    },
                    "use": {"exact": 12},
                    "geometry": "preserve_exact",
                }
            },
            "generation": {"length": {"minimum": 2, "maximum": 2}},
        })

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("T",),
            timesteps=50,
        )

        self.assertGreaterEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(
            candidate.resolution_frontend,
            "supplied_oligomer_interface_incidence_v1",
        )
        self.assertEqual(
            len(
                candidate.design.components[
                    "component__c2"
                ].finite_orbit_action.coset_representative_ids
            ),
            6,
        )
        self.assertEqual(
            len(
                candidate.design.components[
                    "component__c3"
                ].finite_orbit_action.coset_representative_ids
            ),
            4,
        )
        instances = expand_symmetry_instances(
            lower_user_design(candidate.design).specification
        )
        self.assertEqual(len(instances.interfaces), 12)
        self.assertEqual(len(instances.motion_groups), 10)

    def test_one_supplied_t_c2_c3_interface_strictly_replays(self) -> None:
        source = self.root / "t_c2_c3_replay_interface.pdb"
        _write_t_c2_c3_interface_seed(source)
        intent = SimpleCageIntentSpec.model_validate({
            "name": "ordinary-t-c2-c3-interface-replay",
            "input": source,
            "goal": {
                "architecture": "cage",
                "composition": "auto",
                "symmetry": ["T"],
            },
            "interface_seeds": {
                "natural_interface": {
                    "participants": ["c2", "c3"],
                    "selectors": {
                        "c2": "A1-2,A5-6,B1-2,B5-6",
                        "c3": "C1-2,C5-6,D1-2,D5-6,E1-2,E5-6",
                    },
                    "use": {"exact": 12},
                    "geometry": "preserve_exact",
                }
            },
            "generation": {"length": {"minimum": 2, "maximum": 2}},
        })

        report = resolve_simple_intent(
            intent,
            self.root / "t-c2-c3-resolution",
            symmetry_ids=("T",),
            timesteps=50,
            top_count=1,
        )

        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(
            report["accepted_count"],
            1,
            report["ranking"],
        )
        self.assertEqual(
            report["selected_count"],
            1,
            report["ranking"],
        )
        self.assertEqual(report["replay_failure_count"], 0)
        self.assertEqual(
            report["resolver"],
            "rfd3_mosaic.supplied_oligomer_interface_incidence_v1",
        )
        self.assertEqual(
            report["resolved_component_incidence"][
                "component_orbit_multiplicities"
            ],
            {"component__c2": 6, "component__c3": 4},
        )
        selected = report["ranking"][0]
        self.assertTrue(selected["replay_validated"])
        self.assertTrue(selected["rfd3_adapter_validated"])
        self.assertTrue(selected["rfd3_adapter_prevalidated"])
        self.assertEqual(selected["physical_interface_count"], 12)
        self.assertEqual(
            selected["supplied_interface_ids"],
            ["natural_interface"],
        )
        self.assertEqual(selected["invented_interface_count"], 0)

    def test_o_strictly_replays_and_i_fails_at_runtime_chain_capacity(
        self,
    ) -> None:
        cases = (
            ("O", 4, 24, 12, 6),
            ("I", 5, 60, 30, 12),
        )
        for symmetry, right_valency, interfaces, c2_count, right_count in cases:
            with self.subTest(symmetry=symmetry):
                source = self.root / f"{symmetry.lower()}_c2_c{right_valency}.pdb"
                _write_polyhedral_c2_cn_interface_seed(
                    source,
                    symmetry=symmetry,
                    right_valency=right_valency,
                )
                right_id = f"c{right_valency}"
                right_chains = tuple(
                    chr(ord("C") + index)
                    for index in range(right_valency)
                )
                right_selector = ",".join(
                    f"{chain}1-2,{chain}5-6"
                    for chain in right_chains
                )
                intent = SimpleCageIntentSpec.model_validate({
                    "name": f"ordinary-{symmetry.lower()}-c2-{right_id}",
                    "input": source,
                    "goal": {
                        "architecture": "cage",
                        "composition": "auto",
                        "symmetry": [symmetry],
                    },
                    "interface_seeds": {
                        "natural_interface": {
                            "participants": ["c2", right_id],
                            "selectors": {
                                "c2": "A1-2,A5-6,B1-2,B5-6",
                                right_id: right_selector,
                            },
                            "use": {"exact": interfaces},
                            "geometry": "preserve_exact",
                        }
                    },
                    "generation": {
                        "length": {"minimum": 2, "maximum": 2}
                    },
                })

                report = resolve_simple_intent(
                    intent,
                    self.root / f"{symmetry.lower()}-mixed-resolution",
                    symmetry_ids=(symmetry,),
                    timesteps=50,
                    top_count=1,
                )

                selected = report["ranking"][0]
                self.assertEqual(selected["physical_interface_count"], interfaces)
                self.assertEqual(
                    selected["component_orbit_multiplicities"],
                    {
                        "component__c2": c2_count,
                        f"component__{right_id}": right_count,
                    },
                )
                self.assertEqual(selected["invented_interface_count"], 0)
                if symmetry == "O":
                    self.assertEqual(
                        report["selected_count"], 1, report["ranking"]
                    )
                    self.assertTrue(selected["replay_validated"])
                    self.assertTrue(selected["rfd3_adapter_prevalidated"])
                else:
                    self.assertEqual(report["selected_count"], 0)
                    self.assertIn(
                        "120 physical polymer chains",
                        selected["error"],
                    )
                    self.assertIn(
                        "local-neighbourhood backend",
                        selected["error"],
                    )

    def test_two_seed_candidates_use_the_standard_expert_graph(self) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(),
            symmetry_ids=("C3",),
        )

        for candidate in candidates:
            design = candidate.design
            self.assertIsInstance(design, UserDesignSpec)
            self.assertEqual(design.user_mode, "expert")
            self.assertEqual(
                design.task,
                UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY,
            )
            self.assertEqual(design.symmetry, "C3")
            self.assertEqual(len(design.components), 2)
            self.assertEqual(len(design.ports), 4)
            self.assertEqual(len(design.interfaces), 2)
            self.assertEqual(len(design.connections), 2)
            self.assertFalse(design.generation)
            self.assertFalse(design.constraints)

            self.assertEqual(
                {interface.id for interface in design.interfaces},
                {"interface_alpha", "interface_beta"},
            )
            self.assertTrue(
                all(
                    interface.relation.mode == "preserve_input"
                    for interface in design.interfaces
                )
            )
            self.assertTrue(
                all(
                    component.geometry == "joint_rigid"
                    for component in design.components.values()
                )
            )
            lowered = lower_user_design(design)
            self.assertTrue(
                all(
                    edge.satisfaction_stage == "input"
                    for edge in lowered.specification.interfaces.values()
                )
            )

            endpoint_selectors: list[str] = []
            for connection in design.connections:
                source = connection.from_endpoint
                target = connection.to_endpoint
                # A generated polymer edge must join different supplied
                # interface seeds; directly joining the two sides of one
                # seed would destroy the interface/unit distinction.
                self.assertNotEqual(source.component, target.component)
                self.assertIsNotNone(source.selector)
                self.assertIsNotNone(target.selector)
                endpoint_selectors.extend((source.selector, target.selector))
            self.assertEqual(
                set(endpoint_selectors),
                {"A/1-4/*", "B/1-4/*", "C/1-4/*", "D/1-4/*"},
            )
            self.assertEqual(len(endpoint_selectors), 4)
            offsets = [
                connection.copy_relation.orbit_offset
                for connection in design.connections
            ]
            self.assertEqual(sum(offset != 0 for offset in offsets), 1)
            self.assertEqual(sum(abs(offset or 0) for offset in offsets), 1)
            self.assertEqual(
                candidate.expanded_topology_status,
                "valid_interface_unit_graph",
            )
            self.assertEqual(candidate.polymer_units_per_copy, 2)
            self.assertEqual(candidate.physical_polymer_unit_count, 6)
            self.assertIn(candidate.connection_orbit_offset, (-1, 1))
            seam_links = [
                link for link in candidate.polymer_links if link[2] != 0
            ]
            self.assertEqual(len(seam_links), 1)
            self.assertEqual(candidate.connection_order, seam_links[0][:2])
            self.assertEqual(
                candidate.connection_orbit_offset,
                seam_links[0][2],
            )
            self.assertEqual(
                sum(
                    value != "orbit_offset:+0"
                    for value in candidate.metadata()[
                        "neighbour_transforms"
                    ].values()
                ),
                1,
            )

    def test_c2_has_one_unique_winding_direction(self) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["goal"]["symmetry"] = ["C2"]
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C2",),
        )

        # 2 path covers x 2 chemical directions x 2 seams x one unique C2
        # winding direction.
        self.assertEqual(len(candidates), 8)
        self.assertTrue(
            all(
                candidate.connection_orbit_offset == 1
                for candidate in candidates
            )
        )

    def test_binary_seed_cycle_fails_closed_for_noncyclic_group(self) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["goal"] = {
            "architecture": "cage",
            "composition": "auto",
            "symmetry": ["T"],
        }
        intent = SimpleCageIntentSpec.model_validate(payload)

        with self.assertRaisesRegex(ValueError, "cycle rank 1"):
            enumerate_simple_design_candidates(intent)

    def test_three_seed_c3_enumeration_is_complete(self) -> None:
        structure = self.root / "three_interface_seeds.pdb"
        _write_two_seed_structure(structure, include_third_seed=True)
        payload = self._intent(structure=structure).model_dump(mode="json")
        payload["interface_seeds"]["interface_gamma"] = {
            "participants": ["E", "F"],
            "selectors": {"E": "E/1-4/*", "F": "F/1-4/*"},
            "use": "auto",
            "geometry": "preserve_exact",
        }
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
        )

        # Explicit C3 is a direct-design request: preserve the historical
        # bounded two-face topology family rather than starting broad
        # architecture discovery.
        self.assertEqual(len(candidates), 96)
        self.assertTrue(
            all(
                candidate.polymer_units_per_copy == 3
                for candidate in candidates
            )
        )
        self.assertTrue(
            all(
                candidate.physical_polymer_unit_count == 9
                for candidate in candidates
            )
        )

    def test_three_seeds_resolve_two_three_face_polymer_units(self) -> None:
        structure = self.root / "three_interface_seeds.pdb"
        _write_two_seed_structure(structure, include_third_seed=True)
        payload = self._intent(structure=structure).model_dump(mode="json")
        payload["goal"]["symmetry"] = "auto"
        payload["interface_seeds"]["interface_gamma"] = {
            "participants": ["E", "F"],
            "selectors": {"E": "E/1-4/*", "F": "F/1-4/*"},
            "use": "auto",
            "geometry": "preserve_exact",
        }
        intent = SimpleCageIntentSpec.model_validate(payload)
        relation_plan = ({
            "label": "test_winding",
            "relations": (1, 0, 0, 0),
            "primary_index": 0,
            "primary_relation": 1,
        },)

        with patch(
            "rfd3_mosaic.simple_resolver._connection_relation_plans",
            return_value=relation_plan,
        ):
            candidates = enumerate_simple_design_candidates(
                intent,
                symmetry_ids=("C3",),
                max_candidates=256,
            )

        self.assertTrue(candidates)
        multi_face_candidates = [
            candidate
            for candidate in candidates
            if candidate.polymer_units_per_copy == 2
        ]
        self.assertTrue(multi_face_candidates)
        for candidate in multi_face_candidates:
            self.assertEqual(
                candidate.resolution_frontend,
                "multi_face_polymer_unit_v1",
            )
            self.assertEqual(candidate.polymer_units_per_copy, 2)
            self.assertEqual(candidate.physical_polymer_unit_count, 6)
            self.assertEqual(len(candidate.design.connections), 4)
            self.assertTrue(
                all(
                    interface.relation.mode == "preserve_input"
                    for interface in candidate.design.interfaces
                )
            )
            self.assertTrue(
                all(
                    component.geometry == "joint_rigid"
                    and component.pose.mode == "fixed"
                    for component in candidate.design.components.values()
                )
            )
            self.assertTrue(
                all(
                    connection.from_endpoint.component
                    != connection.to_endpoint.component
                    for connection in candidate.design.connections
                )
            )
            self.assertEqual(
                candidate.expanded_topology_status,
                "valid_interface_unit_graph",
            )
            self.assertFalse(candidate.preflight_failures)

    def test_two_three_participant_seeds_lower_as_exact_hyperedges(
        self,
    ) -> None:
        structure = self.root / "two_three_way_seeds.pdb"
        lines: list[str] = []
        serial = 1
        for chain_index, chain in enumerate("ABCDEF"):
            seed_offset = 0.0 if chain_index < 3 else 40.0
            participant_index = chain_index % 3
            for residue in range(1, 5):
                for atom_name, z in (
                    ("N", -0.5),
                    ("CA", 0.0),
                    ("C", 0.5),
                    ("CB", 1.0),
                ):
                    lines.append(
                        _atom_line(
                            serial,
                            atom_name,
                            chain,
                            residue,
                            seed_offset + 2.0 * (residue - 1),
                            3.0 * participant_index,
                            z,
                        )
                    )
                    serial += 1
        structure.write_text("".join(lines) + "END\n", encoding="utf-8")
        intent = SimpleCageIntentSpec.model_validate(
            {
                "name": "ordinary-two-three-way-interfaces",
                "input": structure,
                "goal": {
                    "architecture": "auto",
                    "composition": "auto",
                    "symmetry": ["C3", "D3", "T"],
                },
                "interface_seeds": {
                    "interface_alpha": {
                        "participants": ["A", "B", "C"],
                        "selectors": {
                            chain: f"{chain}/1-4/*" for chain in "ABC"
                        },
                        "use": "auto",
                    },
                    "interface_beta": {
                        "participants": ["D", "E", "F"],
                        "selectors": {
                            chain: f"{chain}/1-4/*" for chain in "DEF"
                        },
                        "use": "auto",
                    },
                },
                "generation": {
                    "length": {"minimum": 20, "maximum": 60}
                },
            }
        )

        candidates = enumerate_simple_design_candidates(intent)

        # 3! complete cross-hyperedge matchings x two chemical directions x
        # three seam positions x two C3 winding directions.
        by_symmetry = {
            symmetry: [
                candidate
                for candidate in candidates
                if candidate.symmetry == symmetry
            ]
            for symmetry in ("C3", "D3", "T")
        }
        self.assertEqual(len(by_symmetry["C3"]), 72)
        self.assertTrue(by_symmetry["D3"])
        self.assertTrue(by_symmetry["T"])
        self.assertEqual(
            {
                candidate.physical_polymer_unit_count
                for candidate in by_symmetry["D3"]
            },
            {18},
        )
        self.assertEqual(
            {
                candidate.physical_polymer_unit_count
                for candidate in by_symmetry["T"]
            },
            {36},
        )
        for candidate in candidates:
            self.assertEqual(len(candidate.design.components), 2)
            self.assertEqual(len(candidate.design.ports), 6)
            self.assertEqual(len(candidate.design.interfaces), 2)
            self.assertEqual(len(candidate.design.connections), 3)
            self.assertEqual(candidate.polymer_units_per_copy, 3)
            self.assertEqual(
                candidate.physical_polymer_unit_count,
                3 * symmetry_group_action_count(candidate.symmetry),
            )
            self.assertEqual(
                dict(candidate.interface_hyperedges),
                {
                    "interface_alpha": (
                        "interface_alpha__member_01",
                        "interface_alpha__member_02",
                    ),
                    "interface_beta": (
                        "interface_beta__member_01",
                        "interface_beta__member_02",
                    ),
                },
            )
            self.assertTrue(
                all(
                    component.geometry == "joint_rigid"
                    for component in candidate.design.components.values()
                )
            )
            self.assertTrue(
                all(
                    len(interface.between) == 3
                    and len(interface.contact_pairs) == 2
                    for interface in candidate.design.interfaces
                )
            )
            lowered = lower_user_design(candidate.design)
            self.assertEqual(len(lowered.specification.interfaces), 4)
            self.assertEqual(
                {
                    edge.hyperedge_id
                    for edge in lowered.specification.interfaces.values()
                },
                {"interface_alpha", "interface_beta"},
            )

    def test_requested_timesteps_are_frozen_into_every_candidate(self) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(),
            symmetry_ids=("C3",),
            timesteps=50,
        )
        self.assertTrue(
            all(
                candidate.design.sampling.timesteps == 50
                for candidate in candidates
            )
        )
        with self.assertRaisesRegex(ValueError, "timesteps"):
            enumerate_simple_design_candidates(
                self._intent(),
                symmetry_ids=("C3",),
                timesteps=1,
            )

    def test_explicit_mixed_symmetry_request_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Explicit resolver"):
            enumerate_simple_design_candidates(
                self._intent(),
                symmetry_ids=("C3", "T"),
            )

    def test_missing_boundary_backbone_anchor_fails_closed(self) -> None:
        incomplete = self.root / "missing_anchor.pdb"
        _write_two_seed_structure(
            incomplete,
            missing_backbone_anchor=True,
        )

        with patch(
            "rfd3_mosaic.simple_resolver.rank_design_candidates"
        ) as rank, self.assertRaises(ValueError) as context:
            resolve_simple_intent(
                self._intent(structure=incomplete),
                self.root / "missing-anchor-resolution",
            )

        rank.assert_not_called()
        message = str(context.exception).lower()
        self.assertTrue(
            any(
                word in message
                for word in ("termin", "anchor", "backbone")
            ),
            message,
        )

    def test_disconnected_preserve_exact_seed_fails_before_ranking(self) -> None:
        disconnected = self.root / "disconnected_seed.pdb"
        _write_two_seed_structure(
            disconnected,
            disconnect_second_seed=True,
        )

        with patch(
            "rfd3_mosaic.simple_resolver.rank_design_candidates"
        ) as rank, self.assertRaises(ValueError) as context:
            resolve_simple_intent(
                self._intent(structure=disconnected),
                self.root / "resolution",
            )

        rank.assert_not_called()
        message = str(context.exception).lower()
        self.assertTrue(
            any(
                word in message
                for word in ("contact", "connected", "geometry")
            ),
            message,
        )

    def test_candidate_budget_never_silently_selects_one_topology(self) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(),
            symmetry_ids=("C3",),
        )
        self.assertGreater(len(candidates), 1)

        with patch(
            "rfd3_mosaic.simple_resolver.lower_user_design"
        ) as lower, self.assertRaisesRegex(ValueError, "max_candidates"):
            enumerate_simple_design_candidates(
                self._intent(),
                symmetry_ids=("C3",),
                max_candidates=1,
            )
        lower.assert_not_called()

    def test_subunit_range_counts_physical_polymer_units(self) -> None:
        accepted_payload = self._intent().model_dump(mode="json")
        accepted_payload["goal"]["subunits"] = {
            "minimum": 6,
            "maximum": 6,
        }
        accepted = SimpleCageIntentSpec.model_validate(accepted_payload)
        self.assertEqual(
            len(
                enumerate_simple_design_candidates(
                    accepted,
                    symmetry_ids=("C3",),
                )
            ),
            16,
        )

        rejected_payload = self._intent().model_dump(mode="json")
        rejected_payload["goal"]["subunits"] = {
            "minimum": 3,
            "maximum": 3,
        }
        rejected = SimpleCageIntentSpec.model_validate(rejected_payload)
        with self.assertRaisesRegex(ValueError, "polymer units per copy"):
            enumerate_simple_design_candidates(
                rejected,
                symmetry_ids=("C3",),
            )

    def test_real_two_patch_resolution_preserves_source_chain_ownership(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[3]
        structure = (
            repository
            / "examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb"
        )
        intent = SimpleCageIntentSpec.model_validate(
            {
                "name": "real-two-seed-adapter-regression",
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

        report = resolve_simple_intent(
            intent,
            self.root / "real-resolution",
            symmetry_ids=("C3",),
            timesteps=50,
            top_count=2,
        )

        self.assertEqual(report["candidate_count"], 16)
        # Half of the purely combinatorial path covers incorrectly split the
        # two patches from source chain A (and likewise B) across different
        # polymer units.  They are retained as explained rejected candidates
        # but can no longer be submitted as ordinary designs.
        self.assertEqual(report["accepted_count"], 8)
        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(report["replay_failure_count"], 0)
        selected = [
            item for item in report["ranking"] if item.get("resolved_design")
        ]
        self.assertEqual(len(selected), 2)
        self.assertTrue(
            all(item["rfd3_adapter_validated"] for item in selected)
        )
        self.assertTrue(
            all(item["rfd3_adapter_prevalidated"] for item in selected)
        )
        self.assertTrue(
            all(item["rfd3_runtime_atom_count"] > 0 for item in selected)
        )
        self.assertTrue(
            all(
                item["replay_topology"]["is_closed_alternating_cycle"]
                for item in selected
            )
        )

        rejected_split = next(
            item for item in report["ranking"]
            if item["candidate_id"] == "candidate_000007"
        )
        self.assertFalse(rejected_split["accepted"])
        self.assertEqual(len(rejected_split["preflight_failures"]), 2)
        self.assertTrue(
            all(
                "split across different polymer units" in failure
                for failure in rejected_split["preflight_failures"]
            )
        )

        source_preserving = next(
            item for item in selected
            if item["candidate_id"] == "candidate_000012"
        )
        adapter_payload = next(
            iter(
                json.loads(
                    Path(source_preserving["rfd3_adapter_input"]).read_text(
                        encoding="utf-8"
                    )
                ).values()
            )
        )
        self.assertEqual(
            set(adapter_payload["select_fixed_atoms"]),
            {"A1-4", "B1-3", "H1-2", "I1-2"},
        )
        component_groups = [
            group
            for group in adapter_payload["extra"][
                "motif_constraint_groups"
            ]
            if group["coupling_group_id"] == "fixed_component_001"
        ]
        self.assertEqual(
            [
                [
                    (
                        tuple(member["src_components"]),
                        member["sym_transform_id"],
                    )
                    for member in group["members"]
                ]
                for group in component_groups
            ],
            [
                [(('A1', 'A2', 'A3', 'A4'), 0),
                 (('B1', 'B2', 'B3'), 0)],
                [(('A1', 'A2', 'A3', 'A4'), 1),
                 (('B1', 'B2', 'B3'), 1)],
                [(('A1', 'A2', 'A3', 'A4'), 2),
                 (('B1', 'B2', 'B3'), 2)],
            ],
        )


if __name__ == "__main__":
    unittest.main()
