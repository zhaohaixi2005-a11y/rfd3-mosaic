import unittest
from pathlib import Path

from pydantic import ValidationError

from rfd3_mosaic.compile import (
    expand_symmetry_instances,
    load_interface_seed_config,
)
from rfd3_mosaic.provenance import (
    InstanceKind,
    MappingRegistry,
    RFD3IndexMapping,
    build_mapping_registry,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


def make_registry() -> MappingRegistry:
    spec = load_interface_seed_config(LHD101_CONFIG)
    instances = expand_symmetry_instances(spec)
    return build_mapping_registry(spec, instances)


class MappingRegistryTestCase(unittest.TestCase):
    def test_registry_covers_every_compiled_instance(self) -> None:
        registry = make_registry()

        self.assertEqual(len(registry.records), 21)

    def test_source_fragment_maps_to_all_c3_copies(self) -> None:
        registry = make_registry()

        self.assertEqual(
            registry.source_to_instances("left", kind=InstanceKind.FRAGMENT),
            (
                "left@primary_orbit[0]",
                "left@primary_orbit[1]",
                "left@primary_orbit[2]",
            ),
        )

    def test_instance_maps_back_to_source_structure(self) -> None:
        registry = make_registry()

        record = registry.instance_to_source("left@primary_orbit[2]")

        self.assertEqual(record.source_id, "left")
        self.assertEqual(record.source_selection, "A/165-194/*")
        self.assertEqual(record.copy_index, 2)
        self.assertEqual(record.transform_id, "C3:r2")

    def test_group_and_port_membership_queries(self) -> None:
        registry = make_registry()

        self.assertEqual(
            registry.group_fragments("primary_seed@primary_orbit[1]"),
            (
                "left@primary_orbit[1]",
                "right@primary_orbit[1]",
            ),
        )
        self.assertEqual(
            registry.port_fragments("right_port@primary_orbit[1]"),
            ("right@primary_orbit[1]",),
        )
        self.assertEqual(
            registry.link_fragments("protomer@primary_orbit[1]"),
            (
                "right@primary_orbit[1]",
                "left@primary_orbit[2]",
            ),
        )
        self.assertEqual(
            registry.edge_ports("ring_interface@primary_orbit[1]"),
            (
                "left_port@primary_orbit[1]",
                "right_port@primary_orbit[1]",
            ),
        )

    def test_registry_json_round_trip(self) -> None:
        registry = make_registry()

        restored = MappingRegistry.model_validate_json(
            registry.model_dump_json()
        )

        self.assertEqual(restored, registry)

    def test_rfd3_indices_are_attached_without_mutating_registry(self) -> None:
        registry = make_registry()
        mapping = RFD3IndexMapping(
            entity_id="entity_0",
            chain_id="A",
            residue_indices=(0, 1, 2),
            atom_indices=(0, 1, 2, 3),
        )

        updated = registry.with_rfd3_mappings(
            {"left@primary_orbit[0]": mapping}
        )

        self.assertEqual(
            updated.rfd3_indices("left@primary_orbit[0]"),
            mapping,
        )
        with self.assertRaises(KeyError):
            registry.rfd3_indices("left@primary_orbit[0]")

    def test_rfd3_mapping_rejects_unknown_instance(self) -> None:
        registry = make_registry()

        with self.assertRaises(ValidationError):
            registry.with_rfd3_mappings(
                {
                    "missing": RFD3IndexMapping(
                        entity_id="entity_0",
                        chain_id="A",
                        atom_indices=(0,),
                    )
                }
            )

    def test_duplicate_rfd3_atom_ownership_is_rejected(self) -> None:
        registry = make_registry()

        with self.assertRaises(ValidationError):
            registry.with_rfd3_mappings(
                {
                    "left@primary_orbit[0]": RFD3IndexMapping(
                        entity_id="entity_0",
                        chain_id="A",
                        atom_indices=(0, 1),
                    ),
                    "right@primary_orbit[0]": RFD3IndexMapping(
                        entity_id="entity_0",
                        chain_id="A",
                        atom_indices=(1, 2),
                    ),
                }
            )

    def test_unknown_instance_query_is_explicit(self) -> None:
        registry = make_registry()

        with self.assertRaises(KeyError):
            registry.instance_to_source("missing")


if __name__ == "__main__":
    unittest.main()
