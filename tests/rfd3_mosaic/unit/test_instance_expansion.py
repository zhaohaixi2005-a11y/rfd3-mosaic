import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.compile import (
    expand_symmetry_instances,
    load_interface_seed_config,
)
from rfd3_mosaic.schema import InterfaceSeedSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


class InstanceExpansionTestCase(unittest.TestCase):
    def test_lhd101_c3_expands_all_objects(self) -> None:
        spec = load_interface_seed_config(LHD101_CONFIG)

        instances = expand_symmetry_instances(spec)

        self.assertEqual(len(instances.motion_groups), 3)
        self.assertEqual(len(instances.fragments), 6)
        self.assertEqual(len(instances.ports), 6)
        self.assertEqual(len(instances.interfaces), 3)

    def test_instance_ids_and_transform_ids_are_stable(self) -> None:
        instances = expand_symmetry_instances(
            load_interface_seed_config(LHD101_CONFIG)
        )

        expected_group_ids = {
            "primary_seed@primary_orbit[0]",
            "primary_seed@primary_orbit[1]",
            "primary_seed@primary_orbit[2]",
        }
        self.assertEqual(set(instances.motion_groups), expected_group_ids)
        self.assertEqual(
            instances.motion_groups[
                "primary_seed@primary_orbit[2]"
            ].transform_id,
            "C3:r2",
        )

    def test_fragment_ownership_is_preserved_per_copy(self) -> None:
        instances = expand_symmetry_instances(
            load_interface_seed_config(LHD101_CONFIG)
        )

        group = instances.motion_groups[
            "primary_seed@primary_orbit[1]"
        ]
        self.assertEqual(
            group.fragment_instance_ids,
            (
                "left@primary_orbit[1]",
                "right@primary_orbit[1]",
            ),
        )
        for fragment_id in group.fragment_instance_ids:
            self.assertEqual(
                instances.fragments[fragment_id].motion_group_instance_id,
                group.id,
            )

    def test_port_references_fragments_from_same_copy(self) -> None:
        instances = expand_symmetry_instances(
            load_interface_seed_config(LHD101_CONFIG)
        )

        port = instances.ports["left_port@primary_orbit[2]"]
        self.assertEqual(
            port.motion_group_instance_id,
            "primary_seed@primary_orbit[2]",
        )
        self.assertEqual(
            port.fragment_instance_ids,
            ("left@primary_orbit[2]",),
        )

    def test_copy_transform_matches_fragment_group_and_port(self) -> None:
        instances = expand_symmetry_instances(
            load_interface_seed_config(LHD101_CONFIG)
        )

        group = instances.motion_groups[
            "primary_seed@primary_orbit[1]"
        ]
        fragment = instances.fragments["left@primary_orbit[1]"]
        port = instances.ports["left_port@primary_orbit[1]"]

        np.testing.assert_allclose(fragment.transform, group.transform)
        np.testing.assert_allclose(port.transform, group.transform)

    def test_unsymmetrized_group_gets_one_identity_copy(self) -> None:
        payload = load_interface_seed_config(LHD101_CONFIG).model_dump(
            mode="python"
        )
        payload["fragments"]["cargo"] = {
            "source": "inputs/cargo.pdb",
            "selection": "C/1-5/*",
            "entity_type": "protein",
            "role": "functional_component",
        }
        payload["motion_groups"]["cargo_group"] = {
            "members": ["cargo"],
            "mode": "fixed",
        }
        payload["ports"]["cargo_port"] = {
            "group": "cargo_group",
            "fragments": ["cargo"],
            "atoms": "heavy",
            "frame": {"method": "reference_interface_pca"},
        }
        spec = InterfaceSeedSpec.model_validate(payload)

        instances = expand_symmetry_instances(spec)

        group = instances.motion_groups["cargo_group@identity[0]"]
        self.assertEqual(group.transform_id, "E:e")
        np.testing.assert_allclose(group.transform, np.eye(4))
        self.assertIn("cargo@identity[0]", instances.fragments)
        self.assertIn("cargo_port@identity[0]", instances.ports)

    def test_expansion_is_deterministic(self) -> None:
        spec = load_interface_seed_config(LHD101_CONFIG)

        first = expand_symmetry_instances(spec)
        second = expand_symmetry_instances(spec)

        self.assertEqual(first, second)

    def test_interface_offset_zero_preserves_reference_pairing(self) -> None:
        instances = expand_symmetry_instances(
            load_interface_seed_config(LHD101_CONFIG)
        )
        edge = instances.interfaces["ring_interface@primary_orbit[2]"]

        self.assertEqual(
            edge.left_port_instance_id,
            "left_port@primary_orbit[2]",
        )
        self.assertEqual(
            edge.right_port_instance_id,
            "right_port@primary_orbit[2]",
        )

    def test_d3_expands_both_cyclic_cosets_without_offset_leakage(self) -> None:
        payload = load_interface_seed_config(LHD101_CONFIG).model_dump(
            mode="python"
        )
        payload["symmetry"]["transform_sets"]["ring_c3"] = {
            "type": "dihedral",
            "order": 3,
            "axis": (0.0, 0.0, 1.0),
            "secondary_axis": (1.0, 0.0, 0.0),
            "center": (0.0, 0.0, 0.0),
        }
        spec = InterfaceSeedSpec.model_validate(payload)

        instances = expand_symmetry_instances(spec)

        self.assertEqual(len(instances.motion_groups), 6)
        self.assertEqual(
            instances.motion_groups[
                "primary_seed@primary_orbit[5]"
            ].transform_id,
            "D3:s2",
        )
        reflected_link = instances.scaffold_links[
            "protomer@primary_orbit[5]"
        ]
        self.assertEqual(reflected_link.target_copy_index, 3)
        self.assertEqual(
            reflected_link.to_fragment_instance_id,
            "left@primary_orbit[3]",
        )

    def test_d3_named_relation_connects_twofold_partner_copies(self) -> None:
        payload = load_interface_seed_config(LHD101_CONFIG).model_dump(
            mode="python"
        )
        payload["symmetry"]["transform_sets"]["ring_c3"] = {
            "type": "dihedral",
            "order": 3,
            "axis": (0.0, 0.0, 1.0),
            "secondary_axis": (1.0, 0.0, 0.0),
            "center": (0.0, 0.0, 0.0),
        }
        payload["interfaces"]["ring_interface"]["copy_relation"] = {
            "transform": "D3:s0",
        }
        spec = InterfaceSeedSpec.model_validate(payload)

        instances = expand_symmetry_instances(spec)

        expected_target_indices = (3, 4, 5, 0, 1, 2)
        observed_target_indices = tuple(
            instances.interfaces[
                f"ring_interface@primary_orbit[{source_copy_index}]"
            ].target_copy_index
            for source_copy_index in range(6)
        )
        self.assertEqual(observed_target_indices, expected_target_indices)


if __name__ == "__main__":
    unittest.main()
