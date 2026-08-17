import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from pydantic import ValidationError

from rfd3_mosaic.compile import expand_symmetry_instances
from rfd3_mosaic.cli import main
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.schema import UserDesignSpec


def _atom_line(serial: int, residue: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}   CA ALA A{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class PublicAssemblyGraphTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "motif.pdb"
        self.structure.write_text(
            "".join(
                _atom_line(index, index, float(index * 5))
                for index in range(1, 7)
            )
            + "END\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _payload(self) -> dict[str, object]:
        return {
            "name": "three-component-graph",
            "input": str(self.structure),
            "symmetry": "C3",
            "components": {
                "alpha": {"selectors": ["A1-2"]},
                "beta": {"selectors": ["A3-4"]},
                "gamma": {
                    "selectors": ["A5", "A6"],
                    "geometry": "joint_rigid",
                },
            },
            "interfaces": [
                {
                    "id": "alpha_beta",
                    "between": ["alpha", "beta"],
                    "relation": {"mode": "preserve_input"},
                },
                {
                    "id": "beta_gamma",
                    "between": ["beta", "gamma"],
                    "relation": {
                        "mode": "contact",
                        "distance": {"minimum": 5.0, "maximum": 20.0},
                        "minimum_heavy_atom_contacts": 1,
                        "cutoff": 8.0,
                    },
                },
            ],
            "connections": [
                {
                    "id": "alpha_to_beta",
                    "from": "alpha.C",
                    "to": "beta.N",
                    "length": {"minimum": 10, "maximum": 20},
                },
                {
                    "id": "beta_to_gamma",
                    "from_endpoint": {
                        "component": "beta",
                        "terminus": "c",
                    },
                    "to_endpoint": {
                        "component": "gamma",
                        "selector": "A5",
                        "terminus": "n",
                    },
                    "length": 15,
                },
                {
                    "id": "gamma_internal",
                    "from_endpoint": {
                        "component": "gamma",
                        "selector": "A5",
                        "terminus": "c",
                    },
                    "to_endpoint": {
                        "component": "gamma",
                        "selector": "A6",
                        "terminus": "n",
                    },
                    "length": 5,
                },
            ],
        }

    def _multi_face_payload(self) -> dict[str, object]:
        return {
            "name": "one-component-multi-face-graph",
            "input": str(self.structure),
            "symmetry": "C3",
            "components": {
                "protomer": {
                    "selectors": ["A1-2", "A3-4", "A5", "A6"],
                    "geometry": "joint_rigid",
                },
            },
            "ports": {
                "face_alpha": {
                    "component": "protomer",
                    "selectors": ["A1-2"],
                },
                "face_beta": {
                    "component": "protomer",
                    "selectors": ["A3-4"],
                },
                "face_gamma": {
                    "component": "protomer",
                    "selectors": ["A5"],
                },
            },
            "interfaces": [
                {
                    "id": "alpha_to_beta_neighbour",
                    "between": ["face_alpha", "face_beta"],
                    "copy_relation": {"transform": "C3:r1"},
                    "relation": {"mode": "preserve_input"},
                },
                {
                    "id": "gamma_homotypic_neighbour",
                    "between": ["face_gamma", "face_gamma"],
                    "copy_relation": {"transform": "C3:r2"},
                    "relation": {
                        "mode": "contact",
                        "distance": {"minimum": 1.0, "maximum": 100.0},
                    },
                },
            ],
            "connections": [
                {
                    "id": "part_1",
                    "from": {
                        "component": "protomer",
                        "selector": "A1-2",
                        "terminus": "c",
                    },
                    "to": {
                        "component": "protomer",
                        "selector": "A3-4",
                        "terminus": "n",
                    },
                    "length": 5,
                },
                {
                    "id": "part_2",
                    "from": {
                        "component": "protomer",
                        "selector": "A3-4",
                        "terminus": "c",
                    },
                    "to": {
                        "component": "protomer",
                        "selector": "A5",
                        "terminus": "n",
                    },
                    "length": 5,
                },
                {
                    "id": "part_3",
                    "from": {
                        "component": "protomer",
                        "selector": "A5",
                        "terminus": "c",
                    },
                    "to": {
                        "component": "protomer",
                        "selector": "A6",
                        "terminus": "n",
                    },
                    "length": 5,
                },
            ],
        }

    def test_lowers_multiple_named_faces_on_one_rigid_component(
        self,
    ) -> None:
        lowered = lower_user_design(
            UserDesignSpec.model_validate(self._multi_face_payload())
        )
        spec = lowered.specification

        self.assertEqual(len(spec.motion_groups), 1)
        self.assertEqual(len(spec.ports), 3)
        self.assertEqual(len(spec.interfaces), 2)
        self.assertEqual(
            spec.ports["port__face_alpha"].group,
            "fixed_component_001",
        )
        self.assertEqual(
            spec.interfaces["alpha_to_beta_neighbour"].copy_relation.transform,
            "C3:r1",
        )
        self.assertEqual(
            spec.interfaces["gamma_homotypic_neighbour"].left_port,
            "port__face_gamma",
        )
        self.assertEqual(
            spec.interfaces["gamma_homotypic_neighbour"].right_port,
            "port__face_gamma",
        )

        instances = expand_symmetry_instances(spec)
        self.assertEqual(len(instances.ports), 9)
        self.assertEqual(len(instances.interfaces), 6)

    def test_interface_count_is_data_driven_not_hard_coded(self) -> None:
        """The graph/compiler must not encode a two- or eight-edge limit."""

        payload = self._multi_face_payload()
        payload["interfaces"].extend(
            {
                "id": f"additional_seed_relation_{index:02d}",
                "between": [
                    "face_alpha" if index % 2 == 0 else "face_beta",
                    "face_gamma",
                ],
                "copy_relation": {
                    "transform": "C3:r1" if index % 2 == 0 else "C3:r2"
                },
                "relation": {"mode": "preserve_input"},
            }
            for index in range(6)
        )

        specification = lower_user_design(
            UserDesignSpec.model_validate(payload)
        ).specification
        instances = expand_symmetry_instances(specification)

        self.assertEqual(len(specification.interfaces), 8)
        self.assertEqual(len(instances.interfaces), 24)

    def test_interface_use_validates_physical_multiplicity(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"][0]["use"] = 3

        lowered = lower_user_design(UserDesignSpec.model_validate(payload))

        resolution = lowered.interface_usage[0]
        self.assertEqual(resolution.interface_id, "alpha_to_beta_neighbour")
        self.assertEqual(resolution.requested, "exactly 3")
        self.assertEqual(resolution.physical_instance_count, 3)
        self.assertTrue(resolution.satisfied)

    def test_hyperedge_use_is_counted_once_not_once_per_member_pair(
        self,
    ) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"] = [
            {
                "id": "three_way__member_01",
                "hyperedge_id": "three_way",
                "between": ["face_alpha", "face_beta"],
                "copy_relation": {"transform": "C3:r1"},
                "relation": {"mode": "preserve_input"},
                "use": {"exact": 3},
            },
            {
                "id": "three_way__member_02",
                "hyperedge_id": "three_way",
                "between": ["face_beta", "face_gamma"],
                "copy_relation": {"transform": "C3:r1"},
                "relation": {"mode": "preserve_input"},
                "use": {"exact": 3},
            },
        ]

        lowered = lower_user_design(UserDesignSpec.model_validate(payload))

        self.assertEqual(len(lowered.interface_usage), 1)
        resolution = lowered.interface_usage[0]
        self.assertEqual(resolution.interface_id, "three_way")
        self.assertEqual(resolution.physical_instance_count, 3)
        self.assertEqual(
            {
                edge.hyperedge_id
                for edge in lowered.specification.interfaces.values()
            },
            {"three_way"},
        )
        instances = expand_symmetry_instances(lowered.specification)
        self.assertEqual(
            {edge.hyperedge_id for edge in instances.interfaces.values()},
            {"three_way"},
        )

    def test_atomic_three_participant_interface_lowers_to_one_hyperedge(
        self,
    ) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"] = [
            {
                "id": "three_way",
                "between": ["face_alpha", "face_beta", "face_gamma"],
                "contact_pairs": [
                    ["face_alpha", "face_beta"],
                    ["face_beta", "face_gamma"],
                ],
                "copy_relation": {"transform": "C3:r1"},
                "relation": {"mode": "preserve_input"},
                "use": {"exact": 3},
            }
        ]

        design = UserDesignSpec.model_validate(payload)
        self.assertEqual(len(design.interfaces), 1)
        self.assertEqual(len(design.interfaces[0].between), 3)
        lowered = lower_user_design(design)

        self.assertEqual(len(lowered.interface_usage), 1)
        self.assertEqual(
            lowered.interface_usage[0].interface_id,
            "three_way",
        )
        self.assertEqual(
            lowered.interface_usage[0].physical_instance_count,
            3,
        )
        self.assertEqual(len(lowered.specification.interfaces), 2)
        self.assertEqual(
            {
                edge.hyperedge_id
                for edge in lowered.specification.interfaces.values()
            },
            {"three_way"},
        )
        runtime_port_endpoints = [
            port_id
            for edge in lowered.specification.interfaces.values()
            for port_id in (edge.left_port, edge.right_port)
        ]
        self.assertEqual(len(runtime_port_endpoints), 4)
        self.assertEqual(len(set(runtime_port_endpoints)), 4)
        instances = expand_symmetry_instances(lowered.specification)
        self.assertEqual(len(instances.interfaces), 6)
        self.assertEqual(
            {edge.hyperedge_id for edge in instances.interfaces.values()},
            {"three_way"},
        )

    def test_atomic_hyperedge_rejects_disconnected_contact_members(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"] = [
            {
                "id": "three_way",
                "between": ["face_alpha", "face_beta", "face_gamma"],
                "contact_pairs": [["face_alpha", "face_beta"]],
                "relation": {"mode": "preserve_input"},
            }
        ]

        with self.assertRaisesRegex(
            ValidationError,
            "connect every participant",
        ):
            UserDesignSpec.model_validate(payload)

    def test_atomic_hyperedge_cannot_be_generated_contact_target(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"] = [
            {
                "id": "three_way",
                "between": ["face_alpha", "face_beta", "face_gamma"],
                "contact_pairs": [
                    ["face_alpha", "face_beta"],
                    ["face_beta", "face_gamma"],
                ],
                "relation": {"mode": "contact"},
            }
        ]

        with self.assertRaisesRegex(
            ValidationError,
            "require preserve_input",
        ):
            UserDesignSpec.model_validate(payload)

    def test_hyperedge_members_require_one_usage_contract(self) -> None:
        payload = self._multi_face_payload()
        for interface in payload["interfaces"]:
            interface["hyperedge_id"] = "one_site"
        payload["interfaces"][0]["use"] = {"exact": 3}
        payload["interfaces"][1]["use"] = "auto"

        with self.assertRaisesRegex(ValueError, "inconsistent use"):
            lower_user_design(UserDesignSpec.model_validate(payload))

    def test_omitted_interface_use_defaults_to_auto(self) -> None:
        design = UserDesignSpec.model_validate(self._multi_face_payload())

        self.assertEqual(
            tuple(interface.use.description for interface in design.interfaces),
            ("auto", "auto"),
        )

    def test_interface_use_rejects_incompatible_symmetry(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"][0]["use"] = {"exact": 4}

        with self.assertRaisesRegex(
            ValueError,
            "requested exactly 4 physical instances.*produces 3",
        ):
            lower_user_design(UserDesignSpec.model_validate(payload))

    def test_interface_use_accepts_range_and_auto_forms(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"][0]["use"] = {
            "minimum": 2,
            "maximum": 4,
        }
        payload["interfaces"][1]["use"] = "auto"

        lowered = lower_user_design(UserDesignSpec.model_validate(payload))

        self.assertEqual(
            tuple(item.requested for item in lowered.interface_usage),
            ("2..4", "auto"),
        )

    def test_interface_use_rejects_inverted_range(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"][0]["use"] = {
            "minimum": 5,
            "maximum": 2,
        }

        with self.assertRaisesRegex(
            ValidationError,
            "minimum cannot exceed maximum",
        ):
            UserDesignSpec.model_validate(payload)

    def test_rejects_port_selector_outside_owning_component(self) -> None:
        payload = self._multi_face_payload()
        payload["ports"]["face_alpha"]["selectors"] = ["A7"]

        with self.assertRaisesRegex(
            ValidationError,
            "selectors do not belong to component",
        ):
            UserDesignSpec.model_validate(payload)

    def test_rejects_identity_self_interface(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"][1]["copy_relation"] = {"orbit_offset": 0}

        with self.assertRaisesRegex(
            ValidationError,
            "self-interface must target a non-identity symmetry copy",
        ):
            UserDesignSpec.model_validate(payload)

    def test_lowering_rejects_noncanonical_named_neighbour(self) -> None:
        payload = self._multi_face_payload()
        payload["interfaces"][0]["copy_relation"] = {
            "transform": "C3:r01"
        }
        design = UserDesignSpec.model_validate(payload)

        with self.assertRaisesRegex(
            ValueError,
            "invalid symmetry neighbour.*C3:r1",
        ):
            lower_user_design(design)

    def test_lowers_three_component_graph_into_common_assembly_ir(
        self,
    ) -> None:
        lowered = lower_user_design(
            UserDesignSpec.model_validate(self._payload())
        )
        spec = lowered.specification

        self.assertEqual(len(spec.fragments), 4)
        self.assertEqual(spec.constraint_group_strategy, "motion_groups")
        self.assertEqual(len(spec.motion_groups), 3)
        self.assertEqual(len(spec.ports), 4)
        self.assertEqual(len(spec.interfaces), 2)
        self.assertEqual(len(spec.generated_segments), 3)
        self.assertEqual(
            len(spec.motion_groups["fixed_component_003"].members),
            2,
        )
        self.assertEqual(
            spec.interfaces["alpha_beta"].target_geometry.mode,
            "reference_transform",
        )
        self.assertEqual(
            spec.interfaces[
                "alpha_beta"
            ].target_geometry.minimum_heavy_atom_contacts,
            1,
        )
        self.assertEqual(
            spec.interfaces["alpha_beta"].satisfaction_stage,
            "input",
        )
        self.assertEqual(
            spec.interfaces["beta_gamma"].target_geometry.mode,
            "geometric_constraints",
        )
        self.assertEqual(
            spec.interfaces[
                "beta_gamma"
            ].target_geometry.contacts.min_heavy_atom_contacts,
            1,
        )
        self.assertEqual(
            spec.interfaces["beta_gamma"].satisfaction_stage,
            "output",
        )

        instances = expand_symmetry_instances(spec)
        self.assertEqual(len(instances.fragments), 12)
        self.assertEqual(len(instances.interfaces), 6)
        self.assertEqual(len(instances.generated_segments), 9)

    def test_contact_intent_needs_no_user_tuned_contact_count(self) -> None:
        payload = self._payload()
        payload["interfaces"][1]["relation"] = {"mode": "contact"}

        spec = lower_user_design(
            UserDesignSpec.model_validate(payload)
        ).specification
        geometry = spec.interfaces["beta_gamma"].target_geometry

        self.assertEqual(geometry.contacts.min_heavy_atom_contacts, 0)
        self.assertEqual(geometry.contacts.cutoff, 4.5)
        self.assertEqual(geometry.coverage.mode, "auto")

    def test_graph_components_compile_as_fixed_constraint_components(
        self,
    ) -> None:
        lowered = lower_user_design(
            UserDesignSpec.model_validate(self._payload())
        )

        self.assertEqual(
            tuple(
                operator.coupling_group
                for operator in lowered.constraint_plan.operators
            ),
            ("alpha", "beta", "gamma"),
        )
        self.assertEqual(
            tuple(
                len(operator.atom_ids)
                for operator in lowered.bound_constraints.operators
            ),
            (2, 2, 2),
        )

    def test_rejects_unknown_interface_component(self) -> None:
        payload = self._payload()
        payload["interfaces"][0]["between"] = ["alpha", "missing"]

        with self.assertRaisesRegex(ValidationError, "unknown components"):
            UserDesignSpec.model_validate(payload)

    def test_multi_selector_connection_requires_explicit_selector(
        self,
    ) -> None:
        payload = self._payload()
        payload["connections"][1]["to_endpoint"].pop("selector")

        with self.assertRaisesRegex(
            ValidationError,
            "must select one of its multiple component selectors",
        ):
            UserDesignSpec.model_validate(payload)

    def test_rejects_malformed_compact_connection_endpoint(self) -> None:
        payload = self._payload()
        payload["connections"][0]["from"] = "alpha.side"

        with self.assertRaisesRegex(
            ValidationError,
            "component.N or component.C",
        ):
            UserDesignSpec.model_validate(payload)

    def test_lowering_rejects_unattached_component_fragment(self) -> None:
        payload = self._payload()
        payload["connections"] = payload["connections"][:-1]
        design = UserDesignSpec.model_validate(payload)

        with self.assertRaisesRegex(
            NotImplementedError,
            "unattached selectors: A6",
        ):
            lower_user_design(design)

    def test_rejects_mixing_graph_and_legacy_frontends(self) -> None:
        payload = self._payload()
        payload["constraints"] = [
            {"kind": "fixed_xyz", "selector": "A1-2"}
        ]

        with self.assertRaisesRegex(
            ValidationError,
            "cannot be mixed with legacy",
        ):
            UserDesignSpec.model_validate(payload)

    def test_multiple_component_selectors_require_joint_rigid(self) -> None:
        payload = self._payload()
        payload["components"]["gamma"].pop("geometry")

        with self.assertRaisesRegex(
            ValidationError,
            "geometry=joint_rigid",
        ):
            UserDesignSpec.model_validate(payload)

    def test_plan_cli_reports_graph_shape(self) -> None:
        config = self.root / "graph.yaml"
        import yaml

        config.write_text(
            yaml.safe_dump(self._payload(), sort_keys=False),
            encoding="utf-8",
        )
        output = StringIO()

        with redirect_stdout(output):
            main(["plan", str(config)])

        text = output.getvalue()
        self.assertEqual(
            UserDesignSpec.model_validate(self._payload()).user_mode,
            "expert",
        )
        self.assertIn("user mode:  expert", text)
        self.assertIn(
            "assembly graph: 3 component(s), 0 port(s), 2 interface(s), "
            "3 connection(s)",
            text,
        )
        self.assertIn("generation: 3 region(s)", text)
        self.assertIn("assembly lowering: ready", text)

    def test_plan_json_uses_public_connection_endpoint_names(self) -> None:
        config = self.root / "graph.yaml"
        import yaml

        config.write_text(
            yaml.safe_dump(self._payload(), sort_keys=False),
            encoding="utf-8",
        )
        output = StringIO()

        with redirect_stdout(output):
            main(["plan", str(config), "--format", "json"])

        payload = json.loads(output.getvalue())
        connection = payload["connections"][0]
        self.assertIn("from", connection)
        self.assertIn("to", connection)
        self.assertNotIn("from_endpoint", connection)
        self.assertNotIn("to_endpoint", connection)

    def test_plan_explains_multi_face_neighbour_relations(self) -> None:
        config = self.root / "multi_face_graph.yaml"
        import yaml

        config.write_text(
            yaml.safe_dump(self._multi_face_payload(), sort_keys=False),
            encoding="utf-8",
        )
        output = StringIO()

        with redirect_stdout(output):
            main(["plan", str(config)])

        text = output.getvalue()
        self.assertIn(
            "assembly graph: 1 component(s), 3 port(s), 2 interface(s), "
            "3 connection(s)",
            text,
        )
        self.assertIn(
            "face_alpha: component=protomer selectors=A1-2",
            text,
        )
        self.assertIn(
            "alpha_to_beta_neighbour: face_alpha -> "
            "face_beta@C3:r1 relation=preserve_input "
            "use=auto required=True",
            text,
        )
        self.assertIn(
            "alpha_to_beta_neighbour: requested=auto "
            "physical_instances=3",
            text,
        )


if __name__ == "__main__":
    unittest.main()
