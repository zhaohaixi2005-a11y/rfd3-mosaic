from contextlib import redirect_stdout
from io import StringIO
import json
import unittest

from rfd3_mosaic.capabilities import (
    CapabilityMaturity,
    capability_by_id,
    capability_manifest,
    required_capabilities_for_design,
)
from rfd3_mosaic.cli import main
from rfd3_mosaic.schema import UserDesignSpec


class CapabilityLedgerTestCase(unittest.TestCase):
    def test_maturity_ladder_is_ordered(self) -> None:
        self.assertLess(
            CapabilityMaturity.CPU_VALIDATED,
            CapabilityMaturity.STABLE,
        )
        self.assertLess(
            CapabilityMaturity.STABLE,
            CapabilityMaturity.SCIENTIFICALLY_VALIDATED,
        )

    def test_dependencies_reference_known_capabilities(self) -> None:
        manifest = capability_manifest()
        identifiers = {item["id"] for item in manifest["capabilities"]}
        for item in manifest["capabilities"]:
            self.assertTrue(set(item["dependencies"]).issubset(identifiers))

    def test_cylindrical_projector_is_not_overclaimed(self) -> None:
        record = capability_by_id("cylindrical_projector")

        self.assertEqual(record.maturity, CapabilityMaturity.SCHEMA_ONLY)

    def test_functional_geometry_is_not_overclaimed(self) -> None:
        schema = capability_by_id("functional_geometry_schema")
        runtime = capability_by_id("cooperative_site_orbit")

        self.assertEqual(schema.maturity, CapabilityMaturity.SCHEMA_ONLY)
        self.assertEqual(runtime.maturity, CapabilityMaturity.PLANNED)

    def test_polyhedral_execution_is_not_overclaimed(self) -> None:
        record = capability_by_id("polyhedral_groups")

        self.assertEqual(record.maturity, CapabilityMaturity.CPU_VALIDATED)

    def test_capabilities_cli_emits_machine_readable_json(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main(["capabilities", "--format", "json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema_version"], 1)
        self.assertGreater(len(payload["capabilities"]), 5)

    def test_design_requirements_expose_schema_only_features(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "requirements",
                "input": "motif.pdb",
                "symmetry": "D3",
                "constraints": [
                    {
                        "kind": "cylindrical",
                        "selector": "A1-10",
                        "keep": ["radius"],
                    }
                ],
                "sampling": {
                    "initial_pose": {
                        "radius": {"minimum": 20.0, "maximum": 30.0}
                    }
                },
            }
        )

        requirements = required_capabilities_for_design(design)
        observed = {item.id: item.maturity for item in requirements}
        self.assertEqual(
            observed["cylindrical_projector"],
            CapabilityMaturity.SCHEMA_ONLY,
        )
        self.assertEqual(
            observed["static_pose_sampling"],
            CapabilityMaturity.ENGINEERING,
        )
        self.assertEqual(
            observed["dn_static"],
            CapabilityMaturity.GPU_CANARY,
        )

    def test_mobile_fixed_component_declares_runtime_capability(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "mobile-component",
                "input": "motif.pdb",
                "symmetry": "C3",
                "constraints": [
                    {
                        "kind": "fixed_xyz",
                        "selector": "A1-10",
                        "pose": {
                            "mode": "bounded_mobile",
                            "max_translation": 3.0,
                            "max_rotation_deg": 10.0,
                        },
                    }
                ],
            }
        )

        observed = {
            item.id for item in required_capabilities_for_design(design)
        }
        self.assertEqual(
            observed,
            {"public_fixed_xyz", "bounded_orbit_mobility"},
        )

    def test_component_initial_poses_require_static_pose_sampling(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "component-initialization",
                "input": "motif.pdb",
                "symmetry": "T",
                "sampling": {
                    "initial_poses": {
                        "site_alpha": {
                            "radius": {
                                "minimum": 50.0,
                                "maximum": 50.0,
                            }
                        }
                    }
                },
            }
        )

        observed = {
            item.id for item in required_capabilities_for_design(design)
        }
        self.assertEqual(
            observed,
            {"static_pose_sampling", "polyhedral_groups"},
        )

    def test_mobile_dihedral_design_declares_dynamic_dn_capability(
        self,
    ) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "mobile-d3-component",
                "input": "motif.pdb",
                "symmetry": "D3",
                "constraints": [
                    {
                        "kind": "fixed_xyz",
                        "selector": "A1-10",
                        "pose": {
                            "mode": "bounded_mobile",
                            "max_translation": 3.0,
                            "max_rotation_deg": 10.0,
                        },
                    }
                ],
            }
        )

        observed = {
            item.id for item in required_capabilities_for_design(design)
        }
        self.assertEqual(
            observed,
            {
                "public_fixed_xyz",
                "bounded_orbit_mobility",
                "dn_static",
                "dn_dynamic_multi_orbit",
            },
        )

    def test_assembly_graph_declares_public_graph_capability(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "three-component-graph",
                "input": "motif.pdb",
                "symmetry": "C3",
                "components": {
                    "alpha": {"selectors": ["A1-2"]},
                    "beta": {"selectors": ["A3-4"]},
                    "gamma": {"selectors": ["A5-6"]},
                },
                "interfaces": [
                    {
                        "id": "alpha_beta",
                        "between": ["alpha", "beta"],
                        "relation": {"mode": "preserve_input"},
                    }
                ],
                "connections": [
                    {
                        "id": "alpha_to_beta",
                        "from": "alpha.C",
                        "to": "beta.N",
                        "length": 20,
                    }
                ],
            }
        )

        observed = {
            item.id: item.maturity
            for item in required_capabilities_for_design(design)
        }
        self.assertEqual(
            observed,
            {
                "public_assembly_graph": CapabilityMaturity.GPU_CANARY,
                "public_fixed_xyz": CapabilityMaturity.ENGINEERING,
            },
        )

    def test_contact_edge_declares_graph_guidance_capability(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "designed-interface-c3",
                "input": "motif.pdb",
                "symmetry": "C3",
                "components": {
                    "seed": {"selectors": ["A1-2"]},
                },
                "ports": {
                    "face": {
                        "component": "seed",
                        "selectors": ["A1-2"],
                    }
                },
                "interfaces": [
                    {
                        "id": "designed_face",
                        "between": ["face", "face"],
                        "copy_relation": {"orbit_offset": 1},
                        "relation": {
                            "mode": "contact",
                            "minimum_heavy_atom_contacts": 4,
                        },
                    }
                ],
                "connections": [
                    {
                        "id": "extension",
                        "from": "seed.C",
                        "to": "seed.N",
                        "length": 20,
                    }
                ],
            }
        )

        observed = {
            item.id for item in required_capabilities_for_design(design)
        }
        self.assertEqual(
            observed,
            {
                "public_assembly_graph",
                "public_fixed_xyz",
                "graph_interface_guidance",
            },
        )

    def test_terminal_contig_infers_graph_guidance_capability(self) -> None:
        design = UserDesignSpec.model_validate(
            {
                "name": "central-motif-auto-interface",
                "input": "motif.pdb",
                "symmetry": "C3",
                "generation": [
                    {
                        "kind": "terminal",
                        "anchor": "A1-2",
                        "terminus": "n",
                        "length": 20,
                    }
                ],
                "constraints": [
                    {"kind": "fixed_xyz", "selector": "A1-2"}
                ],
            }
        )

        observed = {
            item.id for item in required_capabilities_for_design(design)
        }
        self.assertEqual(
            observed,
            {"public_fixed_xyz", "graph_interface_guidance"},
        )


if __name__ == "__main__":
    unittest.main()
