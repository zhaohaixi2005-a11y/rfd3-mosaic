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
            CapabilityMaturity.CPU_VALIDATED,
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


if __name__ == "__main__":
    unittest.main()
