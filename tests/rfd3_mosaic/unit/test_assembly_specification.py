import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
import yaml

from rfd3_mosaic import (
    AssemblySpecification,
    InterfaceSeedSpec,
    load_assembly_config,
    load_interface_seed_config,
)
from rfd3_mosaic.compile import expand_symmetry_instances


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


def _central_motif_payload() -> dict:
    return {
        "schema_version": 2,
        "mode": "constraint_assembly",
        "fragments": {
            "motif": {
                "source": "motif.cif",
                "selection": "A/20-50/*",
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
                "anchor": {"fragment": "motif", "terminus": "N"},
                "length": {"minimum": 35, "maximum": 35},
            },
            "c_flank": {
                "anchor": {"fragment": "motif", "terminus": "C"},
                "length": {"minimum": 35, "maximum": 35},
            },
        },
    }


class AssemblySpecificationTestCase(unittest.TestCase):
    def test_interface_seed_name_is_a_compatibility_alias(self) -> None:
        self.assertIs(InterfaceSeedSpec, AssemblySpecification)

    def test_central_motif_uses_the_same_assembly_schema(self) -> None:
        spec = AssemblySpecification.model_validate(
            _central_motif_payload()
        )

        self.assertEqual(spec.mode, "constraint_assembly")
        self.assertEqual(set(spec.generated_segments), {"n_flank", "c_flank"})
        self.assertEqual(spec.ports, {})
        self.assertEqual(spec.interfaces, {})

    def test_central_motif_can_compile_bounded_se3_orbit_mobility(
        self,
    ) -> None:
        payload = _central_motif_payload()
        payload["objectives"] = {
            "junction": {
                "metric": "junction_distance",
                "mode": "minimize",
            }
        }
        payload["symmetry"]["orbits"]["motif_orbit"]["mobility"] = {
            "mode": "orbit_rigid",
            "bounds": {
                "max_translation": 4.0,
                "max_rotation_deg": 20.0,
            },
            "subspace": "bounded_se3",
            "proposal": "scaffold_objectives",
            "schedule": {
                "start_fraction": 0.1,
                "end_fraction": 0.7,
                "response": 0.25,
                "max_step_translation": 0.2,
                "max_step_rotation_deg": 1.5,
            },
            "objectives": ["junction"],
        }

        spec = AssemblySpecification.model_validate(payload)
        instances = expand_symmetry_instances(spec)
        orbit = instances.constraint_orbits["motif_orbit"]

        self.assertEqual(len(orbit.group_instance_ids), 3)
        self.assertEqual(len(instances.generated_segments), 6)
        self.assertIn(
            "n_flank@motif_orbit[0]",
            instances.generated_segments,
        )
        self.assertIn(
            "c_flank@motif_orbit[2]",
            instances.generated_segments,
        )
        self.assertEqual(orbit.transform_ids, ("C3:e", "C3:r1", "C3:r2"))
        self.assertEqual(orbit.mobility.effective_subspace.value, "bounded_se3")
        self.assertEqual(
            orbit.mobility.effective_proposal.value,
            "scaffold_objectives",
        )
        self.assertEqual(orbit.mobility.objectives, ("junction",))

    def test_orbit_mobility_rejects_unknown_objective(self) -> None:
        payload = _central_motif_payload()
        payload["symmetry"]["orbits"]["motif_orbit"]["mobility"] = {
            "mode": "orbit_rigid",
            "bounds": {
                "max_translation": 2.0,
                "max_rotation_deg": 10.0,
            },
            "objectives": ["missing"],
        }

        with self.assertRaisesRegex(ValidationError, "unknown objective"):
            AssemblySpecification.model_validate(payload)

    def test_generated_segments_cannot_reuse_one_endpoint(self) -> None:
        payload = _central_motif_payload()
        payload["scaffold_links"] = {
            "duplicate": {
                "from_endpoint": {
                    "fragment": "motif",
                    "terminus": "C",
                },
                "to_endpoint": {
                    "fragment": "motif",
                    "terminus": "N",
                },
                "length": {"minimum": 10, "maximum": 10},
            }
        }

        with self.assertRaisesRegex(ValidationError, "used more than once"):
            AssemblySpecification.model_validate(payload)

    def test_loader_accepts_new_assembly_wrapper_and_legacy_api(self) -> None:
        reference = yaml.safe_load(
            REFERENCE_CONFIG.read_text(encoding="utf-8")
        )["interface_seed"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assembly.yaml"
            path.write_text(
                yaml.safe_dump({"assembly": reference}, sort_keys=False),
                encoding="utf-8",
            )
            generic = load_assembly_config(path)
            legacy = load_interface_seed_config(path)

        self.assertEqual(generic, legacy)
        self.assertIsInstance(generic, AssemblySpecification)

    def test_loader_rejects_ambiguous_wrappers(self) -> None:
        payload = _central_motif_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "assembly": payload,
                        "interface_seed": payload,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot define both"):
                load_assembly_config(path)


if __name__ == "__main__":
    unittest.main()
