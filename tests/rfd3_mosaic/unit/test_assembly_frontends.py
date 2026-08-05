import json
from pathlib import Path
import tempfile
import unittest

from rfd3_mosaic.assembly_frontends import (
    lower_central_motif_topology,
    lower_experiment_topology,
)
from rfd3_mosaic.compile import load_assembly_config


def _c3_registry() -> tuple[list[str], dict[str, list[list[float]]]]:
    order = ["C3:e", "C3:r1", "C3:r2"]
    matrices = {
        "C3:e": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "C3:r1": [
            [-0.5, -0.8660254037844386, 0.0, 0.0],
            [0.8660254037844386, -0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "C3:r2": [
            [-0.5, 0.8660254037844386, 0.0, 0.0],
            [-0.8660254037844386, -0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    return order, matrices


class AssemblyFrontendTestCase(unittest.TestCase):
    def test_central_frontend_writes_a_native_assembly_specification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.cif"
            source.write_text("data_source\n", encoding="utf-8")
            order, matrices = _c3_registry()
            template = root / "rfd3_input.json"
            template.write_text(
                json.dumps(
                    {
                        "template": {
                            "input": source.name,
                            "symmetry": {"id": "C3"},
                            "extra": {
                                "registry_transform_order": order,
                                "registry_transform_matrices": matrices,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            request = lower_central_motif_topology(
                {
                    "template_input": str(template),
                    "fixed_selector": "B1-31",
                    "n_terminal_length": 20,
                    "c_terminal_length": 25,
                },
                root / "compiled",
                experiment_name="central-c3",
            )
            spec = load_assembly_config(request.specification_path)

        self.assertEqual(request.example_id, "central-c3")
        self.assertEqual(request.semantic_audit, "central_motif")
        self.assertEqual(
            request.audit_metadata["probe_fixed_selector"],
            "A1-31",
        )
        self.assertEqual(
            spec.fragments["central_motif"].selection,
            "B/1-31/*",
        )
        self.assertEqual(
            set(spec.generated_segments),
            {"n_terminal", "c_terminal"},
        )
        self.assertFalse(spec.interfaces)
        self.assertFalse(spec.scaffold_links)

    def test_interface_frontend_only_normalizes_the_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "assembly.yaml"
            config.write_text(
                """\
assembly:
  schema_version: 2
  mode: constraint_assembly
  fragments:
    motif:
      source: motif.pdb
      selection: A/1/*
      entity_type: protein
      role: functional_motif
  motion_groups:
    motif_group:
      members: [motif]
      mode: fixed
  symmetry:
    transform_sets:
      ring: {type: cyclic, order: 3}
    orbits:
      motif_orbit:
        transform_set: ring
        master_groups: [motif_group]
  generated_segments:
    extension:
      anchor: {fragment: motif, terminus: C}
      length: {minimum: 5, maximum: 5}
""",
                encoding="utf-8",
            )
            request = lower_experiment_topology(
                {
                    "kind": "interface_seed",
                    "config": config.name,
                    "example_id": "example",
                    "pose_seed": 7,
                },
                root / "output",
                project_directory=root,
                experiment_name="ignored",
            )

        self.assertEqual(request.specification_path, config.resolve())
        self.assertEqual(request.example_id, "example")
        self.assertEqual(request.pose_seed, 7)
        self.assertEqual(request.semantic_audit, "interface_seed")


if __name__ == "__main__":
    unittest.main()
