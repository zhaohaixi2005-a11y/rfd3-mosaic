import json
import tempfile
import unittest
from pathlib import Path

from rfd3_mosaic.assembly_frontends import (
    AuditRequirement,
    lower_central_motif_topology,
    lower_experiment_topology,
)
from rfd3_mosaic.compile import load_assembly_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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
    def test_fixed_motif_interface_creation_records_default_core_guidance(
        self,
    ) -> None:
        design = (
            REPOSITORY_ROOT
            / "experiments"
            / "lrz_public_c3_locked_packing_patch_capture_v100_50step.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary:
            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "fixed-motif-default-core",
                },
                Path(temporary) / "output",
                project_directory=REPOSITORY_ROOT,
                experiment_name="fixed-motif-default-core",
                pose_seed=1234,
            )

        self.assertIn(
            AuditRequirement.SCAFFOLD_CORE_GUIDANCE,
            request.audit_requirements,
        )
        self.assertEqual(
            request.audit_metadata["scaffold_core_guidance"]["intra_chain_weight"],
            1.0,
        )

    def test_user_frontend_freezes_pose_specific_cyclic_wiring(self) -> None:
        design = (
            REPOSITORY_ROOT
            / "experiments"
            / "lrz_mosaic_lhd101_c3_guided_50step_template.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forward = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "lhd101-forward",
                },
                root / "forward",
                project_directory=REPOSITORY_ROOT,
                experiment_name="lhd101-forward",
                pose_seed=10001,
            )
            reverse = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "lhd101-reverse",
                },
                root / "reverse",
                project_directory=REPOSITORY_ROOT,
                experiment_name="lhd101-reverse",
                pose_seed=10002,
            )
            forward_spec = load_assembly_config(forward.specification_path)
            reverse_spec = load_assembly_config(reverse.specification_path)

        self.assertEqual(
            forward_spec.generated_segments["generated_001"].copy_relation.orbit_offset,
            1,
        )
        self.assertEqual(
            reverse_spec.generated_segments["generated_001"].copy_relation.orbit_offset,
            -1,
        )
        self.assertIn(
            AuditRequirement.SCAFFOLD_CORE_GUIDANCE,
            forward.audit_requirements,
        )
        self.assertEqual(
            forward.audit_metadata["scaffold_core_guidance"]["intra_chain_weight"],
            1.0,
        )

    def test_scaffold_intra_inter_balance_is_independent_of_new_interface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "seed.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA B   1       8.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: supplied-interface-core
input: seed.pdb
symmetry: C3
generation:
  - kind: between
    from_selector: A1
    to_selector: B1
    length: 20
constraints:
  - {kind: fixed_xyz, selector: A1}
  - {kind: fixed_xyz, selector: B1}
guidance:
  intra_chain_weight: 1.0
  inter_chain_weight: 0.1
sampling:
  scaffold_packing: "off"
""",
                encoding="utf-8",
            )
            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "supplied-interface-core",
                },
                root / "output",
                project_directory=root,
                experiment_name="supplied-interface-core",
            )

        self.assertIn(
            AuditRequirement.SCAFFOLD_CORE_GUIDANCE,
            request.audit_requirements,
        )
        self.assertNotIn(
            AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
            request.audit_requirements,
        )
        self.assertIsNone(
            request.audit_metadata["automatic_symmetric_scaffold_packing"]
        )
        plan = request.audit_metadata["scaffold_core_guidance"]
        self.assertEqual(plan["intra_chain_weight"], 1.0)
        self.assertEqual(plan["inter_chain_weight"], 0.1)
        self.assertEqual(plan["inter_chain_excess_penalty"], 0.0)
        self.assertFalse(plan["quality_contract"]["required"])

    def test_generated_interface_plus_core_requires_both_runtime_audits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "seed.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA B   1       8.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: generated-interface-core
input: seed.pdb
symmetry: C3
generation:
  - kind: between
    from_selector: A1
    to_selector: B1
    length: 20
constraints:
  - {kind: fixed_xyz, selector: A1}
  - {kind: fixed_xyz, selector: B1}
guidance:
  intra_chain_weight: 1.0
  inter_chain_weight: 0.1
sampling:
  scaffold_packing: symmetric_generated
""",
                encoding="utf-8",
            )
            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "generated-interface-core",
                },
                root / "output",
                project_directory=root,
                experiment_name="generated-interface-core",
            )

        self.assertIn(
            AuditRequirement.SCAFFOLD_CORE_GUIDANCE,
            request.audit_requirements,
        )
        self.assertIn(
            AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
            request.audit_requirements,
        )

    def test_supplied_interface_can_opt_into_higher_order_packing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "seed.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       8.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA B   1       8.000   4.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: supplied-interface-higher-oligomer
input: seed.pdb
symmetry: C3
task: preserve_supplied_geometry
generation:
  - {kind: terminal, anchor: A1, terminus: c, length: 20}
  - {kind: terminal, anchor: B1, terminus: c, length: 20}
constraints:
  - {kind: fixed_xyz, selector: A1, coupling_group: seed}
  - {kind: fixed_xyz, selector: B1, coupling_group: seed}
sampling:
  scaffold_packing: symmetric_generated
""",
                encoding="utf-8",
            )
            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "supplied-interface-higher-oligomer",
                },
                root / "output",
                project_directory=root,
                experiment_name="supplied-interface-higher-oligomer",
            )

        self.assertIn(
            AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
            request.audit_requirements,
        )
        self.assertEqual(
            request.audit_metadata["automatic_symmetric_scaffold_packing"]["mode"],
            "symmetric_generated",
        )

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
        self.assertEqual(
            request.audit_requirements,
            (AuditRequirement.EXACT_CONSTRAINT_ORBIT,),
        )
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
        self.assertEqual(
            request.audit_requirements,
            (AuditRequirement.INTERFACE_GEOMETRY,),
        )

    def test_create_interface_task_lowers_to_runtime_guidance_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: public-c3
input: motif.pdb
symmetry: C3
task: create_symmetric_interface
fixed_arrangement: optimize_components
generation:
  - kind: terminal
    anchor: A1
    terminus: n
    length: 15
  - kind: terminal
    anchor: A1
    terminus: c
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
""",
                encoding="utf-8",
            )

            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "public-c3",
                },
                root / "output",
                project_directory=root,
                experiment_name="public-c3",
            )
            spec = load_assembly_config(request.specification_path)

            locked_design = root / "locked-design.yaml"
            locked_design.write_text(
                design.read_text(encoding="utf-8")
                .replace(
                    "name: public-c3\n",
                    "name: public-c3-locked\n",
                )
                .replace(
                    "fixed_arrangement: optimize_components\n",
                    "",
                ),
                encoding="utf-8",
            )
            locked_request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(locked_design),
                    "example_id": "public-c3-locked",
                },
                root / "locked-output",
                project_directory=root,
                experiment_name="public-c3-locked",
            )
            locked_spec = load_assembly_config(locked_request.specification_path)

        self.assertEqual(
            request.audit_requirements,
            (
                AuditRequirement.EXACT_CONSTRAINT_ORBIT,
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS,
                AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
                AuditRequirement.SCAFFOLD_CORE_GUIDANCE,
                AuditRequirement.BOUNDED_COMPONENT_MOBILITY,
            ),
        )
        self.assertEqual(set(spec.fragments), {"motif_001"})
        self.assertEqual(
            set(spec.generated_segments),
            {"generated_001", "generated_002"},
        )
        self.assertEqual(
            set(spec.interfaces),
            {"auto_generated_interface_001"},
        )
        inferred = spec.interfaces["auto_generated_interface_001"]
        self.assertTrue(inferred.required)
        self.assertEqual(inferred.satisfaction_stage, "output")
        self.assertEqual(
            inferred.target_geometry.mode,
            "geometric_constraints",
        )
        self.assertEqual(
            inferred.target_geometry.contacts.min_heavy_atom_contacts,
            0,
        )
        self.assertEqual(inferred.target_geometry.coverage.mode, "auto")
        self.assertIsNone(inferred.target_geometry.distance)
        initialization = spec.initialization["fixed_component_001"]
        self.assertGreater(initialization.placement.radius.mean, 0.0)
        self.assertEqual(initialization.placement.radius.range, 0.0)
        orbit = spec.symmetry.orbits["motif_orbit"]
        mobility = orbit.component_mobility["fixed_component_001"]
        self.assertEqual(mobility.mode.value, "orbit_rigid")
        self.assertEqual(
            mobility.effective_proposal.value,
            "scaffold_objectives",
        )
        self.assertEqual(
            request.audit_metadata["public_task"],
            "create_symmetric_interface",
        )
        self.assertEqual(
            request.audit_metadata["constraint_plan"]["operators"][0]["operator"],
            "fixed_xyz",
        )
        self.assertEqual(
            locked_request.audit_requirements,
            (
                AuditRequirement.EXACT_CONSTRAINT_ORBIT,
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS,
                AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
                AuditRequirement.SCAFFOLD_CORE_GUIDANCE,
            ),
        )
        self.assertGreater(
            locked_spec.initialization["fixed_component_001"].placement.radius.mean,
            0.0,
        )
        self.assertEqual(
            locked_spec.symmetry.orbits["motif_orbit"].component_mobility,
            {},
        )
        self.assertEqual(
            locked_request.audit_metadata["fixed_arrangement"],
            "locked",
        )

    def test_mobile_component_requires_runtime_mobility_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: mobile-c3
input: motif.pdb
symmetry: C3
generation:
  - kind: terminal
    anchor: A1
    terminus: c
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
    coupling_group: mobile_component
    pose:
      mode: bounded_mobile
      max_translation: 3.0
      max_rotation_deg: 10.0
""",
                encoding="utf-8",
            )

            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "mobile-c3",
                },
                root / "output",
                project_directory=root,
                experiment_name="mobile-c3",
            )

        self.assertEqual(
            request.audit_requirements,
            (
                AuditRequirement.EXACT_CONSTRAINT_ORBIT,
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS,
                AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
                AuditRequirement.BOUNDED_COMPONENT_MOBILITY,
            ),
        )

    def test_preserve_task_does_not_invent_interface_or_mobility_audits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA B   1       8.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: supplied-interface-c3
input: motif.pdb
symmetry: C3
task: preserve_supplied_geometry
generation:
  - kind: between
    from_selector: A1
    to_selector: B1
    length: 20
constraints:
  - kind: fixed_xyz
    selector: A1
    coupling_group: supplied_interface
  - kind: fixed_xyz
    selector: B1
    coupling_group: supplied_interface
""",
                encoding="utf-8",
            )

            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "supplied-interface-c3",
                },
                root / "output",
                project_directory=root,
                experiment_name="supplied-interface-c3",
            )
            spec = load_assembly_config(request.specification_path)

        self.assertEqual(
            request.audit_requirements,
            (AuditRequirement.EXACT_CONSTRAINT_ORBIT,),
        )
        self.assertFalse(spec.interfaces)
        self.assertEqual(set(spec.generated_segments), {"generated_001"})
        orbit = spec.symmetry.orbits["motif_orbit"]
        self.assertEqual(orbit.component_mobility, {})
        self.assertEqual(
            spec.motion_groups["fixed_component_001"].members,
            ["motif_001", "motif_002"],
        )
        self.assertEqual(
            request.audit_metadata["public_task"],
            "preserve_supplied_geometry",
        )

    def test_mobile_graph_component_requires_runtime_mobility_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA A   2       8.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: mobile-graph-c3
input: motif.pdb
symmetry: C3
components:
  alpha:
    selectors: [A1]
    pose:
      mode: bounded_mobile
      max_translation: 3.0
      max_rotation_deg: 10.0
  beta:
    selectors: [A2]
interfaces:
  - id: alpha_beta
    between: [alpha, beta]
    relation: {mode: preserve_input}
connections:
  - id: alpha_to_beta
    from: alpha.C
    to: beta.N
    length: 20
""",
                encoding="utf-8",
            )

            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "mobile-graph-c3",
                },
                root / "output",
                project_directory=root,
                experiment_name="mobile-graph-c3",
            )

        self.assertEqual(
            request.audit_requirements,
            (
                AuditRequirement.EXACT_CONSTRAINT_ORBIT,
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS,
                AuditRequirement.BOUNDED_COMPONENT_MOBILITY,
            ),
        )

    def test_contact_graph_requires_sampler_guidance_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA A   2       4.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: contact-graph-c3
input: motif.pdb
symmetry: C3
components:
  alpha: {selectors: [A1]}
  beta: {selectors: [A2]}
interfaces:
  - id: alpha_beta
    between: [alpha, beta]
    copy_relation: {orbit_offset: 1}
    relation:
      mode: contact
      minimum_heavy_atom_contacts: 1
connections:
  - id: alpha_to_beta
    from: alpha.C
    to: beta.N
    length: 20
""",
                encoding="utf-8",
            )

            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "contact-graph-c3",
                },
                root / "output",
                project_directory=root,
                experiment_name="contact-graph-c3",
            )

        self.assertEqual(
            request.audit_requirements,
            (
                AuditRequirement.EXACT_CONSTRAINT_ORBIT,
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS,
                AuditRequirement.GRAPH_INTERFACE_GUIDANCE,
            ),
        )

    def test_optional_contact_graph_does_not_require_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "motif.pdb"
            structure.write_text(
                "ATOM      1   CA ALA A   1       0.000   0.000   0.000"
                "  1.00 20.00           C\n"
                "ATOM      2   CA ALA A   2       4.000   0.000   0.000"
                "  1.00 20.00           C\nEND\n",
                encoding="utf-8",
            )
            design = root / "design.yaml"
            design.write_text(
                """\
schema_version: 1
name: optional-contact-graph-c3
input: motif.pdb
symmetry: C3
components:
  alpha: {selectors: [A1]}
  beta: {selectors: [A2]}
interfaces:
  - id: alpha_beta
    between: [alpha, beta]
    required: false
    copy_relation: {orbit_offset: 1}
    relation:
      mode: contact
      minimum_heavy_atom_contacts: 1
connections:
  - id: alpha_to_beta
    from: alpha.C
    to: beta.N
    length: 20
""",
                encoding="utf-8",
            )

            request = lower_experiment_topology(
                {
                    "kind": "user_design",
                    "config": str(design),
                    "example_id": "optional-contact-graph-c3",
                },
                root / "output",
                project_directory=root,
                experiment_name="optional-contact-graph-c3",
            )

        self.assertEqual(
            request.audit_requirements,
            (
                AuditRequirement.EXACT_CONSTRAINT_ORBIT,
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS,
            ),
        )


if __name__ == "__main__":
    unittest.main()
