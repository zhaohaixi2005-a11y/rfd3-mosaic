"""Complete-scaffold handoff tests using the real independent CA contract.

The small backbone coordinates are a serialization/mapping fixture, not a
claim about peptide stereochemistry or a biologically folded scaffold.
"""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfd3_mosaic.scaffold_input import (
    apply_scaffold_input,
    compiled_scaffold_contract_sha256,
    compiled_scaffold_residue_map,
)


def _write_structure(path, residues):
    lines = []
    for chain, number, coordinates in residues:
        for name, xyz in coordinates.items():
            lines.append(
                f"ATOM  {len(lines) + 1:5d} {name:^4s} ALA {chain}{number:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {name[0]:>2s}\n"
            )
    path.write_text("".join(lines) + "END\n")


def scaffold_fixture(directory, symmetry_id="C2"):
    """Return a bound ordinary-ASU input and its full C2 partial-input artifact."""
    a = np.array(
        [
            [15, 0, 0],
            [15, 3.8, 0],
            [15, 7.6, 0],
            [15, 11.4, 0],
            [18.8, 11.4, 0],
            [22.6, 11.4, 0],
            [22.6, 7.6, 0],
            [22.6, 3.8, 0],
            [22.6, 0, 0],
        ]
    )
    a[:, 2] = 10.0  # A declared C2 template need not be centered along its axis.
    matrices = {"e": np.eye(4).tolist(), "r": np.diag([-1.0, -1.0, 1.0, 1.0]).tolist()}
    if symmetry_id == "D2":
        matrices.update(
            {
                "x": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
                "y": np.diag([-1.0, 1.0, -1.0, 1.0]).tolist(),
            }
        )
    elif symmetry_id != "C2":
        raise ValueError("Unsupported test fixture group")
    order = list(matrices)
    chain_ids = "ABCD"[: len(order)]
    fixed_numbers = {1, 2, 8, 9}
    coordinates = []
    for chain, transform in zip(chain_ids, matrices.values()):
        rotation = np.asarray(transform)[:3, :3]
        for number, ca in enumerate(a, start=1):
            atoms = {
                name: (ca + delta) @ rotation.T
                for name, delta in {
                    "N": [-1, 0, 0],
                    "CA": [0, 0, 0],
                    "C": [1, 0, 0],
                    "O": [1, 1, 0],
                }.items()
            }
            coordinates.append((chain, number, atoms))
    source = directory / "compiled_seed.pdb"
    structure = directory / "complete.pdb"
    _write_structure(
        source,
        [
            ("A" if number < 3 else "D", number if number < 3 else number - 7, atoms)
            for chain, number, atoms in coordinates
            if chain == "A" and number in fixed_numbers
        ],
    )
    _write_structure(structure, coordinates)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    old_groups = [
        {
            "group_id": f"seed{i}",
            "constraint_kind": "interface",
            "members": [
                {
                    "role": role,
                    "source_fragment_id": role,
                    "src_components": [f"{chain}{n}" for n in numbers],
                    "sym_transform_id": i,
                }
                for role, chain, numbers in (
                    ("left", "A", [1, 2]),
                    ("right", "D", [1, 2]),
                )
            ],
        }
        for i in range(len(order))
    ]
    orbits = [
        {
            "constraint_orbit_id": "seeds",
            "mobility_mode": "fixed",
            "group_ids": [f"seed{i}" for i in range(len(order))],
            "master_group_id": "seed0",
            "group_transform_ids": list(range(len(order))),
            "source_components": ["A1", "A2", "D1", "D2"],
        }
    ]
    payload = {
        "test": {
            "dialect": 2,
            "input": source.name,
            "contig": "A1-2,5-5,D1-2",
            "select_fixed_atoms": {"A1-2": "ALL", "D1-2": "ALL"},
            "symmetry": {"id": symmetry_id, "is_symmetric_motif": True},
            "extra": {
                "compiler": "rfd3_mosaic.scaffold_test",
                "adapter_structure_sha256": source_sha,
                "full_standalone_structure_sha256": source_sha,
                "symmetry_action_kind": "regular_full_group",
                "registry_transform_order": order,
                "registry_transform_matrices": matrices,
                "materialized_linker_length": 5,
                "motif_constraint_groups": old_groups,
                "motif_constraint_orbits": orbits,
                "assembly_interface_relations": [],
                "asu_scaffold_segments": [],
                "asu_terminal_extensions": [],
            },
        }
    }
    layout = [
        {"transform_index": i, "entity_id": 0, "is_asu": i == 0}
        for i in range(len(order))
    ]
    component_map = compiled_scaffold_residue_map(
        payload, chain_layout=layout, chain_ids=list(chain_ids)
    )
    groups = copy.deepcopy(old_groups)
    for index, group in enumerate(groups):
        for member in group["members"]:
            member["correspondence_components"] = [
                component_map[(component, 0)] for component in member["src_components"]
            ]
            member["src_components"] = [
                component_map[(component, index)]
                for component in member["src_components"]
            ]
    mapped_orbits = copy.deepcopy(orbits)
    mapped_orbits[0]["source_components"] = [
        component_map[(component, 0)] for component in orbits[0]["source_components"]
    ]
    contract = {
        "schema_version": 1,
        "residues": [
            {
                "chain_id": chain,
                "residue_number": number,
                "fixed": number in fixed_numbers,
                "reference_ca": atoms["CA"].tolist(),
            }
            for chain, number, atoms in coordinates
        ],
        "helix_blocks": [
            {"id": f"{chain}{side}", "residue_indices": [i + offset for i in indices]}
            for chain, offset in zip(chain_ids, range(0, len(chain_ids) * 9, 9))
            for side, indices in (("left", [1, 2, 3]), ("right", [5, 6, 7]))
        ],
        "support_edges": [
            {"left": chain + "left", "right": chain + "right"} for chain in chain_ids
        ],
        "limits": {
            "maximum_ca_deviation": 2.0,
            "contact_distance": 8.0,
            "minimum_helix_contact_fraction": 1.0,
            "minimum_interchain_segment_distance": 1.0,
            "minimum_ca_bond_distance": 3.3,
            "maximum_ca_bond_distance": 4.3,
            "fixed_ca_tolerance": 1e-4,
            "geometry_tolerance": 1e-3,
        },
    }
    artifact = {
        "schema_version": 1,
        "compiled_contract_sha256": compiled_scaffold_contract_sha256(payload),
        "structure_path": structure.name,
        "structure_sha256": hashlib.sha256(structure.read_bytes()).hexdigest(),
        "partial_t": 2.0,
        "native_input": {
            "contig": ",/0,".join(f"{c}1-9" for c in chain_ids),
            "select_fixed_atoms": {
                f"{c}{start}-{end}": "ALL"
                for c in chain_ids
                for start, end in ((1, 2), (8, 9))
            },
            "select_unfixed_sequence": ",".join(f"{c}3-7" for c in chain_ids),
            "symmetry": {
                "id": symmetry_id,
                "is_symmetric_motif": True,
                "use_declared_frames": True,
                "declared_transform_order": order,
                "declared_transform_matrices": matrices,
                "declared_preexpanded_chain_layout": layout,
            },
            "extra": {
                "motif_constraint_groups": groups,
                "motif_constraint_orbits": mapped_orbits,
                "assembly_interface_relations": [],
                "asu_scaffold_segments": [],
                "asu_terminal_extensions": [],
                "preexpanded_chain_layout": layout,
                "asu_chain_count": len(order),
                "asu_polymer_chain_count": len(order),
                "generated_coordinate_initialization": "complete_scaffold_partial_diffusion",
                "mosaic_scaffold_contract": contract,
            },
        },
    }
    return payload, artifact, coordinates


class ScaffoldInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.payload, self.artifact, self.coordinates = scaffold_fixture(self.directory)

    def apply(self):
        path = self.directory / "input_artifact.json"
        path.write_text(json.dumps(self.artifact))
        return apply_scaffold_input(
            self.payload, artifact_path=path, output_directory=self.directory
        )

    def rewrite_structure(self):
        path = self.directory / "complete.pdb"
        _write_structure(path, self.coordinates)
        self.artifact["structure_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    def test_real_contract_handoff_preserves_masks_frame_and_private_payload(self):
        before = copy.deepcopy(self.payload)
        result = self.apply()["test"]
        self.assertEqual(self.payload, before)
        self.assertEqual(result["partial_t"], 2.0)
        self.assertEqual(result["ori_token"], [0.0, 0.0, 0.0])
        self.assertEqual(
            result["select_fixed_atoms"],
            self.artifact["native_input"]["select_fixed_atoms"],
        )
        self.assertEqual(result["select_unfixed_sequence"], "A3-7,B3-7")
        installed = self.directory / result["input"]
        self.assertEqual(
            installed.read_bytes(), (self.directory / "complete.pdb").read_bytes()
        )
        self.assertTrue(
            result["extra"]["scaffold_input"]["input_contract_audit"]["passed"]
        )
        self.assertTrue(
            all(
                report["maximum_error_angstrom"] < 1e-8
                for report in result["extra"]["scaffold_input"]["seed_atom_checks"]
            )
        )
        self.assertEqual(self.apply()["test"], result)

    def test_binding_excludes_paths_but_includes_pose_and_atom_policy(self):
        expected = compiled_scaffold_contract_sha256(self.payload)
        self.payload["test"]["input"] = "/elsewhere/seed.pdb"
        self.assertEqual(compiled_scaffold_contract_sha256(self.payload), expected)
        self.payload["test"]["select_fixed_atoms"]["A1-2"] = "BKBN"
        self.assertNotEqual(compiled_scaffold_contract_sha256(self.payload), expected)
        with self.assertRaisesRegex(ValueError, "compiler contract SHA256"):
            self.apply()

    def test_handoff_accepts_operator_roundoff_but_rejects_real_operator_change(self):
        self.payload = copy.deepcopy(self.payload)
        matrices = self.payload["test"]["extra"]["registry_transform_matrices"]
        transform = next(iter(matrices))
        matrices[transform][0][0] = float(np.nextafter(matrices[transform][0][0], np.inf))
        self.apply()
        matrices[transform][0][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "compiler contract SHA256"):
            self.apply()

    def test_legacy_exact_binding_remains_valid_on_authoring_platform(self):
        self.artifact["compiled_contract_sha256"] = compiled_scaffold_contract_sha256(
            self.payload, legacy_exact_matrices=True
        )
        self.apply()

    def test_complete_d2_orbit_is_validated_without_cyclic_special_case(self):
        self.payload, self.artifact, self.coordinates = scaffold_fixture(
            self.directory, "D2"
        )
        result = self.apply()["test"]
        self.assertEqual(result["symmetry"]["id"], "D2")
        self.assertEqual(
            len(result["symmetry"]["declared_preexpanded_chain_layout"]), 4
        )
        self.assertEqual(len(result["extra"]["scaffold_input"]["seed_atom_checks"]), 4)

    def test_orbit_canonical_component_remapping_cannot_be_forged(self):
        self.artifact["native_input"]["extra"]["motif_constraint_orbits"][0][
            "source_components"
        ][-1] = "A8"
        with self.assertRaisesRegex(ValueError, "canonical ASU remapping"):
            self.apply()

    def test_relation_policy_cannot_change_behind_allowed_extra_name(self):
        relation = {
            "edge_instance_id": "seed0",
            "required": True,
            "left_source_components": ["A1-2"],
            "right_source_components": ["D1-2"],
            "target_geometry": {"mode": "reference_transform", "contact_cutoff": 4.5},
        }
        self.payload["test"]["extra"]["assembly_interface_relations"] = [relation]
        remapped = copy.deepcopy(relation)
        remapped["right_source_components"] = ["A8-9"]
        remapped["target_geometry"]["contact_cutoff"] = 100.0
        self.artifact["native_input"]["extra"]["assembly_interface_relations"] = [
            remapped
        ]
        self.artifact["compiled_contract_sha256"] = compiled_scaffold_contract_sha256(
            self.payload
        )
        with self.assertRaisesRegex(ValueError, "graph/geometry policy"):
            self.apply()

    def test_structure_hash_and_unknown_overrides_rejected(self):
        self.artifact["structure_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "structure SHA256"):
            self.apply()
        self.rewrite_structure()
        self.artifact["native_input"]["extra"][
            "generated_cross_chain_topology_guidance"
        ] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "unsupported extra"):
            self.apply()

    def test_fixed_backbone_error_is_detected_even_when_ca_and_symmetry_unchanged(self):
        self.coordinates[0][2]["O"] += [0, 0, 0.1]
        self.coordinates[9][2]["O"] += [0, 0, 0.1]
        self.rewrite_structure()
        with self.assertRaisesRegex(ValueError, "preserve compiled seed pose"):
            self.apply()

    def test_generated_copy_must_match_declared_symmetry(self):
        self.coordinates[2][2]["O"] += [0, 0, 0.1]
        self.rewrite_structure()
        with self.assertRaisesRegex(
            ValueError, "coordinates violate the declared symmetry"
        ):
            self.apply()

    def test_generated_sequence_and_fixed_sequence_policies_are_explicit(self):
        self.artifact["native_input"]["select_unfixed_sequence"] = "A3-6,B3-7"
        with self.assertRaisesRegex(ValueError, "explicitly have unfixed sequence"):
            self.apply()
        self.artifact["native_input"]["select_unfixed_sequence"] = "A1-7,B1-7"
        with self.assertRaisesRegex(ValueError, "seed sequence conditioning policy"):
            self.apply()

    def test_auxiliary_selectors_cannot_silently_change_residue_identity(self):
        for name in (
            "select_buried",
            "select_partially_buried",
            "select_exposed",
            "select_hbond_acceptor",
            "select_hbond_donor",
            "select_hotspots",
        ):
            with self.subTest(selector=name):
                self.payload["test"][name] = "D1"
                with self.assertRaisesRegex(
                    ValueError, "auxiliary conditioning selectors"
                ):
                    self.apply()
                del self.payload["test"][name]

    def test_changed_compiled_length_rejected_even_with_matching_binding(self):
        self.payload["test"]["contig"] = "A1-2,6-6,D1-2"
        self.artifact["compiled_contract_sha256"] = compiled_scaffold_contract_sha256(
            self.payload
        )
        with self.assertRaisesRegex(ValueError, "per-chain generated length"):
            self.apply()

    def test_wrong_member_copy_and_unsupported_mobility_rejected(self):
        self.artifact["native_input"]["extra"]["motif_constraint_groups"][1]["members"][
            0
        ]["sym_transform_id"] = 0
        with self.assertRaisesRegex(ValueError, "motif member transform"):
            self.apply()
        self.artifact["native_input"]["extra"]["motif_constraint_groups"][1]["members"][
            0
        ]["sym_transform_id"] = 1
        self.payload["test"]["extra"]["motif_constraint_orbits"][0]["mobility_mode"] = (
            "rigid_body"
        )
        self.artifact["compiled_contract_sha256"] = compiled_scaffold_contract_sha256(
            self.payload
        )
        with self.assertRaisesRegex(ValueError, "fixed or bounded rigid"):
            self.apply()

    def test_second_distinct_artifact_cannot_replace_frozen_task(self):
        self.apply()
        installed = self.directory / "scaffold_input" / "artifact.json"
        before = installed.read_bytes()
        self.artifact["partial_t"] = 3.0
        with self.assertRaisesRegex(ValueError, "different frozen scaffold artifact"):
            self.apply()
        self.assertEqual(installed.read_bytes(), before)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "Native RFD3 CPU environment required",
)
class ScaffoldNativeHandoffTests(unittest.TestCase):
    setUp = ScaffoldInputTests.setUp
    apply = ScaffoldInputTests.apply

    def test_ordinary_partial_input_retains_historical_centering_behavior(self):
        from rfd3.inference.input_parsing import DesignInputSpecification

        native = self.apply()["test"]
        native["input"] = str(self.directory / native["input"])
        for key in (
            "generated_coordinate_initialization",
            "mosaic_scaffold_contract",
            "scaffold_input",
            "motif_constraint_groups",
            "motif_constraint_orbits",
        ):
            native["extra"].pop(key, None)
        atoms = DesignInputSpecification(**native).build()
        np.testing.assert_allclose(atoms.coord.mean(axis=0), np.zeros(3), atol=2e-6)
        self.assertTrue(np.all(atoms.src_component == ""))

    def test_native_parser_retains_complete_coordinates_and_group_identity(self):
        from atomworks.ml.transforms.atom_array import CopyAnnotation
        from rfd3.inference.input_parsing import DesignInputSpecification
        from rfd3.transforms.symmetry import AddSymmetryFeats
        from rfd3.transforms.util_transforms import AggregateFeaturesLikeAF3WithoutMSA
        import torch

        native = self.apply()["test"]
        native["input"] = str(self.directory / native["input"])
        atoms = DesignInputSpecification(**copy.deepcopy(native)).build()
        expected = {
            (chain, number, name): xyz
            for chain, number, residue in self.coordinates
            for name, xyz in residue.items()
        }
        observed = [
            (str(chain), int(number), str(name))
            for chain, number, name in zip(
                atoms.chain_id, atoms.res_id, atoms.atom_name, strict=True
            )
        ]
        self.assertEqual(set(observed), set(expected))  # no accidental second expansion
        self.assertEqual(len(observed), len(expected))
        np.testing.assert_allclose(
            atoms.coord, [expected[key] for key in observed], atol=2e-6
        )
        fixed = np.asarray([number in {1, 2, 8, 9} for _, number, _ in observed])
        np.testing.assert_array_equal(atoms.is_motif_atom_with_fixed_coord, fixed)
        np.testing.assert_array_equal(atoms.is_motif_atom_with_fixed_seq, fixed)
        data = {
            "atom_array": atoms,
            "specification": native,
            "is_inference": True,
            "feats": {},
        }
        data = AddSymmetryFeats().forward(data)
        membership = data["feats"]["motif_constraint_group_membership"]
        np.testing.assert_array_equal(membership.any(dim=0).numpy(), fixed)
        self.assertEqual(
            tuple(data["feats"]["motif_constraint_orbit_mobility_mode"].tolist()), (0,)
        )
        # These auxiliary features normally come from earlier token transforms;
        # use neutral values while exercising the actual noising handoff.
        data["feats"].update(
            {
                "ref_atom_name_chars": torch.zeros((len(atoms), 4), dtype=torch.long),
                "ref_element": torch.zeros(len(atoms), dtype=torch.long),
                "ref_pos": torch.zeros((len(atoms), 3)),
            }
        )
        atoms.set_annotation("is_protein", np.ones(len(atoms), dtype=bool))
        if "chain_iid" not in atoms.get_annotation_categories():
            atoms.set_annotation("chain_iid", atoms.chain_id.copy())
        if "occupancy" not in atoms.get_annotation_categories():
            atoms.set_annotation("occupancy", np.ones(len(atoms)))
        data = CopyAnnotation(
            annotation_to_copy="coord", new_annotation="coord_to_be_noised"
        ).forward(data)
        data = AggregateFeaturesLikeAF3WithoutMSA().forward(data)
        np.testing.assert_allclose(
            data["coord_atom_lvl_to_be_noised"].numpy(),
            [expected[key] for key in observed],
            atol=2e-6,
        )
        self.assertEqual(
            data["feats"]["mosaic_scaffold_contract"],
            self.artifact["native_input"]["extra"]["mosaic_scaffold_contract"],
        )
        # The public native prevalidator must distinguish unfixed generated
        # sequence from sequence redesign of the immutable seed.
        from rfd3_mosaic.rfd3_prevalidate import (
            prevalidate_rfd3_input,
            _validate_report,
        )

        path = self.directory / "native_input.json"
        path.write_text(json.dumps({"complete": native}))
        report = prevalidate_rfd3_input(path)
        self.assertEqual(report["status"], "passed")
        policy = report["partial_scaffold_conditioning_audit"]
        self.assertTrue(policy["passed"])
        self.assertFalse(policy["expected_variable_motif_sequence"])
        self.assertEqual(policy["expected_fixed_sequence_atom_count"], int(fixed.sum()))
        self.assertEqual(
            policy["expected_fixed_coordinate_atom_count"], int(fixed.sum())
        )
        policy["sequence_mask_mismatch_count"] = 1
        policy["passed"] = False
        self.assertTrue(
            any("masks do not match" in failure for failure in _validate_report(report))
        )


if __name__ == "__main__":
    unittest.main()


def test_compiler_binding_is_portable_only_for_symmetry_roundoff():
    matrices = {'C3:r1': [[-0.49999999999999983, -0.0], [0.0, 1.0]]}
    payload = {'example': {'symmetry': {'declared_transform_matrices': matrices},
                           'extra': {'registry_transform_matrices': matrices,
                                     'materialized_linker_length': 90}}}
    other = copy.deepcopy(payload)
    for mapping in (other['example']['symmetry']['declared_transform_matrices'],
                    other['example']['extra']['registry_transform_matrices']):
        mapping['C3:r1'][0] = [-0.4999999999999998, 0.0]
    expected = compiled_scaffold_contract_sha256(payload)
    assert compiled_scaffold_contract_sha256(other) == expected
    assert payload['example']['symmetry']['declared_transform_matrices']['C3:r1'][0][0] == -0.49999999999999983
    assert compiled_scaffold_contract_sha256(payload, legacy_exact_matrices=True) != compiled_scaffold_contract_sha256(other, legacy_exact_matrices=True)
    other['example']['extra']['registry_transform_matrices']['C3:r1'][0][0] += 1e-6
    assert compiled_scaffold_contract_sha256(other) != expected
    changed = copy.deepcopy(payload)
    changed['example']['extra']['materialized_linker_length'] = 91
    assert compiled_scaffold_contract_sha256(changed) != expected
