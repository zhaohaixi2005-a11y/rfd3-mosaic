"""Full group action, not a single superposition, defines template symmetry."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rfd3_mosaic.scaffold_pose import align_template_to_registry


def _registry(kind, center):
    n = int(kind[1:])
    rotations = [
        Rotation.from_rotvec([0, 0, 2 * np.pi * k / n]).as_matrix() for k in range(n)
    ]
    if kind[0] == "D":
        flip = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
        rotations.extend([rotation @ flip for rotation in list(rotations)])
    matrices = []
    for rotation in rotations:
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = center - rotation @ center
        matrices.append(matrix)
    return matrices


def _template(kind, *, multiple_entities=False):
    center = np.array([4.0, -8.0, 2.0])
    matrices = _registry(kind, center)
    rng = np.random.default_rng(17)
    templates = []
    for length in [5, 7] if multiple_entities else [5]:
        chain = rng.normal(size=(length, 4, 3)) + [13.0, -7.0, 3.0]
        templates.append(
            [chain @ matrix[:3, :3].T + matrix[:3, 3] for matrix in matrices]
        )
    rotation = Rotation.from_rotvec([0.7, -0.3, 0.4]).as_matrix()
    translation = np.array([10.0, -30.0, 12.0])
    observed = [
        [chain @ rotation.T + translation for chain in entity] for entity in templates
    ]
    return observed, matrices


@pytest.mark.parametrize("kind", ["C1", "C2", "C3", "D2", "D3"])
def test_complete_template_is_rigidly_aligned_to_displaced_registry(kind):
    source, matrices = _template(kind, multiple_entities=True)
    snapshot = [[chain.copy() for chain in entity] for entity in source]
    aligned, report = align_template_to_registry(
        source, matrices, maximum_symmetry_rmsd=1e-5
    )
    assert aligned is not None, report
    assert report["status"] == "verified_symmetry_frame"
    assert report["maximum_group_action_rmsd_angstrom"] < 1e-5
    transform = np.array(report["template_to_registry_transform"])
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-10)
    for entity_index, entity in enumerate(aligned):
        for copy, chain in enumerate(entity):
            np.testing.assert_allclose(
                chain,
                source[entity_index][copy] @ transform[:3, :3].T + transform[:3, 3],
                atol=1e-10,
            )
            np.testing.assert_array_equal(
                source[entity_index][copy], snapshot[entity_index][copy]
            )
            # All internal distances remain unchanged; this operation cannot
            # repair a seed by distorting its internal interface geometry.
            before = source[entity_index][copy].reshape(-1, 3)
            after = chain.reshape(-1, 3)
            np.testing.assert_allclose(
                np.linalg.norm(before[:, None] - before[None], axis=-1),
                np.linalg.norm(after[:, None] - after[None], axis=-1),
                atol=1e-10,
            )
    assert len(report["group_action_records"]) == 2 * len(matrices) ** 2
    center = np.array(report["declared_center"])
    for matrix in matrices:
        np.testing.assert_allclose(
            matrix[:3, :3] @ center + matrix[:3, 3], center, atol=1e-9
        )


def test_wrong_chain_mapping_is_rejected_even_when_each_copy_fits_rigidly():
    source, matrices = _template("C4")
    source[0][1], source[0][2] = source[0][2], source[0][1]
    aligned, report = align_template_to_registry(
        source, matrices, maximum_symmetry_rmsd=1e-5
    )
    assert aligned is None
    assert max(record["rmsd_angstrom"] for record in report["rigid_fit_records"]) < 1e-8
    assert (
        report["reason"]
        == "supplied_chain_mapping_does_not_satisfy_declared_group_action"
    )
    assert report["maximum_group_action_rmsd_angstrom"] > 1.0


def test_entity_specific_symmetry_mismatch_is_not_hidden_by_other_entities():
    source, matrices = _template("D3", multiple_entities=True)
    source[1][2] += [2.0, 0.0, 0.0]
    aligned, report = align_template_to_registry(
        source, matrices, maximum_symmetry_rmsd=0.1
    )
    assert aligned is None
    assert report["reason"] == "template_copies_are_not_common_rigid_symmetry_mates"


def test_incomplete_or_improper_registry_is_invalid():
    source, matrices = _template("C3")
    with pytest.raises(ValueError, match="closed"):
        align_template_to_registry(
            [[source[0][0], source[0][1]]], matrices[:2], maximum_symmetry_rmsd=0.1
        )
    matrices[1][:3, 0] *= -1
    with pytest.raises(ValueError, match="proper rigid"):
        align_template_to_registry(source, matrices, maximum_symmetry_rmsd=0.1)


def test_copy_shape_mismatch_does_not_infer_a_residue_alignment():
    source, matrices = _template("C3")
    source[0][1] = source[0][1][:-1]
    with pytest.raises(ValueError, match="equal finite"):
        align_template_to_registry(source, matrices, maximum_symmetry_rmsd=0.1)
