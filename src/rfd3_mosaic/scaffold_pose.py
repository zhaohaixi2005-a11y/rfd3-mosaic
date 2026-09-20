"""Rigidly align a complete template assembly to a declared symmetry registry.

The supplied entity/copy order is authoritative. This module neither permutes
chains nor deforms their backbones. A finite frame search is accepted only when
all registry operations agree with all corresponding template copies.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rmsd(first, second):
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=-1))))


def _proper_fit(source, target):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    first, second = source - source_center, target - target_center
    if np.linalg.matrix_rank(first, tol=1e-8) < 2:
        raise ValueError(
            "Template backbone does not define a non-collinear rigid frame"
        )
    u, _, vt = np.linalg.svd(first.T @ second)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _fixed_center(rotations, translations, prior):
    # The prior retains the unobservable axial coordinate of cyclic groups.
    a = (np.eye(3)[None] - rotations).reshape(-1, 3)
    b = translations.reshape(-1)
    return prior + np.linalg.lstsq(a, b - a @ prior, rcond=1e-8)[0]


def align_template_to_registry(
    template_chains: list[list[np.ndarray]],
    matrices: list[np.ndarray],
    *,
    maximum_symmetry_rmsd: float,
) -> tuple[list[list[np.ndarray]] | None, dict[str, Any]]:
    """Align ``[entity][registry_copy][residue, N/CA/C/O, xyz]`` rigidly.

    Matrices act on column coordinates and are ordered with identity first.
    Each entity must have the same residue/atom correspondence across copies.
    Template and registry centers are obtained from their affine actions; a
    cyclic group's unconstrained axial origin uses the template centroid and
    the registry's minimum-norm center. No residues or atoms are moved relative
    to one another. Failure to verify the supplied group action returns None.
    """
    if not np.isfinite(maximum_symmetry_rmsd) or maximum_symmetry_rmsd <= 0:
        raise ValueError("maximum_symmetry_rmsd must be positive and finite")
    registry = np.asarray(matrices, dtype=np.float64)
    if registry.ndim != 3 or registry.shape[1:] != (4, 4) or not len(registry):
        raise ValueError("matrices must be a nonempty [copies, 4, 4] registry")
    if not np.isfinite(registry).all():
        raise ValueError("Registry contains non-finite values")
    for matrix in registry:
        rotation = matrix[:3, :3]
        if (
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)
        ):
            raise ValueError("Registry matrices must be proper rigid transforms")
    if not np.allclose(registry[0], np.eye(4), atol=1e-7, rtol=0):
        raise ValueError("Registry copy zero must be identity")
    products = np.empty((len(registry), len(registry)), dtype=int)
    for i, operation in enumerate(registry):
        for j, other in enumerate(registry):
            errors = np.max(np.abs(registry - operation @ other), axis=(1, 2))
            matches = np.flatnonzero(errors < 1e-6)
            if len(matches) != 1:
                raise ValueError("Registry must be closed with unique group elements")
            products[i, j] = matches[0]

    if not template_chains:
        raise ValueError("At least one template entity is required")
    entities = []
    for entity in template_chains:
        if len(entity) != len(registry):
            raise ValueError(
                "Every entity must provide exactly one chain per registry copy"
            )
        copies = [np.asarray(chain, dtype=np.float64) for chain in entity]
        if (
            copies[0].ndim != 3
            or copies[0].shape[1:] != (4, 3)
            or not len(copies[0])
            or any(
                chain.shape != copies[0].shape or not np.isfinite(chain).all()
                for chain in copies
            )
        ):
            raise ValueError(
                "Entity copies must have equal finite [residues, 4, 3] backbones"
            )
        entities.append(copies)
    source = np.concatenate([entity[0].reshape(-1, 3) for entity in entities])
    observed_rotations, observed_translations, fit_records = [], [], []
    copy_centers = []
    for copy in range(len(registry)):
        target = np.concatenate([entity[copy].reshape(-1, 3) for entity in entities])
        rotation, translation = _proper_fit(source, target)
        observed_rotations.append(rotation)
        observed_translations.append(translation)
        copy_centers.append(target.mean(axis=0))
        for entity_index, entity in enumerate(entities):
            fit_records.append(
                {
                    "entity": entity_index,
                    "copy": copy,
                    "rmsd_angstrom": _rmsd(
                        entity[0] @ rotation.T + translation, entity[copy]
                    ),
                }
            )
    observed_rotations = np.asarray(observed_rotations)
    declared_rotations = registry[:, :3, :3]
    report: dict[str, Any] = {
        "status": "unresolved",
        "maximum_symmetry_rmsd_angstrom": maximum_symmetry_rmsd,
        "entity_count": len(entities),
        "copy_count": len(registry),
        "chain_correspondence": "supplied_entity_and_registry_order_no_permutations",
        "rigid_fit_records": fit_records,
        "rotation_start_budget": 8,
        "rotation_max_nfev_per_start": 200,
        "limitations": [
            "Symmetry-frame witness only, not seed compatibility or folding validation",
            "Failure within the finite search does not prove no frame exists",
        ],
    }
    if max(record["rmsd_angstrom"] for record in fit_records) > maximum_symmetry_rmsd:
        report["reason"] = "template_copies_are_not_common_rigid_symmetry_mates"
        return None, report

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    def residual(vector):
        q = Rotation.from_rotvec(vector).as_matrix()
        return (q[None] @ observed_rotations @ q.T[None] - declared_rotations).reshape(
            -1
        )

    starts = [np.zeros(3)]
    starts.extend(sign * np.pi / 2 * axis for axis in np.eye(3) for sign in (-1, 1))
    starts.append(np.full(3, np.pi / np.sqrt(3)))
    fits = [
        least_squares(residual, start, max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        for start in starts
    ]
    best = min(fits, key=lambda fit: float(np.sum(residual(fit.x) ** 2)))
    q = Rotation.from_rotvec(best.x).as_matrix()
    # Use exact conjugated registry rotations to avoid spurious axial rank
    # caused by rounded/noisy independently fitted rotations of a cyclic group.
    model_rotations = q.T[None] @ declared_rotations @ q[None]
    centers = np.asarray(copy_centers)
    translations = centers - np.einsum(
        "gij,j->gi", model_rotations, source.mean(axis=0)
    )
    observed_center = _fixed_center(model_rotations, translations, centers.mean(axis=0))
    declared_center = _fixed_center(declared_rotations, registry[:, :3, 3], np.zeros(3))
    translation = declared_center - q @ observed_center
    aligned = [[chain @ q.T + translation for chain in entity] for entity in entities]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = q, translation
    action_records = []
    for entity_index, entity in enumerate(aligned):
        for operation_index, operation in enumerate(registry):
            for source_copy, chain in enumerate(entity):
                target_copy = int(products[operation_index, source_copy])
                transformed = chain @ operation[:3, :3].T + operation[:3, 3]
                action_records.append(
                    {
                        "entity": entity_index,
                        "operation": operation_index,
                        "source_copy": source_copy,
                        "target_copy": target_copy,
                        "rmsd_angstrom": _rmsd(transformed, entity[target_copy]),
                    }
                )
    maximum = max(record["rmsd_angstrom"] for record in action_records)
    report.update(
        template_to_registry_transform=transform.tolist(),
        observed_center=observed_center.tolist(),
        declared_center=declared_center.tolist(),
        rotation_conjugacy_residual=float(np.linalg.norm(residual(best.x))),
        rotation_fit_nfev=[int(fit.nfev) for fit in fits],
        group_action_records=action_records,
        maximum_group_action_rmsd_angstrom=maximum,
    )
    if maximum > maximum_symmetry_rmsd:
        report["reason"] = (
            "supplied_chain_mapping_does_not_satisfy_declared_group_action"
        )
        return None, report
    report["status"] = "verified_symmetry_frame"
    return aligned, report
