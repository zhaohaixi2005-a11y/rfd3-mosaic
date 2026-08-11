"""Construct and audit an RFD3 atom array without loading a checkpoint."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _to_numpy(value):
    """Convert torch/NumPy/list values without importing torch eagerly."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _runtime_homogeneous_transforms(
    symmetry_features: dict[str, Any],
) -> dict[int, Any]:
    """Convert RFD3's runtime ``(R, T)`` frames into 4x4 matrices."""

    import numpy as np

    matrices: dict[int, Any] = {}
    for raw_transform_id, (rotation, translation) in symmetry_features[
        "sym_transform"
    ].items():
        transform_id = int(raw_transform_id)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _to_numpy(rotation)
        matrix[:3, 3] = _to_numpy(translation)
        matrices[transform_id] = matrix
    return matrices


def _rotation_error_degrees(left, right) -> float:
    """Return the geodesic SO(3) distance after numeric polar cleanup."""

    import numpy as np

    def nearest_rotation(rotation):
        u, _, vh = np.linalg.svd(rotation)
        result = u @ vh
        if np.linalg.det(result) < 0.0:
            u[:, -1] *= -1.0
            result = u @ vh
        return result

    delta = nearest_rotation(left) @ nearest_rotation(right).T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _audit_runtime_transform_matrices(
    runtime_matrices: dict[int, Any],
    registry_matrices: dict[str, Any],
    declared_order: list[str],
    *,
    max_rotation_error_degrees: float = 0.01,
    max_translation_error_angstrom: float = 1e-3,
    max_orthogonality_error: float = 1e-5,
    max_determinant_error: float = 1e-5,
) -> dict[str, Any]:
    """Resolve runtime frames to Mosaic frames and reject ID-direction drift."""

    import numpy as np

    failures: list[str] = []
    runtime = {
        int(transform_id): _to_numpy(matrix)
        for transform_id, matrix in runtime_matrices.items()
    }
    registry = {
        str(transform_id): _to_numpy(matrix)
        for transform_id, matrix in registry_matrices.items()
    }
    if not runtime:
        failures.append("runtime contains no symmetry transforms")
    if len(runtime) != len(registry):
        failures.append(
            "runtime and registry transform counts differ: "
            f"{len(runtime)} != {len(registry)}"
        )
    if set(declared_order) != set(registry):
        failures.append(
            "declared registry transform order does not cover the matrix set"
        )

    candidates: dict[int, list[tuple[str, float, float]]] = {}
    proper_metrics: dict[int, tuple[float, float, float]] = {}
    valid_registry_ids: set[str] = set()
    homogeneous_row = np.asarray([0.0, 0.0, 0.0, 1.0])
    for registry_id, matrix in registry.items():
        if matrix.shape != (4, 4):
            failures.append(
                f"registry transform {registry_id!r} is not 4x4"
            )
            continue
        if not np.isfinite(matrix).all():
            failures.append(
                f"registry transform {registry_id!r} contains NaN or Inf"
            )
            continue
        if not np.allclose(
            matrix[3],
            homogeneous_row,
            atol=max_orthogonality_error,
        ):
            failures.append(
                f"registry transform {registry_id!r} has an invalid "
                "homogeneous last row"
            )
            continue
        rotation = matrix[:3, :3]
        orthogonality_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        )
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > max_orthogonality_error:
            failures.append(
                f"registry transform {registry_id!r} is not orthogonal "
                f"({orthogonality_error:.3g})"
            )
            continue
        if abs(determinant - 1.0) > max_determinant_error:
            failures.append(
                f"registry transform {registry_id!r} is not a proper "
                f"rotation (det={determinant:.8f})"
            )
            continue
        valid_registry_ids.add(registry_id)

    for runtime_id, matrix in runtime.items():
        if matrix.shape != (4, 4):
            failures.append(
                f"runtime transform {runtime_id} is not a 4x4 matrix"
            )
            continue
        if not np.isfinite(matrix).all():
            failures.append(
                f"runtime transform {runtime_id} contains NaN or Inf"
            )
            continue
        if not np.allclose(
            matrix[3],
            homogeneous_row,
            atol=max_orthogonality_error,
        ):
            failures.append(
                f"runtime transform {runtime_id} has an invalid "
                "homogeneous last row"
            )
            continue
        rotation = matrix[:3, :3]
        orthogonality_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        )
        determinant = float(np.linalg.det(rotation))
        determinant_error = abs(determinant - 1.0)
        proper_metrics[runtime_id] = (
            orthogonality_error,
            determinant,
            determinant_error,
        )
        if orthogonality_error > max_orthogonality_error:
            failures.append(
                f"runtime transform {runtime_id} is not orthogonal "
                f"({orthogonality_error:.3g})"
            )
        if determinant_error > max_determinant_error:
            failures.append(
                f"runtime transform {runtime_id} is not a proper rotation "
                f"(det={determinant:.8f})"
            )
        matches = []
        for registry_id, registry_matrix in registry.items():
            if registry_id not in valid_registry_ids:
                continue
            rotation_error = _rotation_error_degrees(
                rotation,
                registry_matrix[:3, :3],
            )
            translation_error = float(
                np.linalg.norm(
                    matrix[:3, 3] - registry_matrix[:3, 3]
                )
            )
            if (
                rotation_error <= max_rotation_error_degrees
                and translation_error
                <= max_translation_error_angstrom
            ):
                matches.append(
                    (
                        registry_id,
                        rotation_error,
                        translation_error,
                    )
                )
        candidates[runtime_id] = matches

    runtime_ids = sorted(runtime)
    solutions: list[dict[int, tuple[str, float, float]]] = []

    def search(index, used, selected):
        if len(solutions) > 1:
            return
        if index == len(runtime_ids):
            solutions.append(dict(selected))
            return
        runtime_id = runtime_ids[index]
        for candidate in candidates.get(runtime_id, ()):
            registry_id = candidate[0]
            if registry_id in used:
                continue
            used.add(registry_id)
            selected[runtime_id] = candidate
            search(index + 1, used, selected)
            selected.pop(runtime_id)
            used.remove(registry_id)

    search(0, set(), {})
    if not solutions:
        failures.append(
            "runtime symmetry matrices do not match the Mosaic registry"
        )
        resolved: dict[int, tuple[str, float, float]] = {}
    elif len(solutions) > 1:
        failures.append(
            "runtime-to-registry transform matching is ambiguous"
        )
        resolved = {}
    else:
        resolved = solutions[0]

    actual_mapping = {
        str(runtime_id): candidate[0]
        for runtime_id, candidate in resolved.items()
    }
    declared_mapping = {
        str(runtime_id): (
            declared_order[runtime_id]
            if 0 <= runtime_id < len(declared_order)
            else "<missing>"
        )
        for runtime_id in runtime_ids
    }
    for runtime_id, registry_id in actual_mapping.items():
        if declared_mapping.get(runtime_id) != registry_id:
            failures.append(
                f"runtime transform {runtime_id} is {registry_id}, not "
                f"declared {declared_mapping.get(runtime_id)}"
            )
    if runtime_ids != list(range(len(runtime_ids))):
        failures.append(
            "runtime transform IDs must be contiguous and begin at zero"
        )
    if "0" in actual_mapping and declared_order:
        if actual_mapping["0"] != declared_order[0]:
            failures.append(
                "runtime transform 0 does not map to registry identity"
            )
        declared_identity = registry.get(declared_order[0])
        if (
            declared_identity is None
            or not np.allclose(
                declared_identity,
                np.eye(4),
                atol=max_translation_error_angstrom,
            )
        ):
            failures.append(
                "the first declared registry transform is not identity"
            )

    per_transform = []
    for runtime_id in runtime_ids:
        orthogonality_error, determinant, determinant_error = (
            proper_metrics.get(
                runtime_id,
                (float("inf"), float("nan"), float("inf")),
            )
        )
        match = resolved.get(runtime_id)
        per_transform.append(
            {
                "runtime_transform_id": runtime_id,
                "registry_transform_id": (
                    match[0] if match is not None else None
                ),
                "rotation_error_degrees": (
                    match[1] if match is not None else None
                ),
                "translation_error_angstrom": (
                    match[2] if match is not None else None
                ),
                "orthogonality_error": orthogonality_error,
                "determinant": determinant,
                "determinant_error": determinant_error,
            }
        )

    return {
        "passed": not failures,
        "runtime_source": "AddSymmetryFeats.forward",
        "registry_source": "extra.registry_transform_matrices",
        "runtime_transform_count": len(runtime),
        "registry_transform_count": len(registry),
        "runtime_to_registry": actual_mapping,
        "declared_runtime_to_registry": declared_mapping,
        "per_transform": per_transform,
        "thresholds": {
            "max_rotation_error_degrees": max_rotation_error_degrees,
            "max_translation_error_angstrom": (
                max_translation_error_angstrom
            ),
            "max_orthogonality_error": max_orthogonality_error,
            "max_determinant_error": max_determinant_error,
        },
        "failures": failures,
    }


def _error_metrics(errors) -> dict[str, Any]:
    """Summarize one flat vector of coordinate errors."""

    import numpy as np

    values = _to_numpy(errors).reshape(-1)
    if not len(values):
        return {
            "atom_count": 0,
            "rmsd_angstrom": 0.0,
            "maximum_atom_error_angstrom": 0.0,
        }
    if not np.isfinite(values).all():
        return {
            "atom_count": int(len(values)),
            "rmsd_angstrom": float("inf"),
            "maximum_atom_error_angstrom": float("inf"),
        }
    return {
        "atom_count": int(len(values)),
        "rmsd_angstrom": float(np.sqrt(np.mean(np.square(values)))),
        "maximum_atom_error_angstrom": float(np.max(values)),
    }


def _audit_fixed_target_projection(
    atom_array,
    symmetry_features: dict[str, Any],
    constraint_groups: list[dict[str, Any]],
    *,
    max_rmsd_angstrom: float = 0.01,
    max_atom_error_angstrom: float = 0.03,
) -> dict[str, Any]:
    """Measure whether RFD3's true projector moves the fixed target."""

    import numpy as np
    import torch
    from rfd3.inference.symmetry.atom_array import FIXED_ENTITY_ID
    from rfd3.inference.symmetry.symmetry_utils import (
        build_symmetry_orbit_layout,
        project_symmetry_orbit_average,
        symmetry_orbit_mask_mismatch_count,
        symmetry_orbit_residual,
    )

    coordinates = torch.as_tensor(
        np.asarray(atom_array.coord),
        dtype=torch.float64,
    )[None, ...]
    fixed = np.asarray(
        atom_array.is_motif_atom_with_fixed_coord,
        dtype=bool,
    )
    entity_ids = np.asarray(
        atom_array.get_annotation("sym_entity_id")
    )
    projectable = fixed & (entity_ids != FIXED_ENTITY_ID)
    projector_features = {
        key: value
        for key, value in symmetry_features.items()
        if key.startswith("sym_") or key == "is_sym_asu"
    }
    layout = build_symmetry_orbit_layout(
        projector_features,
        like=coordinates,
    )
    projected = project_symmetry_orbit_average(
        coordinates,
        projector_features,
        partial_diffusion=True,
        layout=layout,
    )
    residual_rms, residual_maximum = symmetry_orbit_residual(
        coordinates,
        projector_features,
        atom_mask=torch.as_tensor(projectable),
        layout=layout,
    )
    errors = torch.linalg.vector_norm(
        projected[0] - coordinates[0],
        dim=-1,
    )
    overall = _error_metrics(errors[torch.as_tensor(projectable)])
    mismatch_count = symmetry_orbit_mask_mismatch_count(
        fixed,
        projector_features,
        layout=layout,
    )

    transform_ids = np.asarray(
        atom_array.get_annotation("sym_transform_id")
    )
    per_transform = []
    for transform_id in sorted(
        int(value)
        for value in np.unique(transform_ids[projectable])
    ):
        mask = projectable & (transform_ids == transform_id)
        per_transform.append(
            {
                "runtime_transform_id": transform_id,
                **_error_metrics(errors[torch.as_tensor(mask)]),
            }
        )

    membership = symmetry_features.get(
        "motif_constraint_group_membership"
    )
    per_group = []
    if membership is not None:
        membership = torch.as_tensor(membership, dtype=torch.bool)
        for group_index, group in enumerate(constraint_groups):
            group_mask = membership[group_index] & torch.as_tensor(
                projectable
            )
            per_group.append(
                {
                    "group_id": group.get(
                        "group_id",
                        f"group-{group_index}",
                    ),
                    **_error_metrics(errors[group_mask]),
                }
            )

    failures = []
    finite = (
        torch.isfinite(coordinates).all()
        and torch.isfinite(projected).all()
        and torch.isfinite(errors).all()
        and torch.isfinite(residual_rms).all()
        and torch.isfinite(residual_maximum).all()
    )
    if not bool(finite):
        failures.append(
            "fixed motif target projection contains NaN or Inf"
        )
    if overall["atom_count"] == 0:
        failures.append(
            "no projectable fixed motif atoms were found"
        )
    if mismatch_count:
        failures.append(
            "fixed motif mask is not closed over runtime symmetry orbits "
            f"({mismatch_count} mismatched atom slots)"
        )
    if overall["rmsd_angstrom"] > max_rmsd_angstrom:
        failures.append(
            "fixed motif projection RMSD exceeds threshold: "
            f"{overall['rmsd_angstrom']:.6f} > "
            f"{max_rmsd_angstrom:.6f} A"
        )
    if (
        overall["maximum_atom_error_angstrom"]
        > max_atom_error_angstrom
    ):
        failures.append(
            "fixed motif maximum projection error exceeds threshold: "
            f"{overall['maximum_atom_error_angstrom']:.6f} > "
            f"{max_atom_error_angstrom:.6f} A"
        )

    return {
        "passed": not failures,
        "method": (
            "project_symmetry_orbit_average(all runtime copies)"
        ),
        "fixed_atom_count": int(fixed.sum()),
        "projectable_fixed_atom_count": int(projectable.sum()),
        "projector_excluded_fixed_atom_count": int(
            (fixed & ~projectable).sum()
        ),
        **overall,
        "runtime_residual_rmsd_angstrom": (
            float(residual_rms.max().item())
            if torch.isfinite(residual_rms).all()
            else float("inf")
        ),
        "runtime_residual_maximum_atom_error_angstrom": (
            float(residual_maximum.max().item())
            if torch.isfinite(residual_maximum).all()
            else float("inf")
        ),
        "fixed_mask_orbit_mismatch_count": mismatch_count,
        "per_transform": per_transform,
        "per_constraint_group": per_group,
        "thresholds": {
            "max_rmsd_angstrom": max_rmsd_angstrom,
            "max_atom_error_angstrom": max_atom_error_angstrom,
        },
        "failures": failures,
    }


def _expected_multiplicity(symmetry_id: str) -> int:
    normalized = symmetry_id.strip().upper()
    if normalized.startswith("C") and normalized[1:].isdigit():
        order = int(normalized[1:])
        if order >= 2:
            return order
    if normalized.startswith("D") and normalized[1:].isdigit():
        order = int(normalized[1:])
        if order >= 2:
            return 2 * order
    polyhedral = {"T": 12, "O": 24, "I": 60}
    if normalized in polyhedral:
        return polyhedral[normalized]
    raise ValueError(
        f"Prevalidation requires a finite Cn/Dn/T/O/I symmetry, got "
        f"{symmetry_id!r}"
    )


def _validate_report(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected = report["expected_multiplicity"]
    expected_asu_chains = report.get("expected_asu_chain_count", 1)
    expected_chain_count = expected * expected_asu_chains
    if report["chain_count"] != expected_chain_count:
        failures.append(
            f"expected {expected_chain_count} chains, observed "
            f"{report['chain_count']}"
        )
    if report["symmetry_transform_ids"] != list(range(expected)):
        failures.append(
            "symmetry transform IDs do not cover every native copy"
        )
    residue_counts = list(report["residues_per_chain"].values())
    residue_count_frequencies = Counter(residue_counts)
    if not residue_counts or any(
        frequency % expected != 0
        for frequency in residue_count_frequencies.values()
    ):
        failures.append(
            "chain residue counts do not repeat across every symmetry copy"
        )
    if report["motif_atom_count"] <= 0:
        failures.append("RFD3 did not recognize any motif atoms")
    if report["fixed_coordinate_atom_count"] <= 0:
        failures.append("RFD3 did not recognize any fixed motif coordinates")
    if report["fixed_coordinate_atom_count"] != report["motif_atom_count"]:
        failures.append(
            "not every motif atom has fixed coordinates "
            f"({report['fixed_coordinate_atom_count']}/"
            f"{report['motif_atom_count']})"
        )
    if report["fixed_sequence_atom_count"] != report["motif_atom_count"]:
        failures.append(
            "not every motif atom has fixed sequence identity "
            f"({report['fixed_sequence_atom_count']}/"
            f"{report['motif_atom_count']})"
        )
    if report["asu_atom_count"] <= 0:
        failures.append("RFD3 did not mark an asymmetric unit")
    if report["symmetry_ids"] != [report["expected_symmetry_id"]]:
        failures.append("constructed symmetry annotation does not match input")
    declared_group_count = report.get(
        "declared_motif_constraint_group_count",
        0,
    )
    compiler = str(report.get("compiler") or "")
    requires_grouped_mosaic_input = compiler.startswith(
        "rfd3_mosaic."
    )
    if requires_grouped_mosaic_input and declared_group_count <= 0:
        failures.append(
            "Mosaic adapter input must declare motif constraint groups"
        )
    if declared_group_count:
        if (
            report.get("resolved_motif_constraint_group_count")
            != declared_group_count
        ):
            failures.append(
                "not every declared motif constraint group was resolved"
            )
        if report.get("motif_constraint_group_covered_atom_count") != report[
            "fixed_coordinate_atom_count"
        ]:
            failures.append(
                "motif constraint groups do not cover every fixed atom"
            )
    if declared_group_count or requires_grouped_mosaic_input:
        transform_audit = report.get(
            "symmetry_transform_matrix_audit"
        )
        if not transform_audit or not transform_audit.get("passed"):
            failures.append(
                "runtime symmetry matrices do not match the declared "
                "Mosaic registry"
            )
        target_audit = report.get("fixed_target_symmetry_audit")
        if not target_audit or not target_audit.get("passed"):
            failures.append(
                "fixed motif targets are incompatible with the runtime "
                "symmetry projector"
            )
    declared_orbit_count = report.get(
        "declared_motif_constraint_orbit_count",
        0,
    )
    if requires_grouped_mosaic_input and declared_orbit_count <= 0:
        failures.append(
            "Mosaic adapter input must declare motif constraint orbits"
        )
    if declared_orbit_count and report.get(
        "resolved_motif_constraint_orbit_count"
    ) != declared_orbit_count:
        failures.append(
            "not every declared motif constraint orbit was resolved"
        )
    return failures


def prevalidate_rfd3_input(
    input_path: str | Path,
    *,
    example_id: str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load one emitted specification and run RFD3's complete input builder."""

    # Imports stay lazy so schema/compiler users do not require an RFD3 runtime.
    import numpy as np
    import torch
    from rfd3.inference.input_parsing import (
        DesignInputSpecification,
        ensure_input_is_abspath,
    )
    from rfd3.transforms.conditioning_base import get_motif_features
    from rfd3.transforms.symmetry import AddSymmetryFeats

    path = Path(input_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("RFD3 input JSON must contain at least one example")
    selected_id = example_id or next(iter(payload))
    if selected_id not in payload:
        raise KeyError(f"Unknown RFD3 example ID {selected_id!r}")
    raw_spec = payload[selected_id]
    if not isinstance(raw_spec, dict):
        raise ValueError("Each RFD3 example must contain a JSON object")
    raw_spec = ensure_input_is_abspath(dict(raw_spec), path)
    design = DesignInputSpecification.safe_init(**raw_spec)
    atom_array, metadata = design.build(return_metadata=True)

    chain_ids = sorted(str(value) for value in np.unique(atom_array.chain_id))
    residues_per_chain: dict[str, int] = {}
    atoms_per_chain: dict[str, int] = {}
    for chain_id in chain_ids:
        mask = atom_array.chain_id == chain_id
        atoms_per_chain[chain_id] = int(mask.sum())
        residues_per_chain[chain_id] = len(
            {
                int(residue_id)
                for residue_id in atom_array.res_id[mask]
            }
        )

    categories = set(atom_array.get_annotation_categories())
    motif_mask = get_motif_features(atom_array)["is_motif_atom"].astype(bool)
    expected_symmetry_id = str(raw_spec["symmetry"]["id"]).upper()
    extra = raw_spec.get("extra") or {}
    compiler = str(extra.get("compiler") or "")
    constraint_groups = extra.get("motif_constraint_groups") or []
    constraint_orbits = extra.get("motif_constraint_orbits") or []
    runtime_symmetry_features = AddSymmetryFeats().forward(
        {
            "atom_array": atom_array,
            "feats": {},
            "specification": raw_spec,
        }
    )["feats"]
    nonfinite_runtime_features = sorted(
        name
        for name, value in runtime_symmetry_features.items()
        if isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    )
    if nonfinite_runtime_features:
        raise ValueError(
            "RFD3 input construction emitted non-finite runtime features: "
            + ", ".join(nonfinite_runtime_features)
        )
    group_membership = runtime_symmetry_features.get(
        "motif_constraint_group_membership"
    )
    if group_membership is not None:
        group_sizes = [
            int(value)
            for value in group_membership.sum(dim=1).tolist()
        ]
        group_covered_atom_count = int(
            group_membership.any(dim=0).sum().item()
        )
    else:
        group_membership = None
        group_sizes = []
        group_covered_atom_count = 0

    runtime_matrices = _runtime_homogeneous_transforms(
        runtime_symmetry_features
    )
    registry_matrices = extra.get("registry_transform_matrices") or {}
    declared_order = list(extra.get("registry_transform_order") or ())
    transform_matrix_audit = _audit_runtime_transform_matrices(
        runtime_matrices,
        registry_matrices,
        declared_order,
    )
    if compiler.startswith("rfd3_mosaic."):
        fixed_target_audit = _audit_fixed_target_projection(
            atom_array,
            runtime_symmetry_features,
            constraint_groups,
        )
    else:
        fixed_target_audit = {
            "passed": None,
            "skipped": True,
            "reason": (
                "exact fixed-target orbit audit is required only for "
                "rfd3_mosaic compiler inputs"
            ),
        }
    orbit_master_indices = runtime_symmetry_features.get(
        "motif_constraint_orbit_master_group_index"
    )
    resolved_constraint_orbit_count = (
        int(orbit_master_indices.shape[0])
        if orbit_master_indices is not None
        else 0
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "pending",
        "input_path": str(path),
        "example_id": selected_id,
        "expected_symmetry_id": expected_symmetry_id,
        "compiler": extra.get("compiler"),
        "expected_multiplicity": int(
            extra.get(
                "symmetry_multiplicity",
                _expected_multiplicity(expected_symmetry_id),
            )
        ),
        "full_symmetry_multiplicity": int(
            extra.get(
                "full_symmetry_multiplicity",
                _expected_multiplicity(expected_symmetry_id),
            )
        ),
        "symmetry_action_kind": extra.get(
            "symmetry_action_kind",
            "regular_full_group",
        ),
        "expected_asu_chain_count": int(extra.get("asu_chain_count", 1)),
        "atom_count": int(len(atom_array)),
        "chain_count": len(chain_ids),
        "chain_ids": chain_ids,
        "atoms_per_chain": atoms_per_chain,
        "residues_per_chain": residues_per_chain,
        "motif_atom_count": int(motif_mask.sum()),
        "fixed_coordinate_atom_count": int(
            atom_array.is_motif_atom_with_fixed_coord.astype(bool).sum()
        ),
        "fixed_sequence_atom_count": int(
            atom_array.is_motif_atom_with_fixed_seq.astype(bool).sum()
        ),
        "declared_motif_constraint_group_count": len(constraint_groups),
        "resolved_motif_constraint_group_count": (
            int(group_membership.shape[0])
            if group_membership is not None
            else 0
        ),
        "motif_constraint_group_sizes": group_sizes,
        "motif_constraint_group_covered_atom_count": (
            group_covered_atom_count
        ),
        "declared_motif_constraint_orbit_count": len(
            constraint_orbits
        ),
        "resolved_motif_constraint_orbit_count": (
            resolved_constraint_orbit_count
        ),
        "asu_atom_count": int(atom_array.is_sym_asu.astype(bool).sum()),
        "symmetry_ids": sorted(
            str(value) for value in np.unique(atom_array.symmetry_id)
        ),
        "symmetry_transform_ids": sorted(
            int(value) for value in np.unique(atom_array.sym_transform_id)
        ),
        "annotation_count": len(categories),
        "symmetry_transform_matrix_audit": transform_matrix_audit,
        "fixed_target_symmetry_audit": fixed_target_audit,
        "rfd3_metadata": metadata,
    }
    failures = _validate_report(report)
    report["status"] = "passed" if not failures else "failed"
    report["failures"] = failures

    destination = (
        Path(report_path)
        if report_path is not None
        else path.with_name("rfd3_prevalidation.json")
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(destination.resolve())
    if failures:
        raise ValueError(
            "RFD3 input construction failed semantic checks: "
            + "; ".join(failures)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and audit an RFD3 input without loading a checkpoint."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--example-id")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    report = prevalidate_rfd3_input(
        arguments.input,
        example_id=arguments.example_id,
        report_path=arguments.report,
    )
    print("RFD3 input construction: PASSED")
    print(f"example:    {report['example_id']}")
    print(f"chains:     {report['chain_count']} {report['chain_ids']}")
    print(f"residues:   {report['residues_per_chain']}")
    print(f"motif atoms:{report['motif_atom_count']}")
    print(f"fixed atoms:{report['fixed_coordinate_atom_count']}")
    print(
        "constraint groups:"
        f"{report['resolved_motif_constraint_group_count']} "
        f"{report['motif_constraint_group_sizes']}"
    )
    print(f"transforms: {report['symmetry_transform_ids']}")
    matrix_audit = report["symmetry_transform_matrix_audit"]
    print(
        "runtime mapping:"
        f"{matrix_audit['runtime_to_registry']}"
    )
    target_audit = report["fixed_target_symmetry_audit"]
    if target_audit.get("skipped"):
        print(f"fixed target:skipped ({target_audit['reason']})")
    else:
        print(
            "fixed target:"
            f"rms={target_audit['rmsd_angstrom']:.6f} A, "
            "max="
            f"{target_audit['maximum_atom_error_angstrom']:.6f} A"
        )
    print(f"report:     {report['report_path']}")


if __name__ == "__main__":
    main()
