"""CPU construction of a peptide backbone between immutable anchor residues.

This is a finite inverse-kinematics search, not a folding predictor. Coordinates
are ordered ``[residue, (N, CA, C, O), xyz]``. A successful result is a witness
for the explicitly checked backbone geometry only; other seed atoms, assembly
copies, sidechains and the requested fold must be audited by the caller.

The generated chain has fixed standard bond lengths/angles and trans peptide
bonds. Its phi/psi torsions are the search variables. The left carbonyl O fixes
the outgoing peptide plane, so its psi is NOT an independent search variable.
The right anchor's observed N-CA-C geometry is used for the closure frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Any

import numpy as np


_CN, _NCA, _CAC, _CO = 1.329, 1.458, 1.525, 1.229
_CACN, _CNCA, _NCAC, _CACO = np.deg2rad((116.2, 121.7, 111.2, 120.8))
_ATOM_NAMES = ("N", "CA", "C", "O")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError("Degenerate backbone frame")
    return vector / norm


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(a - b), _unit(c - b)), -1, 1)))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    axis = _unit(c - b)
    first, second = _unit(np.cross(b - a, axis)), _unit(np.cross(axis, d - c))
    return float(
        np.arctan2(np.dot(np.cross(first, second), axis), np.dot(first, second))
    )


def _place_atom(a, b, c, length: float, angle: float, torsion: float):
    axis = _unit(c - b)
    normal = _unit(np.cross(b - a, axis))
    in_plane = np.cross(normal, axis)
    return c + length * (
        -np.cos(angle) * axis
        + np.sin(angle) * (np.cos(torsion) * in_plane + np.sin(torsion) * normal)
    )


def _checked_backbone(value, *, name: str, residues: int | None = None):
    array = np.asarray(value, dtype=np.float64)
    expected = (residues, 4, 3) if residues is not None else None
    if (
        array.ndim != 3
        or array.shape[1:] != (4, 3)
        or (expected and array.shape != expected)
    ):
        raise ValueError(
            f"{name} must have shape [residues, 4, 3]"
            + (f" = {expected}" if expected else "")
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return array.copy()


def _secondary(value: str | None, length: int, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != length or set(value) - {"H", "L"}:
        raise ValueError(f"{name} must contain exactly {length} H/L assignments")
    return value


@dataclass(frozen=True)
class _ChainModel:
    left: np.ndarray
    right: np.ndarray
    length: int
    psi_left: float
    right_nca: float
    right_cac: float
    right_angle: float

    @classmethod
    def from_anchors(cls, left, right, length):
        # O is opposite N_next around CA-C in the carbonyl peptide plane.
        psi_left = _dihedral(*left) + np.pi
        return cls(
            left,
            right,
            length,
            psi_left,
            float(np.linalg.norm(right[1] - right[0])),
            float(np.linalg.norm(right[2] - right[1])),
            _angle(*right[:3]),
        )

    def forward(self, torsions):
        """Return FK coordinates and torsion axes for analytic derivatives."""
        xyz = np.empty((self.length + 2, 4, 3), dtype=np.float64)
        xyz[0] = self.left
        origins = np.empty((2 * self.length + 1, 3), dtype=np.float64)
        axes = np.empty_like(origins)
        affected_from = np.empty(len(origins), dtype=int)
        for residue in range(1, self.length + 2):
            previous = xyz[residue - 1]
            psi = self.psi_left if residue == 1 else torsions[2 * residue - 3]
            xyz[residue, 0] = _place_atom(*previous[:3], _CN, _CACN, psi)
            final = residue == self.length + 1
            xyz[residue, 1] = _place_atom(
                previous[1],
                previous[2],
                xyz[residue, 0],
                self.right_nca if final else _NCA,
                _CNCA,
                np.pi,
            )
            phi_index = 2 * (residue - 1)
            xyz[residue, 2] = _place_atom(
                previous[2],
                xyz[residue, 0],
                xyz[residue, 1],
                self.right_cac if final else _CAC,
                self.right_angle if final else _NCAC,
                torsions[phi_index],
            )
            origins[phi_index] = xyz[residue, 0]
            axes[phi_index] = _unit(xyz[residue, 1] - xyz[residue, 0])
            affected_from[phi_index] = 4 * residue + 2
            if not final:
                psi_index = phi_index + 1
                xyz[residue, 3] = _place_atom(
                    *xyz[residue, :3],
                    _CO,
                    _CACO,
                    torsions[psi_index] + np.pi,
                )
                origins[psi_index] = xyz[residue, 1]
                axes[psi_index] = _unit(xyz[residue, 2] - xyz[residue, 1])
                affected_from[psi_index] = 4 * residue + 3
        # The last oxygen is outside the solved loop; it is never optimized.
        xyz[-1, 3] = self.right[3]
        return xyz, origins, axes, affected_from


def _point_jacobian(xyz, origins, axes, affected_from, indices):
    points = xyz.reshape(-1, 3)[indices]
    jac = np.cross(axes[None, :, :], points[:, None, :] - origins[None, :, :])
    jac *= (indices[:, None] >= affected_from[None, :])[:, :, None]
    # FK explicitly holds the right anchor oxygen constant.
    jac[indices == xyz.shape[0] * 4 - 1] = 0
    return jac.transpose(0, 2, 1)


def _nonlocal_clash_terms(
    xyz, origins, axes, affected, residue_pairs, right, *, clearance, weight
):
    """Nearest heavy-backbone pair per nonlocal residue pair, in final frame.

    The right anchor is fixed during this objective as it is in final output;
    its FK prediction is reserved for the separate closure residual. Only
    active nearest-pair atom derivatives are materialized, not all pairwise
    atom-by-torsion derivatives. At a tie the selected min is a subgradient.
    """
    from scipy.sparse import csr_matrix

    physical = xyz.copy()
    physical[-1] = right
    first, second = residue_pairs
    differences = physical[first, :, None, :] - physical[second, None, :, :]
    distances = np.linalg.norm(differences, axis=-1).reshape(len(first), 16)
    nearest = np.argmin(distances, axis=1)
    minimum = distances[np.arange(len(first)), nearest]
    scale = np.sqrt(weight)
    residual = scale * np.maximum(0.0, clearance - minimum)
    jac = csr_matrix((len(first), len(axes)))
    active = np.flatnonzero(minimum < clearance)
    if len(active):
        left_indices = 4 * first[active] + nearest[active] // 4
        right_indices = 4 * second[active] + nearest[active] % 4
        indices, inverse = np.unique(
            np.concatenate((left_indices, right_indices)), return_inverse=True
        )
        point_jac = _point_jacobian(physical, origins, axes, affected, indices)
        point_jac[indices >= 4 * (len(physical) - 1)] = 0
        difference = (
            physical.reshape(-1, 3)[left_indices]
            - physical.reshape(-1, 3)[right_indices]
        )
        direction = difference / np.maximum(minimum[active, None], 1e-12)
        active_jac = -scale * np.einsum(
            "pi,pij->pj",
            direction,
            point_jac[inverse[: len(active)]] - point_jac[inverse[len(active) :]],
        )
        rows, columns = np.nonzero(active_jac)
        jac = csr_matrix(
            (active_jac[rows, columns], (active[rows], columns)),
            shape=(len(first), len(axes)),
        )
    return residual, jac


def _reference_torsions(reference):
    length = len(reference) - 2
    torsions = np.empty(2 * length + 1)
    for residue in range(1, length + 2):
        torsions[2 * residue - 2] = _dihedral(
            reference[residue - 1, 2],
            *reference[residue, :3],
        )
        if residue <= length:
            torsions[2 * residue - 1] = _dihedral(
                *reference[residue, :3],
                reference[residue + 1, 0],
            )
    return torsions


def _resized_torsions(reference, old_secondary, secondary):
    """Resample torsions within declared corresponding blocks, never XYZ."""
    old_blocks = [(kind, len(list(run))) for kind, run in groupby(old_secondary)]
    new_blocks = [(kind, len(list(run))) for kind, run in groupby(secondary)]
    if [kind for kind, _ in old_blocks] != [kind for kind, _ in new_blocks]:
        raise ValueError(
            "Reference rebuilding requires the same ordered H/L block kinds"
        )
    source = _reference_torsions(reference)
    result = np.empty(2 * len(secondary) + 1)
    old_to_new = np.empty(len(old_secondary) + 2, dtype=int)
    old_to_new[0], old_to_new[-1] = 0, len(secondary) + 1
    old_offset = new_offset = 0
    for (_, old_length), (_, new_length) in zip(old_blocks, new_blocks):
        old_to_new[old_offset + 1 : old_offset + old_length + 1] = (
            new_offset
            + 1
            + np.rint(np.linspace(0, new_length - 1, old_length)).astype(int)
        )
        for slot in (0, 1):
            angles = source[2 * old_offset + slot : 2 * (old_offset + old_length) : 2]
            result[2 * new_offset + slot : 2 * (new_offset + new_length) : 2] = (
                np.interp(
                    np.linspace(0, 1, new_length),
                    np.linspace(0, 1, old_length),
                    np.unwrap(angles),
                )
            )
        old_offset += old_length
        new_offset += new_length
    result[-1] = source[-1]
    return result, old_to_new


def _geometry_report(backbone, *, bond_tolerance, angle_tolerance_deg, clash_distance):
    """Independent measurements on final coordinates, including BOTH junctions."""
    distance_errors, angle_errors, omega_errors, plane_errors = [], [], [], []
    for residue in range(1, len(backbone) - 1):
        n, ca, c, o = backbone[residue]
        distance_errors.extend(
            (
                abs(np.linalg.norm(n - ca) - _NCA),
                abs(np.linalg.norm(ca - c) - _CAC),
                abs(np.linalg.norm(c - o) - _CO),
            )
        )
        angle_errors.extend(
            (abs(_angle(n, ca, c) - _NCAC), abs(_angle(ca, c, o) - _CACO))
        )
    for left, right in zip(backbone[:-1], backbone[1:]):
        distance_errors.append(abs(np.linalg.norm(left[2] - right[0]) - _CN))
        angle_errors.extend(
            (
                abs(_angle(left[1], left[2], right[0]) - _CACN),
                abs(_angle(left[2], right[0], right[1]) - _CNCA),
            )
        )
        omega = _dihedral(left[1], left[2], right[0], right[1])
        omega_errors.append(abs((omega - np.pi + np.pi) % (2 * np.pi) - np.pi))
        # Carbonyl O, CA, C and next N must be in the same peptide plane.
        plane = _dihedral(left[3], left[1], left[2], right[0])
        plane_errors.append(min(abs(plane), abs(abs(plane) - np.pi)))
    # Exclude covalent and near-local contacts; this is NOT an all-atom audit.
    minimum = None
    for index in range(len(backbone) - 2):
        distances = np.linalg.norm(
            backbone[index, :, None, :]
            - backbone[index + 2 :].reshape(-1, 3)[None, :, :],
            axis=-1,
        )
        value = float(distances.min())
        minimum = value if minimum is None else min(minimum, value)
    result = {
        "maximum_bond_length_error_angstrom": float(max(distance_errors, default=0)),
        "maximum_bond_angle_error_deg": float(np.rad2deg(max(angle_errors, default=0))),
        "maximum_trans_omega_error_deg": float(
            np.rad2deg(max(omega_errors, default=0))
        ),
        "maximum_peptide_plane_error_deg": float(
            np.rad2deg(max(plane_errors, default=0))
        ),
        "minimum_nonlocal_backbone_distance_angstrom": minimum,
    }
    result["passed"] = bool(
        result["maximum_bond_length_error_angstrom"] <= bond_tolerance
        and result["maximum_bond_angle_error_deg"] <= angle_tolerance_deg
        and result["maximum_trans_omega_error_deg"] <= angle_tolerance_deg
        and result["maximum_peptide_plane_error_deg"] <= angle_tolerance_deg
        and (minimum is None or minimum >= clash_distance)
    )
    return result


def repair_backbone(
    reference_backbone: np.ndarray | None,
    target_left: np.ndarray,
    target_right: np.ndarray,
    *,
    generated_length: int,
    secondary_structure: str | None = None,
    reference_secondary_structure: str | None = None,
    seed: int = 0,
    attempts: int = 4,
    max_nfev: int = 150,
    closure_tolerance: float = 0.005,
    bond_tolerance: float = 0.04,
    angle_tolerance_deg: float = 8.0,
    clash_distance: float = 2.0,
    reference_contact_cutoff: float = 8.0,
    reference_sequence_separation: int = 8,
    maximum_reference_contact_error: float = 2.0,
    reference_shape_weight: float = 0.0,
    maximum_reference_ca_deviation: float | None = None,
    reference_ca_target: np.ndarray | None = None,
    nonlocal_clash_weight: float | None = None,
    clash_clearance_margin: float = 0.1,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find and independently verify a fixed-anchor backbone within a budget.

    ``secondary_structure`` assigns every generated residue to H (helical
    initialization) or L (turn/loop). Without a complete reference it is
    required. Changing a reference length additionally requires its original
    H/L assignment and corresponding block kinds; only torsions are resampled.
    Resizing is a new construction, not a claim to preserve the original fold.

    Reference nonlocal generated-CA contact distances are retained as explicit
    geometric targets, each within the stated error. For a declared rebuild,
    their residues map by relative position within corresponding H/L blocks.
    This is a reference-shape contract, not an energy or stability estimate.
    Neither a small endpoint gap nor optimizer success is accepted without
    final-coordinate verification. An unsuccessful search returns None.

    The least-squares residual concatenates Cartesian right N/CA/C errors
    divided by ``closure_tolerance`` and contact-distance errors divided by
    ``maximum_reference_contact_error``. Final acceptance checks every bound
    independently; a low aggregate objective cannot override a failed bound.

    Optional shape residuals are ``sqrt(reference_shape_weight) * (CA-target)``;
    the weight is in inverse square Angstroms. Coordinates must already share
    the fixed anchors' frame. A same-length reference supplies these CA targets
    unless ``reference_ca_target[L,3]`` overrides them. A changed length needs
    that explicit new target; Cartesian coordinates are never interpolated.
    ``maximum_reference_ca_deviation`` is a separate final per-CA bound, whether
    or not the shape objective has nonzero weight. No post-hoc alignment hides
    displacement relative to the fixed seed.

    For every residue pair separated by at least two sequence positions, a
    repulsive residual is ``sqrt(nonlocal_clash_weight) * max(0,
    clash_distance + clash_clearance_margin - nearest_backbone_atom_distance)``.
    A None weight resolves to ``1 / clash_distance**2`` (inverse square Å);
    zero explicitly disables this objective, never the independent final
    clash check. The small clearance margin is an optimization target only.
    """
    if (
        isinstance(generated_length, bool)
        or not isinstance(generated_length, int)
        or generated_length < 0
    ):
        raise ValueError("generated_length must be a nonnegative integer")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or not 1 <= attempts <= 32
    ):
        raise ValueError("attempts must be an integer from 1 to 32")
    if (
        isinstance(max_nfev, bool)
        or not isinstance(max_nfev, int)
        or not 1 <= max_nfev <= 5000
    ):
        raise ValueError("max_nfev must be an integer from 1 to 5000")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(reference_sequence_separation, bool)
        or not isinstance(reference_sequence_separation, int)
        or reference_sequence_separation < 2
    ):
        raise ValueError(
            "reference_sequence_separation must be an integer of at least two"
        )
    for name, value in (
        ("closure_tolerance", closure_tolerance),
        ("bond_tolerance", bond_tolerance),
        ("angle_tolerance_deg", angle_tolerance_deg),
        ("clash_distance", clash_distance),
        ("reference_contact_cutoff", reference_contact_cutoff),
        ("maximum_reference_contact_error", maximum_reference_contact_error),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(reference_shape_weight) or reference_shape_weight < 0:
        raise ValueError("reference_shape_weight must be nonnegative and finite")
    if maximum_reference_ca_deviation is not None and (
        not np.isfinite(maximum_reference_ca_deviation)
        or maximum_reference_ca_deviation <= 0
    ):
        raise ValueError("maximum_reference_ca_deviation must be positive and finite")
    if nonlocal_clash_weight is None:
        nonlocal_clash_weight = 1.0 / clash_distance**2
    if not np.isfinite(nonlocal_clash_weight) or nonlocal_clash_weight < 0:
        raise ValueError("nonlocal_clash_weight must be nonnegative and finite")
    if not np.isfinite(clash_clearance_margin) or clash_clearance_margin < 0:
        raise ValueError("clash_clearance_margin must be nonnegative and finite")
    left = _checked_backbone(
        np.asarray(target_left)[None], name="target_left", residues=1
    )[0]
    right = _checked_backbone(
        np.asarray(target_right)[None], name="target_right", residues=1
    )[0]
    model = _ChainModel.from_anchors(left, right, generated_length)
    secondary_structure = _secondary(
        secondary_structure, generated_length, name="secondary_structure"
    )
    reference = (
        None
        if reference_backbone is None
        else _checked_backbone(reference_backbone, name="reference_backbone")
    )
    if reference is not None and len(reference) < 2:
        raise ValueError("reference_backbone must contain both anchor residues")
    if reference is not None:
        reference_secondary_structure = _secondary(
            reference_secondary_structure,
            len(reference) - 2,
            name="reference_secondary_structure",
        )
    if reference is None and secondary_structure is None:
        raise ValueError(
            "Construction without a reference requires an explicit H/L assignment"
        )
    same_length = reference is not None and len(reference) == generated_length + 2
    if (
        reference is not None
        and not same_length
        and (secondary_structure is None or reference_secondary_structure is None)
    ):
        raise ValueError(
            "Reference length mismatch requires both explicit H/L block assignments"
        )

    if reference_ca_target is not None:
        ca_target = np.asarray(reference_ca_target, dtype=np.float64)
        if ca_target.shape != (generated_length, 3) or not np.isfinite(ca_target).all():
            raise ValueError(
                "reference_ca_target must have finite shape [generated_length, 3]"
            )
        ca_target = ca_target.copy()
        shape_source = "explicit_generated_CA_target"
    elif same_length:
        ca_target = reference[1:-1, 1].copy()
        shape_source = "same_length_reference_in_anchor_frame"
    else:
        ca_target, shape_source = None, None
    if ca_target is None and (
        reference_shape_weight > 0 or maximum_reference_ca_deviation is not None
    ):
        raise ValueError(
            "Reference shape restraint requires a same-length reference or explicit reference_ca_target"
        )

    report: dict[str, Any] = {
        "status": "unresolved",
        "geometry_verified": False,
        "generated_length": generated_length,
        "atom_order": list(_ATOM_NAMES),
        "seed": seed,
        "attempt_budget": attempts,
        "max_nfev_per_attempt": max_nfev,
        "solver": "scipy.optimize.least_squares",
        "trust_region_linear_solver": "lsmr",
        "secondary_structure_initialization": secondary_structure,
        "reference_secondary_structure": reference_secondary_structure,
        "reference_shape_weight_per_angstrom_squared": float(reference_shape_weight),
        "maximum_reference_ca_deviation_angstrom": maximum_reference_ca_deviation,
        "reference_shape_target_source": shape_source,
        "nonlocal_clash_weight_per_angstrom_squared": float(nonlocal_clash_weight),
        "clash_clearance_margin_angstrom": float(clash_clearance_margin),
        "nonlocal_clash_target_distance_angstrom": clash_distance
        + clash_clearance_margin,
        "closure_tolerance_angstrom": closure_tolerance,
        "bond_tolerance_angstrom": bond_tolerance,
        "angle_tolerance_deg": angle_tolerance_deg,
        "minimum_nonlocal_backbone_distance_angstrom": clash_distance,
        "reference_contact_cutoff_angstrom": reference_contact_cutoff,
        "reference_sequence_separation": reference_sequence_separation,
        "maximum_reference_contact_error_angstrom": maximum_reference_contact_error,
        "attempts": [],
        "limitations": [
            "Backbone geometric witness, not sequence designability or folding success",
            "Other seed atoms, other chains and sidechains require the caller's assembly audit",
            "No residue-specific Ramachandran or sidechain validation is performed",
            "Finite search failure is not a proof of global infeasibility",
            "H/L assignments initialize torsions; they do not certify secondary structure",
        ],
    }
    # A rigorous contour bound for this fixed-bond internal-coordinate model.
    contour = (generated_length + 1) * _CN + generated_length * (_NCA + _CAC)
    if np.linalg.norm(left[2] - right[0]) > contour + closure_tolerance:
        report.update(
            reason="target_exceeds_internal_coordinate_contour",
            model_contour_upper_bound_angstrom=contour,
        )
        return None, report

    rng = np.random.default_rng(seed)
    if same_length:
        initial = _reference_torsions(reference)
        old_to_new = np.arange(len(reference))
        report["initialization"] = "reference_torsions"
        report["reference_contact_mapping"] = "same_residue_indices"
    elif reference is not None:
        initial, old_to_new = _resized_torsions(
            reference, reference_secondary_structure, secondary_structure
        )
        report["initialization"] = "explicit_HL_block_torsion_rebuild"
        report["reference_contact_mapping"] = (
            "relative_position_within_declared_HL_blocks"
        )
        report["reference_length"] = len(reference) - 2
    else:
        initial = np.empty(2 * generated_length + 1)
        for index, kind in enumerate(secondary_structure):
            initial[2 * index : 2 * index + 2] = np.deg2rad(
                (-60, -45) if kind == "H" else (-90, 90)
            )
        initial[-1] = np.deg2rad(-60)
        report["initialization"] = "explicit_HL_internal_coordinates"

    # Targets are independent of the reference's arbitrary rigid-body frame.
    contact_pairs, reference_distances = [], []
    if reference is not None:
        for i in range(1, len(reference) - 1):
            for j in range(i + reference_sequence_separation, len(reference) - 1):
                distance = float(np.linalg.norm(reference[i, 1] - reference[j, 1]))
                mapped = (int(old_to_new[i]), int(old_to_new[j]))
                if (
                    distance <= reference_contact_cutoff
                    and mapped[1] - mapped[0] >= reference_sequence_separation
                ):
                    contact_pairs.append(mapped)
                    reference_distances.append(distance)
    pair_array = np.asarray(contact_pairs, dtype=int).reshape(-1, 2)
    reference_distances = np.asarray(reference_distances)
    report["reference_contact_count"] = len(contact_pairs)
    report["reference_packing_contract"] = bool(contact_pairs)
    report["reference_contact_pairs"] = [list(pair) for pair in contact_pairs]
    report["reference_contact_distances_angstrom"] = reference_distances.tolist()
    end_indices = np.arange(4 * (generated_length + 1), 4 * (generated_length + 1) + 3)
    ca_indices = 4 * np.arange(generated_length + 2) + 1
    nonlocal_pairs = np.triu_indices(generated_length + 2, k=2)
    report["nonlocal_clash_residue_pair_count"] = len(nonlocal_pairs[0])

    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix, vstack

    cached_q = cached_value = None

    def evaluate(q):
        nonlocal cached_q, cached_value
        if cached_q is None or not np.array_equal(q, cached_q):
            xyz, origins, axes, affected = model.forward(q)
            closure = (xyz[-1, :3] - right[:3]).reshape(-1)
            jac = _point_jacobian(xyz, origins, axes, affected, end_indices).reshape(
                9, -1
            )
            residual = closure / closure_tolerance
            jac = jac / closure_tolerance
            ca_jac = None
            if len(pair_array) or reference_shape_weight > 0:
                ca_jac = _point_jacobian(xyz, origins, axes, affected, ca_indices)
            if len(pair_array):
                difference = xyz[pair_array[:, 0], 1] - xyz[pair_array[:, 1], 1]
                distances = np.linalg.norm(difference, axis=-1)
                directions = difference / np.maximum(distances[:, None], 1e-12)
                contact_jac = np.einsum(
                    "pi,pij->pj",
                    directions,
                    ca_jac[pair_array[:, 0]] - ca_jac[pair_array[:, 1]],
                )
                residual = np.concatenate(
                    (
                        residual,
                        (distances - reference_distances)
                        / maximum_reference_contact_error,
                    )
                )
                jac = np.concatenate(
                    (jac, contact_jac / maximum_reference_contact_error)
                )
            if reference_shape_weight > 0:
                scale = np.sqrt(reference_shape_weight)
                residual = np.concatenate(
                    (residual, scale * (xyz[1:-1, 1] - ca_target).reshape(-1))
                )
                jac = np.concatenate(
                    (jac, scale * ca_jac[1:-1].reshape(3 * generated_length, len(q)))
                )
            if nonlocal_clash_weight > 0 and len(nonlocal_pairs[0]):
                clash_residual, clash_jac = _nonlocal_clash_terms(
                    xyz,
                    origins,
                    axes,
                    affected,
                    nonlocal_pairs,
                    right,
                    clearance=clash_distance + clash_clearance_margin,
                    weight=nonlocal_clash_weight,
                )
                residual = np.concatenate((residual, clash_residual))
                # Most residue pairs are outside the repulsive support. Keep
                # those fixed residual slots, but do not repeatedly multiply
                # their zero Jacobian rows in the inner LSMR iterations.
                jac = vstack((csr_matrix(jac), clash_jac), format="csr")
            cached_q, cached_value = q.copy(), (residual, jac)
        return cached_value

    for attempt in range(attempts):
        trial = initial.copy()
        if attempt:
            scale = np.full(len(trial), np.deg2rad(15.0))
            if secondary_structure:
                for index, kind in enumerate(secondary_structure):
                    if kind == "L":
                        scale[2 * index : 2 * index + 2] = np.deg2rad(60.0)
            trial += rng.normal(size=len(trial)) * scale
        # A rigid terminal frame has only six independent Cartesian freedoms;
        # loop torsions usually outnumber constraints. LSMR avoids unstable
        # dense-SVD nullspace steps (covered by the rigid-frame regression).
        result = least_squares(
            lambda q: evaluate(q)[0],
            trial,
            jac=lambda q: evaluate(q)[1],
            tr_solver="lsmr",
            max_nfev=max_nfev,
            tr_options={"atol": 1e-12, "btol": 1e-12, "maxiter": 10 * len(trial)},
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        candidate = model.forward(result.x)[0]
        closure_error = float(
            np.linalg.norm(candidate[-1, :3] - right[:3], axis=-1).max()
        )
        # Immutable anchors are restored before the independent geometry audit.
        candidate[0], candidate[-1] = left, right
        geometry = _geometry_report(
            candidate,
            bond_tolerance=bond_tolerance,
            angle_tolerance_deg=angle_tolerance_deg,
            clash_distance=clash_distance,
        )
        contact_error = (
            float(
                np.max(
                    np.abs(
                        np.linalg.norm(
                            candidate[pair_array[:, 0], 1]
                            - candidate[pair_array[:, 1], 1],
                            axis=-1,
                        )
                        - reference_distances
                    )
                )
            )
            if len(pair_array)
            else None
        )
        ca_displacement = (
            float(
                np.max(
                    np.linalg.norm(candidate[1:-1, 1] - ca_target, axis=-1), initial=0.0
                )
            )
            if ca_target is not None
            else None
        )
        accepted = bool(
            closure_error <= closure_tolerance
            and geometry["passed"]
            and (
                contact_error is None
                or contact_error <= maximum_reference_contact_error
            )
            and (
                maximum_reference_ca_deviation is None
                or ca_displacement <= maximum_reference_ca_deviation
            )
        )
        record = {
            "index": attempt,
            "nfev": int(result.nfev),
            "optimizer_success": bool(result.success),
            "maximum_anchor_closure_error_angstrom": closure_error,
            "maximum_reference_contact_error_angstrom": contact_error,
            "generated_ca_maximum_displacement_angstrom": ca_displacement,
            "geometry": geometry,
            "accepted": accepted,
        }
        report["attempts"].append(record)
        if accepted:
            report.update(
                status="verified_backbone_geometry",
                geometry_verified=True,
                fixed_anchor_maximum_displacement_angstrom=0.0,
                generated_ca_maximum_displacement_angstrom=ca_displacement,
                selected_attempt=attempt,
            )
            return candidate, report
    report["reason"] = "no_verified_backbone_within_search_budget"
    return None, report
