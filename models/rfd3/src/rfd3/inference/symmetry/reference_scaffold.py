"""Tensor binding and clean-coordinate residuals for an explicit scaffold."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReferenceScaffold:
    contract: dict
    ca_atom_indices: torch.Tensor
    reference_ca: torch.Tensor
    fixed_ca: torch.Tensor
    chains: tuple[torch.Tensor, ...]
    blocks: dict[str, torch.Tensor]
    reference_separations: dict[tuple[int, int], float]
    backbone_atom_indices: torch.Tensor | None = None
    reference_backbone: torch.Tensor | None = None


def bind_reference_scaffold(features, topology) -> ReferenceScaffold | None:
    raw = features.get("mosaic_scaffold_contract")
    if raw is None:
        return None
    from rfd3_mosaic.validation.scaffold_contract import (
        _segment_distances,
        contract_arrays,
        validate_scaffold_contract,
    )

    contract = validate_scaffold_contract(raw)
    reference, fixed, chains, blocks = contract_arrays(contract)
    if "mosaic_scaffold_ca_atom_indices" not in features:
        raise ValueError("Scaffold contract lacks validated CA identity binding")
    device = topology.atom_to_token.device
    indices = torch.as_tensor(
        features["mosaic_scaffold_ca_atom_indices"], dtype=torch.long, device=device
    )
    all_ca = torch.cat([chain.ca_atom_indices for chain in topology.chains])
    if (
        indices.ndim != 1
        or len(indices) != len(reference)
        or len(indices) != len(all_ca)
        or not torch.equal(torch.sort(indices).values, torch.sort(all_ca).values)
    ):
        raise ValueError("Scaffold contract must bind every protein CA exactly once")
    actual_fixed = ~topology.generated_atom_mask[indices]
    if not np.array_equal(actual_fixed.cpu().numpy(), fixed):
        raise ValueError("Scaffold contract fixed CA mask differs from runtime")
    mobility = features.get("motif_constraint_orbit_mobility_mode")
    if (
        mobility is not None
        and bool(torch.any(torch.as_tensor(mobility) != 0))
        and features.get("mosaic_reference_transport") is None
    ):
        raise ValueError(
            "Mobile scaffold contract requires a validated coupled reference transport plan"
        )
    motif = features.get("motif_pos")
    if motif is None:
        raise ValueError("Scaffold contract requires runtime fixed motif coordinates")
    motif = torch.as_tensor(motif, device=device)
    if motif.ndim == 3 and motif.shape[0] == 1:
        motif = motif[0]
    if motif.shape != (len(topology.atom_to_token), 3):
        raise ValueError("Scaffold contract motif coordinate shape mismatch")
    # Never quantize the reference to bf16/float16 merely because network
    # activations use that dtype; this would conceal a damaged fixed frame.
    geometry_dtype = (
        torch.float32 if motif.dtype in (torch.float16, torch.bfloat16) else motif.dtype
    )
    reference_tensor = torch.as_tensor(reference, dtype=geometry_dtype, device=device)
    if not torch.isfinite(motif[indices[actual_fixed]]).all() or bool(
        torch.any(
            torch.linalg.vector_norm(
                motif[indices[actual_fixed]] - reference_tensor[actual_fixed], dim=-1
            )
            > contract["limits"]["fixed_ca_tolerance"]
        )
    ):
        raise ValueError(
            "Scaffold reference and fixed motif coordinates are not in the same declared frame"
        )
    separations = {}
    for i, a in enumerate(chains):
        for j in range(i + 1, len(chains)):
            b = chains[j]
            separations[i, j] = float(
                _segment_distances(
                    reference[a[:-1]],
                    reference[a[1:]],
                    reference[b[:-1]],
                    reference[b[1:]],
                ).min()
            )
    backbone_indices = reference_backbone = None
    if contract["schema_version"] == 2:
        raw_indices = features.get("mosaic_scaffold_backbone_atom_indices")
        if raw_indices is None:
            raise ValueError("Version-2 scaffold lacks validated N/CA/C/O atom binding")
        backbone_indices = torch.as_tensor(raw_indices, dtype=torch.long, device=device)
        if (
            backbone_indices.shape != (len(reference), 4)
            or not torch.equal(backbone_indices[:, 1], indices)
            or bool(torch.any(backbone_indices < 0))
            or bool(torch.any(backbone_indices >= len(topology.atom_to_token)))
            or len(torch.unique(backbone_indices)) != backbone_indices.numel()
        ):
            raise ValueError("Scaffold backbone atom identity mapping is invalid")
        reference_bb = np.asarray(
            [r["reference_backbone"] for r in contract["residues"]]
        )
        reference_backbone = torch.as_tensor(
            reference_bb, dtype=geometry_dtype, device=device
        )
        if not torch.equal(
            ~topology.generated_atom_mask[backbone_indices],
            actual_fixed[:, None].expand(-1, 4),
        ):
            raise ValueError("Scaffold backbone fixed mask differs from reference")
        if not torch.isfinite(motif[backbone_indices[actual_fixed]]).all() or bool(
            torch.any(
                torch.linalg.vector_norm(
                    motif[backbone_indices[actual_fixed]]
                    - reference_backbone[actual_fixed],
                    dim=-1,
                )
                > contract["limits"]["fixed_ca_tolerance"]
            )
        ):
            raise ValueError(
                "Scaffold fixed backbone differs from the declared reference frame"
            )
    return ReferenceScaffold(
        contract,
        indices,
        reference_tensor,
        actual_fixed,
        tuple(torch.as_tensor(c, dtype=torch.long, device=device) for c in chains),
        {
            k: torch.as_tensor(v, dtype=torch.long, device=device)
            for k, v in blocks.items()
        },
        separations,
        backbone_indices,
        reference_backbone,
    )


def reference_scaffold_deficits(coordinates, reference: ReferenceScaffold):
    """Same residual order/units as the independent NumPy final audit."""
    from .scaffold_core_guidance import _segment_to_segment_distances

    xyz = coordinates[reference.ca_atom_indices]
    target = reference.reference_ca.to(dtype=xyz.dtype)
    limits = reference.contract["limits"]
    deviation = torch.linalg.vector_norm(xyz - target, dim=-1)
    terms = []

    def add(value):
        terms.append(torch.relu(value.reshape(-1)))

    add(deviation[~reference.fixed_ca] - limits["maximum_ca_deviation"])
    add(deviation[reference.fixed_ca] - limits["fixed_ca_tolerance"])
    for edge in reference.contract["support_edges"]:
        for left, right in (
            (edge["left"], edge["right"]),
            (edge["right"], edge["left"]),
        ):
            ids, partners = reference.blocks[left], reference.blocks[right]
            nearest = torch.cdist(xyz[ids], xyz[partners]).min(dim=1).values
            count = math.ceil(limits["minimum_helix_contact_fraction"] * len(ids))
            add(nearest.sort().values[:count] - limits["contact_distance"])
            if (
                "maximum_unsupported_run" in limits
                and reference.contract["schema_version"] == 1
            ):
                width = limits["maximum_unsupported_run"] + 1
                if width <= len(ids):
                    add(
                        nearest.unfold(0, width, 1).min(dim=1).values
                        - limits["contact_distance"]
                    )
    if reference.contract["schema_version"] == 2:
        neighbours = {name: set() for name in reference.blocks}
        for edge in reference.contract["support_edges"]:
            neighbours[edge["left"]].add(edge["right"])
            neighbours[edge["right"]].add(edge["left"])
        width = limits["maximum_unsupported_run"] + 1
        for name, ids in reference.blocks.items():
            if not torch.any(~reference.fixed_ca[ids]) or not neighbours[name]:
                continue
            partners = torch.cat(
                [reference.blocks[other] for other in sorted(neighbours[name])]
            )
            nearest = torch.cdist(xyz[ids], xyz[partners]).min(dim=1).values
            if width <= len(ids):
                add(
                    nearest.unfold(0, width, 1).min(dim=1).values
                    - limits["contact_distance"]
                )
    for i, ids in enumerate(reference.chains):
        bonds = torch.linalg.vector_norm(xyz[ids[1:]] - xyz[ids[:-1]], dim=-1)
        add(limits["minimum_ca_bond_distance"] - bonds)
        add(bonds - limits["maximum_ca_bond_distance"])
        for j in range(i + 1, len(reference.chains)):
            other = reference.chains[j]
            lower = (
                reference.reference_separations[i, j]
                - deviation[ids].max()
                - deviation[other].max()
            )
            actual = _segment_to_segment_distances(
                xyz[ids[:-1]], xyz[ids[1:]], xyz[other[:-1]], xyz[other[1:]]
            ).min()
            add(limits["minimum_interchain_segment_distance"] - lower)
            add(limits["minimum_interchain_segment_distance"] - actual)
    return torch.cat(terms)


def reference_backbone_deficits(coordinates, reference: ReferenceScaffold):
    """Smooth geometry guidance for v2; final actual-alpha audit is authoritative.

    Distances are in Angstrom. Angular excess is multiplied by 1.329 A
    (standard peptide C--N length), yielding an equivalent arc displacement.
    This normalization is a declared engineering scale, not a physical energy.
    No hydrogen-bond energy is mixed into the distance objective. N/CA/C/O
    remain close to the reference while genuine helix identity/support is
    independently remeasured by the required final audit.
    """
    if reference.backbone_atom_indices is None:
        return coordinates.new_empty(0)
    xyz = coordinates[reference.backbone_atom_indices]
    policy = reference.contract["backbone_policy"]
    generated = ~reference.fixed_ca
    terms = []

    def add(value):
        terms.append(torch.relu(value.reshape(-1)))

    def norm(value):
        return torch.linalg.vector_norm(value, dim=-1)

    def unit(value):
        return value / norm(value).clamp_min(1e-8)[..., None]

    def angle(a, b, c):
        cosine = (unit(a - b) * unit(c - b)).sum(dim=-1)
        return torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))

    def angular(actual, target, tolerance):
        return (
            torch.abs(actual - math.radians(target)) - math.radians(tolerance)
        ) * 1.329

    bt = policy["bond_length_tolerance_angstrom"]
    at = policy["bond_angle_tolerance_degrees"]
    pt = math.radians(policy["peptide_planarity_tolerance_degrees"])
    target = reference.reference_backbone.to(dtype=xyz.dtype)
    add(
        norm(xyz[generated] - target[generated])
        - reference.contract["limits"]["maximum_ca_deviation"]
    )
    names = [r["residue_name"] for r in reference.contract["residues"]]
    nca_target = xyz.new_tensor(
        [
            1.451 if name == "GLY" else 1.466 if name == "PRO" else 1.458
            for name in names
        ]
    )
    cac_target = xyz.new_tensor([1.516 if name == "GLY" else 1.525 for name in names])
    add((norm(xyz[:, 0] - xyz[:, 1]) - nca_target).abs()[generated] - bt)
    add((norm(xyz[:, 1] - xyz[:, 2]) - cac_target).abs()[generated] - bt)
    add((norm(xyz[:, 2] - xyz[:, 3]) - 1.231).abs()[generated] - bt)
    add(angular(angle(xyz[:, 0], xyz[:, 1], xyz[:, 2]), 111.2, at)[generated])
    add(angular(angle(xyz[:, 1], xyz[:, 2], xyz[:, 3]), 120.8, at)[generated])
    # Full atom binding supplies physical chain identity, never token offsets
    # guessed from an assumed N/CA/C/O storage order.
    chain_id = torch.empty(len(xyz), dtype=torch.long, device=xyz.device)
    for index, ids in enumerate(reference.chains):
        chain_id[ids] = index
    connected = chain_id[:-1] == chain_id[1:]
    relevant = connected & (generated[:-1] | generated[1:])
    cn_target = xyz.new_tensor(
        [1.341 if name == "PRO" else 1.329 for name in names[1:]]
    )
    add((norm(xyz[:-1, 2] - xyz[1:, 0]) - cn_target).abs()[relevant] - bt)
    add(angular(angle(xyz[:-1, 1], xyz[:-1, 2], xyz[1:, 0]), 116.2, at)[relevant])
    add(angular(angle(xyz[:-1, 2], xyz[1:, 0], xyz[1:, 1]), 121.7, at)[relevant])
    add(angular(angle(xyz[:-1, 3], xyz[:-1, 2], xyz[1:, 0]), 123.0, at)[relevant])
    peptide = xyz[1:, 0] - xyz[:-1, 2]
    first = unit(torch.linalg.cross(xyz[:-1, 1] - xyz[:-1, 2], peptide))
    second = unit(torch.linalg.cross(peptide, xyz[1:, 1] - xyz[1:, 0]))
    # |cos omega| treats planar cis and trans equally, matching the final audit.
    omega = torch.acos((first * second).sum(-1).abs().clamp(max=1 - 1e-7))
    carbonyl_plane = unit(torch.linalg.cross(peptide, xyz[:-1, 3] - xyz[1:, 0]))
    carbonyl = torch.acos((first * carbonyl_plane).sum(-1).abs().clamp(max=1 - 1e-7))
    add((omega[relevant] - pt) * 1.329)
    add((carbonyl[relevant] - pt) * 1.329)
    # One nearest N/CA/C/O pair per relevant residue pair has the same
    # feasible set as testing all 16 pairs. Select in bounded no-grad chunks,
    # then differentiate only selected distances, avoiding a dense atom-pair
    # autograd graph for the complete expanded assembly.
    all_ids = torch.arange(len(xyz), device=xyz.device)
    for start in range(0, len(xyz), 128):
        ids = all_ids[start : start + 128]
        with torch.no_grad():
            distances = torch.cdist(xyz[ids].reshape(-1, 3), xyz.reshape(-1, 3))
            distances = (
                distances.reshape(len(ids), 4, len(xyz), 4)
                .permute(0, 2, 1, 3)
                .reshape(len(ids), len(xyz), 16)
            )
            nearest = distances.argmin(-1)
            relevant = (
                (generated[ids, None] | generated[None, :])
                & (
                    (chain_id[ids, None] != chain_id[None, :])
                    | ((ids[:, None] - all_ids[None, :]).abs() > 1)
                )
                & (ids[:, None] < all_ids[None, :])
            )
            left, right = relevant.nonzero(as_tuple=True)
            pair = nearest[left, right]
        distance = norm(xyz[ids[left], pair // 4] - xyz[right, pair % 4])
        add(policy["minimum_nonlocal_backbone_distance_angstrom"] - distance)
    return torch.cat(terms)


def reference_scaffold_guidance_deficits(coordinates, reference: ReferenceScaffold):
    """CA contract plus explicit v2 backbone guidance, all in A-equivalent units."""
    return torch.cat(
        (
            reference_scaffold_deficits(coordinates, reference),
            reference_backbone_deficits(coordinates, reference),
        )
    )
