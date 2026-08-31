"""Differentiable intra-chain scaffold packing with explicit safety control.

This objective complements, rather than replaces, graph interface guidance.
The fixed motif projector remains authoritative.  Only generated tokens are
translated, and every proposal is projected back through the exact Mosaic
constraint/symmetry runtime before it can be accepted.

``intra_chain_weight`` rewards a supported monomer core.  The three first
version terms deliberately stay small and interpretable: long-range contacts,
an upper hinge on length-normalized radius of gyration, and per-residue
tertiary-contact support.  ``inter_chain_weight`` follows RFdiffusion's
contact-map semantics and is consumed only by declared graph-interface edges;
it does not implicitly become a repulsive core term when no such edge exists.
An expert may independently request a soft *excess* penalty through
``inter_chain_excess_penalty``.  Its default is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ScaffoldCoreChain:
    asym_id: int
    ca_atom_indices: torch.Tensor
    residue_indices: torch.Tensor
    generated_ca_mask: torch.Tensor
    ca_segment_atom_pairs: torch.Tensor
    generated_segment_mask: torch.Tensor


@dataclass(frozen=True)
class ScaffoldCoreGeneratedRun:
    """One generated polymer run bounded by two fixed CA anchors."""

    asym_id: int
    generated_ca_atom_indices: torch.Tensor
    generated_token_indices: torch.Tensor
    left_anchor_ca_atom_index: int
    right_anchor_ca_atom_index: int


@dataclass(frozen=True)
class ScaffoldCoreTopology:
    chains: tuple[ScaffoldCoreChain, ...]
    generated_runs: tuple[ScaffoldCoreGeneratedRun, ...]
    atom_to_token: torch.Tensor
    generated_token_mask: torch.Tensor
    generated_atom_mask: torch.Tensor
    adjacent_token_pairs: torch.Tensor
    adjacent_ca_atom_pairs: torch.Tensor
    adjacent_pair_colors: torch.Tensor
    directed_continuity_groups: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ScaffoldCoreGuidanceConfig:
    intra_chain_weight: float = 0.0
    inter_chain_weight: float = 1.0
    inter_chain_excess_penalty: float = 0.0
    long_range_contact_weight: float = 0.75
    normalized_rg_weight: float = 0.35
    tertiary_support_weight: float = 1.0
    worst_support_weight: float = 1.0
    inter_chain_excess_weight: float = 1.0
    clash_weight: float = 8.0
    continuity_weight: float = 2.0
    routing_ownership_weight: float = 0.0
    contact_distance: float = 8.0
    contact_softness: float = 0.75
    sequence_separation: int = 8
    target_contacts_per_generated_residue: float = 1.5
    target_supported_contacts: float = 2.0
    worst_support_temperature: float = 0.25
    target_normalized_rg: float = 2.60
    incidental_inter_chain_fraction: float = 0.08
    clash_distance: float = 3.2
    backbone_distance: float = 3.8
    backbone_tolerance: float = 0.55
    start_fraction: float = 0.05
    end_fraction: float = 0.90
    maximum_token_step: float = 0.20
    maximum_adjacent_token_step_difference: float = 0.08
    line_search_steps: int = 5
    line_search_contraction: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "intra_chain_weight",
            "inter_chain_weight",
            "inter_chain_excess_penalty",
            "long_range_contact_weight",
            "normalized_rg_weight",
            "tertiary_support_weight",
            "worst_support_weight",
            "inter_chain_excess_weight",
            "clash_weight",
            "continuity_weight",
            "routing_ownership_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "contact_distance",
            "contact_softness",
            "target_contacts_per_generated_residue",
            "target_supported_contacts",
            "worst_support_temperature",
            "target_normalized_rg",
            "clash_distance",
            "backbone_distance",
            "backbone_tolerance",
            "maximum_token_step",
            "maximum_adjacent_token_step_difference",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.sequence_separation < 2:
            raise ValueError("sequence_separation must be at least two")
        if not 0.0 <= self.incidental_inter_chain_fraction < 1.0:
            raise ValueError("incidental_inter_chain_fraction must be in [0, 1)")
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError("Scaffold core guidance requires 0 <= start < end <= 1")
        if self.line_search_steps < 1:
            raise ValueError("line_search_steps must be positive")
        if not 0.0 < self.line_search_contraction < 1.0:
            raise ValueError("line_search_contraction must be in (0, 1)")


@dataclass(frozen=True)
class ScaffoldCoreEnergy:
    total: torch.Tensor
    long_range_contacts: torch.Tensor
    normalized_rg: torch.Tensor
    tertiary_support: torch.Tensor
    worst_support: torch.Tensor
    inter_chain_excess: torch.Tensor
    clash: torch.Tensor
    cross_chain_segment_clash: torch.Tensor
    continuity: torch.Tensor
    routing_ownership: torch.Tensor
    mean_normalized_rg: torch.Tensor
    mean_tertiary_support_fraction: torch.Tensor
    generated_inter_chain_contact_pairs: torch.Tensor
    generated_inter_chain_contact_coverage: torch.Tensor
    minimum_generated_inter_chain_distance: torch.Tensor
    minimum_cross_chain_segment_distance: torch.Tensor
    routing_ownership_violation_fraction: torch.Tensor
    maximum_routing_ownership_excess: torch.Tensor

    def detached_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in (
                "total",
                "long_range_contacts",
                "normalized_rg",
                "tertiary_support",
                "worst_support",
                "inter_chain_excess",
                "clash",
                "cross_chain_segment_clash",
                "continuity",
                "routing_ownership",
                "mean_normalized_rg",
                "mean_tertiary_support_fraction",
                "generated_inter_chain_contact_pairs",
                "generated_inter_chain_contact_coverage",
                "minimum_generated_inter_chain_distance",
                "minimum_cross_chain_segment_distance",
                "routing_ownership_violation_fraction",
                "maximum_routing_ownership_excess",
            )
        }


def _tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def worst_support_deficit_energy(
    generated_support: torch.Tensor,
    config: ScaffoldCoreGuidanceConfig,
) -> torch.Tensor:
    """Smoothly emphasize the worst contiguous tertiary-support deficit."""

    if generated_support.ndim != 1 or not len(generated_support):
        raise ValueError("generated_support must be one non-empty vector")
    support_deficit = torch.square(
        torch.relu(config.target_supported_contacts - generated_support)
    )
    window_size = min(config.sequence_separation, len(support_deficit))
    window_deficit = support_deficit.unfold(0, window_size, 1).mean(dim=-1)
    temperature = torch.as_tensor(
        config.worst_support_temperature,
        dtype=generated_support.dtype,
        device=generated_support.device,
    )
    return temperature * (
        torch.logsumexp(window_deficit / temperature, dim=0)
        - torch.log(
            torch.as_tensor(
                len(window_deficit),
                dtype=generated_support.dtype,
                device=generated_support.device,
            )
        )
    )


def build_scaffold_core_topology(
    f: dict[str, Any],
    fixed_atom_mask: torch.Tensor,
) -> ScaffoldCoreTopology:
    """Resolve protein CA chains and generated tokens from runtime features."""

    required = {"atom_to_token_map", "asym_id", "is_ca"}
    missing = required - set(f)
    if missing:
        raise ValueError(f"Scaffold core guidance requires features {sorted(missing)}")
    fixed = _tensor(fixed_atom_mask, dtype=torch.bool)
    if fixed.ndim != 1:
        raise ValueError("fixed_atom_mask must have shape [L]")
    device = fixed.device
    atom_to_token = _tensor(f["atom_to_token_map"], dtype=torch.long, device=device)
    is_ca = _tensor(f["is_ca"], dtype=torch.bool, device=device)
    if atom_to_token.shape != fixed.shape or is_ca.shape != fixed.shape:
        raise ValueError("atom_to_token_map, is_ca and fixed mask must share [L]")
    if atom_to_token.numel() == 0 or torch.any(atom_to_token < 0):
        raise ValueError("atom_to_token_map must contain non-negative token IDs")
    token_count = int(atom_to_token.max().item()) + 1
    asym_id = _tensor(f["asym_id"], dtype=torch.long, device=device)
    if asym_id.shape != (token_count,):
        raise ValueError("asym_id must have shape [N_tokens]")
    residue_index = _tensor(
        f.get("residue_index", torch.arange(token_count, device=device)),
        dtype=torch.long,
        device=device,
    )
    if residue_index.shape != (token_count,):
        raise ValueError("residue_index must have shape [N_tokens]")
    is_protein = _tensor(
        f.get("is_protein", torch.ones(token_count, device=device)),
        dtype=torch.bool,
        device=device,
    )
    if is_protein.shape != (token_count,):
        raise ValueError("is_protein must have shape [N_tokens]")

    fixed_counts = torch.zeros(token_count, dtype=torch.long, device=device)
    atom_counts = torch.zeros_like(fixed_counts)
    atom_counts.index_add_(0, atom_to_token, torch.ones_like(atom_to_token))
    fixed_counts.index_add_(0, atom_to_token, fixed.to(dtype=torch.long))
    token_fixed = fixed_counts == atom_counts
    token_generated = ~token_fixed
    generated_atom_mask = token_generated[atom_to_token]

    ca_atom_indices = torch.nonzero(is_ca, as_tuple=False).reshape(-1)
    ca_tokens = atom_to_token[ca_atom_indices]
    chains: list[ScaffoldCoreChain] = []
    adjacent_token_pairs: list[tuple[int, int]] = []
    adjacent_ca_atom_pairs: list[tuple[int, int]] = []
    adjacent_pair_colors: list[int] = []
    directed_forward_groups: dict[int, list[tuple[int, int, int, int]]] = {}
    directed_reverse_groups: dict[int, list[tuple[int, int, int, int]]] = {}
    generated_runs: list[ScaffoldCoreGeneratedRun] = []
    for chain_id in torch.unique(asym_id[ca_tokens], sorted=True).tolist():
        select = asym_id[ca_tokens] == int(chain_id)
        selected_atoms = ca_atom_indices[select]
        selected_tokens = ca_tokens[select]
        protein = is_protein[selected_tokens]
        selected_atoms = selected_atoms[protein]
        selected_tokens = selected_tokens[protein]
        if len(selected_atoms) < 2:
            continue
        selected_generated = token_generated[selected_tokens]
        consecutive = (
            residue_index[selected_tokens][1:] - residue_index[selected_tokens][:-1]
        ) == 1
        consecutive_indices = torch.nonzero(
            consecutive,
            as_tuple=False,
        ).reshape(-1)
        for pair_index in consecutive_indices.tolist():
            left_token = int(selected_tokens[pair_index].item())
            right_token = int(selected_tokens[pair_index + 1].item())
            if bool(
                token_generated[left_token].item()
                or token_generated[right_token].item()
            ):
                adjacent_token_pairs.append((left_token, right_token))
                adjacent_ca_atom_pairs.append(
                    (
                        int(selected_atoms[pair_index].item()),
                        int(selected_atoms[pair_index + 1].item()),
                    )
                )
                adjacent_pair_colors.append(int(residue_index[left_token].item()) % 2)
        # Terminal generated runs have an unambiguous polymer anchor.  Record
        # a breadth/depth schedule so one vectorized forward (or reverse)
        # sweep can propagate that anchor through every symmetry copy.  This
        # avoids hundreds of slow Laplacian iterations for a 180-A shell.
        generated_values = selected_generated.tolist()
        run_start = 0
        while run_start < len(generated_values):
            if not generated_values[run_start]:
                run_start += 1
                continue
            run_end = run_start
            while run_end + 1 < len(generated_values) and generated_values[run_end + 1]:
                run_end += 1
            left_fixed = run_start > 0 and not generated_values[run_start - 1]
            right_fixed = (
                run_end + 1 < len(generated_values)
                and not generated_values[run_end + 1]
            )
            if left_fixed:
                for depth, pair_index in enumerate(range(run_start - 1, run_end)):
                    directed_forward_groups.setdefault(depth, []).append(
                        (
                            int(selected_atoms[pair_index].item()),
                            int(selected_atoms[pair_index + 1].item()),
                            int(selected_tokens[pair_index + 1].item()),
                            -1,
                        )
                    )
            if right_fixed:
                for depth, pair_index in enumerate(range(run_end, run_start - 1, -1)):
                    directed_reverse_groups.setdefault(depth, []).append(
                        (
                            int(selected_atoms[pair_index].item()),
                            int(selected_atoms[pair_index + 1].item()),
                            int(selected_tokens[pair_index].item()),
                            1,
                        )
                    )
            if left_fixed and right_fixed:
                generated_runs.append(
                    ScaffoldCoreGeneratedRun(
                        asym_id=int(chain_id),
                        generated_ca_atom_indices=selected_atoms[
                            run_start : run_end + 1
                        ],
                        generated_token_indices=selected_tokens[
                            run_start : run_end + 1
                        ],
                        left_anchor_ca_atom_index=int(
                            selected_atoms[run_start - 1].item()
                        ),
                        right_anchor_ca_atom_index=int(
                            selected_atoms[run_end + 1].item()
                        ),
                    )
                )
            run_start = run_end + 1
        chains.append(
            ScaffoldCoreChain(
                asym_id=int(chain_id),
                ca_atom_indices=selected_atoms,
                residue_indices=residue_index[selected_tokens],
                generated_ca_mask=token_generated[selected_tokens],
                ca_segment_atom_pairs=torch.stack(
                    (
                        selected_atoms[consecutive_indices],
                        selected_atoms[consecutive_indices + 1],
                    ),
                    dim=-1,
                ).reshape(-1, 2),
                generated_segment_mask=(
                    token_generated[selected_tokens[consecutive_indices]]
                    | token_generated[selected_tokens[consecutive_indices + 1]]
                ),
            )
        )
    if not chains:
        raise ValueError("Scaffold core guidance found no protein CA chains")
    if not any(torch.any(chain.generated_ca_mask) for chain in chains):
        raise ValueError("Scaffold core guidance found no generated protein tokens")
    return ScaffoldCoreTopology(
        chains=tuple(chains),
        generated_runs=tuple(generated_runs),
        atom_to_token=atom_to_token,
        generated_token_mask=token_generated,
        generated_atom_mask=generated_atom_mask,
        adjacent_token_pairs=torch.tensor(
            adjacent_token_pairs,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2),
        adjacent_ca_atom_pairs=torch.tensor(
            adjacent_ca_atom_pairs,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2),
        adjacent_pair_colors=torch.tensor(
            adjacent_pair_colors,
            dtype=torch.long,
            device=device,
        ).reshape(-1),
        directed_continuity_groups=tuple(
            torch.tensor(
                group,
                dtype=torch.long,
                device=device,
            ).reshape(-1, 4)
            for group in (
                *(
                    directed_forward_groups[depth]
                    for depth in sorted(directed_forward_groups)
                ),
                *(
                    directed_reverse_groups[depth]
                    for depth in sorted(directed_reverse_groups)
                ),
            )
        ),
    )


def project_generated_polymer_continuity(
    coordinates: torch.Tensor,
    topology: ScaffoldCoreTopology,
    *,
    target_ca_distance: float = 3.8,
    tolerance: float = 0.5,
    iterations: int = 64,
    relaxation: float = 1.0,
    projector: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project generated protein tokens onto adjacent-CA geometry.

    This is a kinematic safety projection, not a packing score.  Fixed tokens
    never move.  Generated tokens receive rigid translations (all atoms in a
    token move together), and a final Mosaic projector restores exact motif
    and symmetry constraints.  Directed anchor sweeps plus bounded constraint
    iterations propagate a fixed boundary through an arbitrarily long
    generated run without prescribing compactness, pore size, or interface
    shape.
    """

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("Polymer continuity projection requires [D, L, 3]")
    if target_ca_distance <= 0.0 or tolerance < 0.0:
        raise ValueError("Polymer continuity distances must be positive")
    if iterations < 1:
        raise ValueError("Polymer continuity iterations must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("Polymer continuity relaxation must be in (0, 1]")
    if not len(topology.adjacent_token_pairs):
        return coordinates, {
            "applied": False,
            "pair_count": 0,
            "iterations": 0,
            "maximum_initial_ca_error": 0.0,
            "maximum_final_ca_error": 0.0,
            "within_tolerance": True,
        }

    token_pairs = topology.adjacent_token_pairs
    atom_pairs = topology.adjacent_ca_atom_pairs
    generated = topology.generated_token_mask
    atom_to_token = topology.atom_to_token
    result = coordinates.detach().clone()

    def errors(value: torch.Tensor) -> torch.Tensor:
        vectors = value[:, atom_pairs[:, 1]] - value[:, atom_pairs[:, 0]]
        return torch.abs(torch.linalg.vector_norm(vectors, dim=-1) - target_ca_distance)

    initial_errors = errors(result)
    applied_iterations = 0

    # First propagate every unambiguous terminal anchor through its generated
    # run.  A group contains at most one operation per chain, so assignments
    # are collision-free and remain fully vectorized across symmetry copies.
    directed_sweeps = 4 if topology.directed_continuity_groups else 0
    for _ in range(directed_sweeps):
        for group in topology.directed_continuity_groups:
            left_atoms = group[:, 0]
            right_atoms = group[:, 1]
            moving_tokens = group[:, 2]
            signs = group[:, 3].to(dtype=result.dtype)
            vectors = result[:, right_atoms] - result[:, left_atoms]
            distances = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1e-8)
            unit = vectors / distances[..., None]
            correction = (
                relaxation
                * (distances - target_ca_distance)[..., None]
                * unit
                * signs[None, :, None]
            )
            for batch_index in range(result.shape[0]):
                token_delta = torch.zeros(
                    (len(generated), 3),
                    dtype=result.dtype,
                    device=result.device,
                )
                token_delta[moving_tokens] = correction[batch_index]
                result[batch_index] += token_delta[atom_to_token]

    for iteration in range(iterations):
        if torch.all(errors(result) <= tolerance):
            break
        # Adjacent constraints are two-coloured by residue parity.  No token
        # occurs twice within one colour, so each half-sweep is vectorized
        # Gauss-Seidel rather than a slowly diffusing Jacobi average.  This is
        # important when an I-symmetry shell starts hundreds of Angstroms from
        # the global origin.
        for color in (0, 1):
            selected = topology.adjacent_pair_colors == color
            selected_tokens = token_pairs[selected]
            selected_atoms = atom_pairs[selected]
            if not len(selected_tokens):
                continue
            vectors = result[:, selected_atoms[:, 1]] - result[:, selected_atoms[:, 0]]
            distances = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1e-8)
            violation = torch.abs(distances - target_ca_distance) > tolerance
            unit = vectors / distances[..., None]
            correction = relaxation * (distances - target_ca_distance)[..., None] * unit
            left = selected_tokens[:, 0]
            right = selected_tokens[:, 1]
            left_generated = generated[left]
            right_generated = generated[right]
            for batch_index in range(result.shape[0]):
                token_delta = torch.zeros(
                    (len(generated), 3),
                    dtype=result.dtype,
                    device=result.device,
                )
                active = violation[batch_index]
                both = active & left_generated & right_generated
                left_only = active & left_generated & ~right_generated
                right_only = active & ~left_generated & right_generated
                value = correction[batch_index]
                token_delta[left[both]] = 0.5 * value[both]
                token_delta[right[both]] = -0.5 * value[both]
                token_delta[left[left_only]] = value[left_only]
                token_delta[right[right_only]] = -value[right_only]
                result[batch_index] += token_delta[atom_to_token]
        applied_iterations = iteration + 1

    if projector is not None:
        result = projector(result)
    final_errors = errors(result)
    maximum_initial = float(initial_errors.max().detach().cpu().item())
    maximum_final = float(final_errors.max().detach().cpu().item())
    return result.detach(), {
        "applied": bool(topology.directed_continuity_groups) or applied_iterations > 0,
        "pair_count": int(len(token_pairs)),
        "directed_group_count": len(topology.directed_continuity_groups),
        "directed_sweeps": directed_sweeps,
        "iterations": applied_iterations,
        "maximum_initial_ca_error": maximum_initial,
        "maximum_final_ca_error": maximum_final,
        "within_tolerance": maximum_final <= tolerance + 1e-6,
    }


def _soft_contacts(distances: torch.Tensor, config: ScaffoldCoreGuidanceConfig):
    return torch.sigmoid(
        (config.contact_distance - distances) / config.contact_softness
    )


def _clamp_step(vector: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    return vector * torch.clamp(
        torch.as_tensor(
            maximum_norm,
            dtype=vector.dtype,
            device=vector.device,
        )
        / torch.clamp(norm, min=1e-8),
        max=1.0,
    )


def _point_to_segment_distances(
    points: torch.Tensor,
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
) -> torch.Tensor:
    """Return all pairwise point-to-segment distances.

    ``points`` has shape ``[P, 3]`` and both segment tensors have shape
    ``[S, 3]``.  The result has shape ``[P, S]``.
    """

    direction = segment_end - segment_start
    offset = points[:, None, :] - segment_start[None, :, :]
    denominator = torch.sum(torch.square(direction), dim=-1).clamp_min(1e-8)
    fraction = torch.sum(offset * direction[None, :, :], dim=-1) / denominator
    fraction = torch.clamp(fraction, min=0.0, max=1.0)
    closest = segment_start[None, :, :] + fraction[..., None] * direction[None, :, :]
    return torch.linalg.vector_norm(points[:, None, :] - closest, dim=-1)


def _segment_to_segment_distances(
    left_start: torch.Tensor,
    left_end: torch.Tensor,
    right_start: torch.Tensor,
    right_end: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable pairwise finite-segment distances.

    Endpoint-to-segment candidates cover boundary optima.  A fifth candidate
    covers an interior/interior closest approach, including the important case
    where two backbone chords cross even though none of their CA endpoints is
    atomically close.  Near-parallel segments safely fall back to the endpoint
    candidates.
    """

    left_start_to_right = _point_to_segment_distances(
        left_start,
        right_start,
        right_end,
    )
    left_end_to_right = _point_to_segment_distances(
        left_end,
        right_start,
        right_end,
    )
    right_start_to_left = _point_to_segment_distances(
        right_start,
        left_start,
        left_end,
    ).transpose(0, 1)
    right_end_to_left = _point_to_segment_distances(
        right_end,
        left_start,
        left_end,
    ).transpose(0, 1)

    left_direction = left_end - left_start
    right_direction = right_end - right_start
    relative = left_start[:, None, :] - right_start[None, :, :]
    a = torch.sum(torch.square(left_direction), dim=-1)[:, None]
    b = torch.sum(
        left_direction[:, None, :] * right_direction[None, :, :],
        dim=-1,
    )
    c = torch.sum(torch.square(right_direction), dim=-1)[None, :]
    d = torch.sum(left_direction[:, None, :] * relative, dim=-1)
    e = torch.sum(right_direction[None, :, :] * relative, dim=-1)
    determinant = a * c - torch.square(b)
    safe_determinant = determinant.clamp_min(1e-8)
    left_fraction = (b * e - c * d) / safe_determinant
    right_fraction = (a * e - b * d) / safe_determinant
    interior_valid = (
        (determinant > 1e-8)
        & (left_fraction >= 0.0)
        & (left_fraction <= 1.0)
        & (right_fraction >= 0.0)
        & (right_fraction <= 1.0)
    )
    left_closest = (
        left_start[:, None, :] + left_fraction[..., None] * left_direction[:, None, :]
    )
    right_closest = (
        right_start[None, :, :]
        + right_fraction[..., None] * right_direction[None, :, :]
    )
    interior_distance = torch.linalg.vector_norm(
        left_closest - right_closest,
        dim=-1,
    )
    infinity = torch.full_like(interior_distance, float("inf"))
    interior_distance = torch.where(
        interior_valid,
        interior_distance,
        infinity,
    )
    return (
        torch.stack(
            (
                left_start_to_right,
                left_end_to_right,
                right_start_to_left,
                right_end_to_left,
                interior_distance,
            ),
            dim=0,
        )
        .min(dim=0)
        .values
    )


def scaffold_core_energy(
    coordinates: torch.Tensor,
    topology: ScaffoldCoreTopology,
    config: ScaffoldCoreGuidanceConfig,
) -> ScaffoldCoreEnergy:
    """Evaluate the supported-monomer objective on one atom-level state."""

    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [L, 3]")
    zero = coordinates.sum() * 0.0
    long_range_terms: list[torch.Tensor] = []
    rg_terms: list[torch.Tensor] = []
    support_terms: list[torch.Tensor] = []
    worst_support_terms: list[torch.Tensor] = []
    normalized_rgs: list[torch.Tensor] = []
    support_fractions: list[torch.Tensor] = []
    continuity_terms: list[torch.Tensor] = []
    clash_terms: list[torch.Tensor] = []
    cross_chain_segment_clash_terms: list[torch.Tensor] = []
    cross_chain_segment_minimums: list[torch.Tensor] = []

    for chain in topology.chains:
        xyz = coordinates[chain.ca_atom_indices]
        count = len(xyz)
        generated = chain.generated_ca_mask
        generated_count = torch.clamp(generated.sum(), min=1).to(xyz.dtype)
        center = xyz.mean(dim=0)
        rg = torch.sqrt(
            torch.mean(torch.sum(torch.square(xyz - center), dim=-1)) + 1e-8
        )
        normalized = rg / (float(count) ** 0.38)
        normalized_rgs.append(normalized)
        rg_terms.append(
            torch.square(torch.relu(normalized - config.target_normalized_rg))
        )

        distances = torch.cdist(xyz, xyz)
        sequence_gap = torch.abs(
            chain.residue_indices[:, None] - chain.residue_indices[None, :]
        )
        upper = torch.triu(
            torch.ones((count, count), dtype=torch.bool, device=xyz.device),
            diagonal=1,
        )
        pair_mask = (
            upper
            & (sequence_gap >= config.sequence_separation)
            & (generated[:, None] | generated[None, :])
        )
        left, right = torch.nonzero(pair_mask, as_tuple=True)
        support = torch.zeros(count, dtype=xyz.dtype, device=xyz.device)
        if len(left):
            contacts = _soft_contacts(distances[left, right], config)
            support = support.index_add(0, left, contacts)
            support = support.index_add(0, right, contacts)
            contacts_per_generated = contacts.sum() / generated_count
        else:
            contacts_per_generated = zero
        long_range_terms.append(
            torch.square(
                torch.relu(
                    torch.as_tensor(
                        config.target_contacts_per_generated_residue,
                        dtype=xyz.dtype,
                        device=xyz.device,
                    )
                    - contacts_per_generated
                )
            )
        )
        generated_support = support[generated]
        if len(generated_support):
            support_deficit = torch.square(
                torch.relu(config.target_supported_contacts - generated_support)
            )
            support_terms.append(torch.mean(support_deficit))
            # A chain-wide mean can hide one long unsupported arm behind a
            # well packed local core.  Average over a sequence-local window,
            # then use a normalized smooth maximum to focus the gradient on
            # the worst contiguous generated region without introducing a
            # hard pass/fail cutoff.  Reusing ``sequence_separation`` ties the
            # window to the same definition of a tertiary contact.
            worst_support_terms.append(
                worst_support_deficit_energy(generated_support, config)
            )
            support_fractions.append(
                torch.mean(
                    (generated_support >= config.target_supported_contacts).to(
                        xyz.dtype
                    )
                )
            )

        if count > 1:
            adjacent = sequence_gap == 1
            adjacent = adjacent & torch.triu(torch.ones_like(adjacent), diagonal=1)
            adjacent = adjacent & (generated[:, None] | generated[None, :])
            a_left, a_right = torch.nonzero(adjacent, as_tuple=True)
            if len(a_left):
                errors = torch.relu(
                    torch.abs(distances[a_left, a_right] - config.backbone_distance)
                    - config.backbone_tolerance
                )
                continuity_terms.append(torch.mean(torch.square(errors)))
        clash_mask = (
            upper & (sequence_gap > 1) & (generated[:, None] | generated[None, :])
        )
        c_left, c_right = torch.nonzero(clash_mask, as_tuple=True)
        if len(c_left):
            clash_terms.append(
                torch.mean(
                    torch.square(
                        torch.relu(config.clash_distance - distances[c_left, c_right])
                    )
                )
            )

    inter_terms: list[torch.Tensor] = []
    inter_pairs_hard: list[torch.Tensor] = []
    inter_coverages: list[torch.Tensor] = []
    inter_minimums: list[torch.Tensor] = []
    for left_index, left_chain in enumerate(topology.chains):
        left_atoms = left_chain.ca_atom_indices[left_chain.generated_ca_mask]
        for right_chain in topology.chains[left_index + 1 :]:
            right_atoms = right_chain.ca_atom_indices[right_chain.generated_ca_mask]
            if len(left_atoms) and len(right_atoms):
                distances = torch.cdist(
                    coordinates[left_atoms],
                    coordinates[right_atoms],
                )
                soft = _soft_contacts(distances, config)
                scale = float(min(len(left_atoms), len(right_atoms)))
                allowance = config.incidental_inter_chain_fraction * scale
                excess = torch.relu(soft.sum() - allowance) / max(scale, 1.0)
                inter_terms.append(torch.square(excess))
                hard = distances < config.contact_distance
                inter_pairs_hard.append(hard.sum().to(coordinates.dtype))
                left_covered = hard.any(dim=1).to(coordinates.dtype).mean()
                right_covered = hard.any(dim=0).to(coordinates.dtype).mean()
                inter_coverages.append(0.5 * (left_covered + right_covered))
                inter_minimums.append(distances.min())

                # Hard safety rejects atomically overlapping generated chains,
                # while ordinary interface-distance contacts remain soft.
                clash_terms.append(
                    torch.mean(
                        torch.square(torch.relu(config.clash_distance - distances))
                    )
                )

            left_segments = left_chain.ca_segment_atom_pairs
            right_segments = right_chain.ca_segment_atom_pairs
            if len(left_segments) and len(right_segments):
                left_generated = left_chain.generated_segment_mask
                right_generated = right_chain.generated_segment_mask
                relevant = left_generated[:, None] | right_generated[None, :]
                if torch.any(relevant):
                    segment_distances = _segment_to_segment_distances(
                        coordinates[left_segments[:, 0]],
                        coordinates[left_segments[:, 1]],
                        coordinates[right_segments[:, 0]],
                        coordinates[right_segments[:, 1]],
                    )[relevant]
                    cross_chain_segment_minimums.append(segment_distances.min())
                    cross_chain_segment_clash_terms.append(
                        torch.max(
                            torch.square(
                                torch.relu(config.clash_distance - segment_distances)
                            )
                        )
                    )

    # A two-anchored generated run owns the Voronoi cell of its compiler-
    # declared endpoint chord.  This is a relative routing constraint: it
    # does not pull a backbone onto the straight chord and it does not prefer
    # inward over outward curvature.  It only penalizes residues that are
    # closer to another chain's endpoint corridor than to their own.
    routing_terms: list[torch.Tensor] = []
    routing_excesses: list[torch.Tensor] = []
    for run_index, run in enumerate(topology.generated_runs):
        competitors = [
            other
            for other_index, other in enumerate(topology.generated_runs)
            if other_index != run_index and other.asym_id != run.asym_id
        ]
        if not competitors or not len(run.generated_ca_atom_indices):
            continue
        points = coordinates[run.generated_ca_atom_indices]
        own = _point_to_segment_distances(
            points,
            coordinates[run.left_anchor_ca_atom_index][None, :],
            coordinates[run.right_anchor_ca_atom_index][None, :],
        )[:, 0]
        other_start = torch.stack(
            [coordinates[item.left_anchor_ca_atom_index] for item in competitors]
        )
        other_end = torch.stack(
            [coordinates[item.right_anchor_ca_atom_index] for item in competitors]
        )
        other = (
            _point_to_segment_distances(
                points,
                other_start,
                other_end,
            )
            .min(dim=1)
            .values
        )
        excess = torch.relu(own - other) / config.backbone_distance
        routing_excesses.append(excess)
        routing_terms.append(torch.mean(torch.square(excess)))

    def mean(items: list[torch.Tensor], default: torch.Tensor = zero):
        return torch.stack(items).mean() if items else default

    long_range = mean(long_range_terms)
    normalized_rg = mean(rg_terms)
    tertiary_support = mean(support_terms)
    worst_support = mean(worst_support_terms)
    inter_excess = mean(inter_terms)
    clash = mean(clash_terms)
    cross_chain_segment_clash = mean(cross_chain_segment_clash_terms)
    continuity = mean(continuity_terms)
    routing_ownership = mean(routing_terms)
    total = (
        config.intra_chain_weight
        * (
            config.long_range_contact_weight * long_range
            + config.normalized_rg_weight * normalized_rg
            + config.tertiary_support_weight * tertiary_support
            + config.worst_support_weight * worst_support
        )
        + config.inter_chain_excess_penalty
        * config.inter_chain_excess_weight
        * inter_excess
        + config.clash_weight * (clash + cross_chain_segment_clash)
        + config.continuity_weight * continuity
        + config.routing_ownership_weight * routing_ownership
    )
    inf = torch.full(
        (), float("inf"), dtype=coordinates.dtype, device=coordinates.device
    )
    all_routing_excess = (
        torch.cat(routing_excesses)
        if routing_excesses
        else torch.zeros(1, dtype=coordinates.dtype, device=coordinates.device)
    )
    return ScaffoldCoreEnergy(
        total=total,
        long_range_contacts=long_range,
        normalized_rg=normalized_rg,
        tertiary_support=tertiary_support,
        worst_support=worst_support,
        inter_chain_excess=inter_excess,
        clash=clash,
        cross_chain_segment_clash=cross_chain_segment_clash,
        continuity=continuity,
        routing_ownership=routing_ownership,
        mean_normalized_rg=mean(normalized_rgs),
        mean_tertiary_support_fraction=mean(support_fractions),
        generated_inter_chain_contact_pairs=torch.stack(inter_pairs_hard).sum()
        if inter_pairs_hard
        else zero,
        generated_inter_chain_contact_coverage=mean(inter_coverages),
        minimum_generated_inter_chain_distance=torch.stack(inter_minimums).min()
        if inter_minimums
        else inf,
        minimum_cross_chain_segment_distance=torch.stack(
            cross_chain_segment_minimums
        ).min()
        if cross_chain_segment_minimums
        else inf,
        routing_ownership_violation_fraction=torch.mean(
            (all_routing_excess > 0.0).to(coordinates.dtype)
        ),
        maximum_routing_ownership_excess=all_routing_excess.max(),
    )


def scaffold_core_window(progress: float, config: ScaffoldCoreGuidanceConfig) -> float:
    if progress < config.start_fraction or progress > config.end_fraction:
        return 0.0
    span = config.end_fraction - config.start_fraction
    local = (progress - config.start_fraction) / max(span, 1e-8)
    return float(min(1.0, 4.0 * local, 4.0 * (1.0 - local)))


def apply_scaffold_core_guidance(
    coordinates: torch.Tensor,
    topology: ScaffoldCoreTopology,
    *,
    progress: float,
    config: ScaffoldCoreGuidanceConfig,
    projector: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one bounded generated-token proposal with line-search rollback."""

    if coordinates.ndim != 3 or coordinates.shape[0] != 1:
        raise ValueError("Scaffold core guidance requires coordinates [1, L, 3]")
    window = scaffold_core_window(progress, config)
    with torch.enable_grad():
        source = coordinates.detach().clone().requires_grad_(True)
        initial = scaffold_core_energy(source[0], topology, config)
        if window <= 0.0 or float(initial.total.detach().item()) == 0.0:
            return coordinates, {
                "applied": False,
                "accepted": False,
                "progress": float(progress),
                "window": float(window),
                "initial": initial.detached_dict(),
                "final": initial.detached_dict(),
            }
        initial.total.backward()
        gradient = source.grad[0]
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError("Scaffold core guidance produced non-finite gradients")
        ca_gradient_by_token = torch.zeros(
            (len(topology.generated_token_mask), 3),
            dtype=gradient.dtype,
            device=gradient.device,
        )
        ca_count_by_token = torch.zeros(
            len(topology.generated_token_mask),
            dtype=gradient.dtype,
            device=gradient.device,
        )
        for chain in topology.chains:
            atoms = chain.ca_atom_indices[chain.generated_ca_mask]
            if not len(atoms):
                continue
            tokens = topology.atom_to_token[atoms]
            ca_gradient_by_token.index_add_(0, tokens, gradient[atoms])
            ca_count_by_token.index_add_(
                0,
                tokens,
                torch.ones(len(tokens), dtype=gradient.dtype, device=gradient.device),
            )
        ca_gradient_by_token = ca_gradient_by_token / torch.clamp(
            ca_count_by_token[:, None], min=1.0
        )
        token_step = -ca_gradient_by_token
        norms = torch.linalg.vector_norm(token_step, dim=-1, keepdim=True)
        token_step = token_step * torch.clamp(
            config.maximum_token_step * window / torch.clamp(norms, min=1e-8),
            max=1.0,
        )
        token_step[~topology.generated_token_mask] = 0.0
        # Adjacent residues cannot receive unrelated translations: their
        # difference is exactly the perturbation applied to every inter-token
        # peptide-bond vector.  Smooth first, then enforce a hard per-step
        # bound while keeping fixed tokens immobile.
        for _ in range(2):
            if not len(topology.adjacent_token_pairs):
                break
            accumulated = token_step.clone()
            counts = torch.ones(
                len(token_step),
                dtype=token_step.dtype,
                device=token_step.device,
            )
            left = topology.adjacent_token_pairs[:, 0]
            right = topology.adjacent_token_pairs[:, 1]
            accumulated.index_add_(0, left, token_step[right])
            accumulated.index_add_(0, right, token_step[left])
            counts.index_add_(0, left, torch.ones_like(left, dtype=token_step.dtype))
            counts.index_add_(0, right, torch.ones_like(right, dtype=token_step.dtype))
            token_step = 0.5 * token_step + 0.5 * (accumulated / counts[:, None])
            token_step[~topology.generated_token_mask] = 0.0
        maximum_adjacent_difference = 0.0
        if len(topology.adjacent_token_pairs):
            for _ in range(3):
                for left_value, right_value in topology.adjacent_token_pairs.tolist():
                    left_index = int(left_value)
                    right_index = int(right_value)
                    left_generated = bool(
                        topology.generated_token_mask[left_index].item()
                    )
                    right_generated = bool(
                        topology.generated_token_mask[right_index].item()
                    )
                    if left_generated and right_generated:
                        midpoint = 0.5 * (
                            token_step[left_index] + token_step[right_index]
                        )
                        difference = _clamp_step(
                            token_step[left_index] - token_step[right_index],
                            config.maximum_adjacent_token_step_difference,
                        )
                        token_step[left_index] = midpoint + 0.5 * difference
                        token_step[right_index] = midpoint - 0.5 * difference
                    elif left_generated:
                        token_step[left_index] = _clamp_step(
                            token_step[left_index],
                            config.maximum_adjacent_token_step_difference,
                        )
                    elif right_generated:
                        token_step[right_index] = _clamp_step(
                            token_step[right_index],
                            config.maximum_adjacent_token_step_difference,
                        )
                token_step[~topology.generated_token_mask] = 0.0
            left = topology.adjacent_token_pairs[:, 0]
            right = topology.adjacent_token_pairs[:, 1]
            maximum_adjacent_difference = float(
                torch.linalg.vector_norm(
                    token_step[left] - token_step[right],
                    dim=-1,
                )
                .max()
                .detach()
                .cpu()
                .item()
            )
        atom_step = token_step[topology.atom_to_token][None, ...]

    accepted = False
    result = coordinates
    final = initial
    accepted_scale = 0.0
    for attempt in range(config.line_search_steps):
        scale = config.line_search_contraction**attempt
        candidate = coordinates + scale * atom_step.detach()
        if projector is not None:
            candidate = projector(candidate)
        with torch.no_grad():
            trial = scaffold_core_energy(candidate[0], topology, config)
        if not torch.isfinite(trial.total):
            continue
        safety_ok = (
            float(trial.clash.item()) <= float(initial.clash.item()) + 1e-7
            and float(trial.cross_chain_segment_clash.item())
            <= float(initial.cross_chain_segment_clash.item()) + 1e-7
            and float(trial.continuity.item())
            <= float(initial.continuity.item()) + 1e-7
            and (
                config.routing_ownership_weight <= 0.0
                or float(trial.routing_ownership.item())
                <= float(initial.routing_ownership.item()) + 1e-7
            )
        )
        if safety_ok and float(trial.total.item()) < float(initial.total.item()) - 1e-8:
            result = candidate.detach()
            final = trial
            accepted = True
            accepted_scale = float(scale)
            break
    return result, {
        "applied": accepted,
        "accepted": accepted,
        "progress": float(progress),
        "window": float(window),
        "line_search_scale": accepted_scale,
        "maximum_token_step": float(
            torch.linalg.vector_norm(token_step, dim=-1).max().detach().cpu().item()
        ),
        "maximum_adjacent_token_step_difference": (maximum_adjacent_difference),
        "initial": initial.detached_dict(),
        "final": final.detached_dict(),
    }


__all__ = [
    "ScaffoldCoreEnergy",
    "ScaffoldCoreGuidanceConfig",
    "ScaffoldCoreTopology",
    "apply_scaffold_core_guidance",
    "build_scaffold_core_topology",
    "project_generated_polymer_continuity",
    "scaffold_core_energy",
    "scaffold_core_window",
    "worst_support_deficit_energy",
]
