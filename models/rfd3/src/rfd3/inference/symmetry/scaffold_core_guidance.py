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


@dataclass(frozen=True)
class ScaffoldCoreTopology:
    chains: tuple[ScaffoldCoreChain, ...]
    atom_to_token: torch.Tensor
    generated_token_mask: torch.Tensor
    generated_atom_mask: torch.Tensor
    adjacent_token_pairs: torch.Tensor


@dataclass(frozen=True)
class ScaffoldCoreGuidanceConfig:
    intra_chain_weight: float = 0.0
    inter_chain_weight: float = 1.0
    inter_chain_excess_penalty: float = 0.0
    long_range_contact_weight: float = 0.75
    normalized_rg_weight: float = 0.35
    tertiary_support_weight: float = 1.0
    inter_chain_excess_weight: float = 1.0
    clash_weight: float = 8.0
    continuity_weight: float = 2.0
    contact_distance: float = 8.0
    contact_softness: float = 0.75
    sequence_separation: int = 8
    target_contacts_per_generated_residue: float = 1.5
    target_supported_contacts: float = 2.0
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
            "inter_chain_excess_weight",
            "clash_weight",
            "continuity_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "contact_distance",
            "contact_softness",
            "target_contacts_per_generated_residue",
            "target_supported_contacts",
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
    inter_chain_excess: torch.Tensor
    clash: torch.Tensor
    continuity: torch.Tensor
    mean_normalized_rg: torch.Tensor
    mean_tertiary_support_fraction: torch.Tensor
    generated_inter_chain_contact_pairs: torch.Tensor
    generated_inter_chain_contact_coverage: torch.Tensor
    minimum_generated_inter_chain_distance: torch.Tensor

    def detached_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in (
                "total",
                "long_range_contacts",
                "normalized_rg",
                "tertiary_support",
                "inter_chain_excess",
                "clash",
                "continuity",
                "mean_normalized_rg",
                "mean_tertiary_support_fraction",
                "generated_inter_chain_contact_pairs",
                "generated_inter_chain_contact_coverage",
                "minimum_generated_inter_chain_distance",
            )
        }


def _tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


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
    for chain_id in torch.unique(asym_id[ca_tokens], sorted=True).tolist():
        select = asym_id[ca_tokens] == int(chain_id)
        selected_atoms = ca_atom_indices[select]
        selected_tokens = ca_tokens[select]
        protein = is_protein[selected_tokens]
        selected_atoms = selected_atoms[protein]
        selected_tokens = selected_tokens[protein]
        if len(selected_atoms) < 2:
            continue
        consecutive = (
            residue_index[selected_tokens][1:] - residue_index[selected_tokens][:-1]
        ) == 1
        for pair_index in (
            torch.nonzero(
                consecutive,
                as_tuple=False,
            )
            .reshape(-1)
            .tolist()
        ):
            left_token = int(selected_tokens[pair_index].item())
            right_token = int(selected_tokens[pair_index + 1].item())
            if bool(
                token_generated[left_token].item()
                or token_generated[right_token].item()
            ):
                adjacent_token_pairs.append((left_token, right_token))
        chains.append(
            ScaffoldCoreChain(
                asym_id=int(chain_id),
                ca_atom_indices=selected_atoms,
                residue_indices=residue_index[selected_tokens],
                generated_ca_mask=token_generated[selected_tokens],
            )
        )
    if not chains:
        raise ValueError("Scaffold core guidance found no protein CA chains")
    if not any(torch.any(chain.generated_ca_mask) for chain in chains):
        raise ValueError("Scaffold core guidance found no generated protein tokens")
    return ScaffoldCoreTopology(
        chains=tuple(chains),
        atom_to_token=atom_to_token,
        generated_token_mask=token_generated,
        generated_atom_mask=generated_atom_mask,
        adjacent_token_pairs=torch.tensor(
            adjacent_token_pairs,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2),
    )


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
    normalized_rgs: list[torch.Tensor] = []
    support_fractions: list[torch.Tensor] = []
    continuity_terms: list[torch.Tensor] = []
    clash_terms: list[torch.Tensor] = []

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
            support_terms.append(
                torch.mean(
                    torch.square(
                        torch.relu(config.target_supported_contacts - generated_support)
                    )
                )
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
        if not len(left_atoms):
            continue
        for right_chain in topology.chains[left_index + 1 :]:
            right_atoms = right_chain.ca_atom_indices[right_chain.generated_ca_mask]
            if not len(right_atoms):
                continue
            distances = torch.cdist(coordinates[left_atoms], coordinates[right_atoms])
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
                torch.mean(torch.square(torch.relu(config.clash_distance - distances)))
            )

    def mean(items: list[torch.Tensor], default: torch.Tensor = zero):
        return torch.stack(items).mean() if items else default

    long_range = mean(long_range_terms)
    normalized_rg = mean(rg_terms)
    tertiary_support = mean(support_terms)
    inter_excess = mean(inter_terms)
    clash = mean(clash_terms)
    continuity = mean(continuity_terms)
    total = (
        config.intra_chain_weight
        * (
            config.long_range_contact_weight * long_range
            + config.normalized_rg_weight * normalized_rg
            + config.tertiary_support_weight * tertiary_support
        )
        + config.inter_chain_excess_penalty
        * config.inter_chain_excess_weight
        * inter_excess
        + config.clash_weight * clash
        + config.continuity_weight * continuity
    )
    inf = torch.full(
        (), float("inf"), dtype=coordinates.dtype, device=coordinates.device
    )
    return ScaffoldCoreEnergy(
        total=total,
        long_range_contacts=long_range,
        normalized_rg=normalized_rg,
        tertiary_support=tertiary_support,
        inter_chain_excess=inter_excess,
        clash=clash,
        continuity=continuity,
        mean_normalized_rg=mean(normalized_rgs),
        mean_tertiary_support_fraction=mean(support_fractions),
        generated_inter_chain_contact_pairs=torch.stack(inter_pairs_hard).sum()
        if inter_pairs_hard
        else zero,
        generated_inter_chain_contact_coverage=mean(inter_coverages),
        minimum_generated_inter_chain_distance=torch.stack(inter_minimums).min()
        if inter_minimums
        else inf,
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
            and float(trial.continuity.item())
            <= float(initial.continuity.item()) + 1e-7
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
    "scaffold_core_energy",
    "scaffold_core_window",
]
