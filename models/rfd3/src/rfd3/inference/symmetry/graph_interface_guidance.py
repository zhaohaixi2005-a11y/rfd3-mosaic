"""Symmetry-coupled soft guidance for graph-declared design interfaces.

The fixed-interface and designed-interface cases share one assembly graph and
one sampler.  Input-stage relations are handled by the exact constraint
projector.  Output-stage contact relations are converted here into a soft
field over generated residues.  Every symmetry-expanded edge contributes to
one joint energy, so guidance cannot move one copy independently or collapse
unrelated protomers through an all-to-all compactness force.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class GraphInterfaceGuidanceConfig:
    """Trust region and schedule for generated-interface guidance."""

    weight: float = 1.0
    coverage_weight: float = 1.0
    continuity_weight: float = 0.5
    clash_weight: float = 2.0
    distance_weight: float = 0.25
    target_ca_distance: float = 8.0
    clash_ca_distance: float = 3.5
    pairs_per_edge: int = 8
    start_fraction: float = 0.05
    end_fraction: float = 0.80
    maximum_token_step: float = 0.25
    token_smoothing_weight: float = 0.5
    token_smoothing_passes: int = 1

    def __post_init__(self) -> None:
        if (
            self.weight < 0.0
            or self.coverage_weight < 0.0
            or self.continuity_weight < 0.0
            or self.clash_weight < 0.0
            or self.distance_weight < 0.0
        ):
            raise ValueError("Interface guidance weights cannot be negative")
        if self.target_ca_distance <= self.clash_ca_distance:
            raise ValueError(
                "target_ca_distance must exceed clash_ca_distance"
            )
        if self.clash_ca_distance <= 0.0:
            raise ValueError("clash_ca_distance must be positive")
        if self.pairs_per_edge < 1:
            raise ValueError("pairs_per_edge must be positive")
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError(
                "Interface guidance requires 0 <= start < end <= 1"
            )
        if self.maximum_token_step <= 0.0:
            raise ValueError("maximum_token_step must be positive")
        if not 0.0 <= self.token_smoothing_weight <= 1.0:
            raise ValueError(
                "token_smoothing_weight must be between zero and one"
            )
        if self.token_smoothing_passes < 0:
            raise ValueError("token_smoothing_passes cannot be negative")


@dataclass(frozen=True)
class GraphInterfaceEdge:
    """One concrete symmetry-neighbour edge acting on generated CA atoms."""

    edge_id: str
    source_interface_id: str
    left_generated_ca_mask: torch.Tensor
    right_generated_ca_mask: torch.Tensor
    left_generated_token_ids: torch.Tensor
    right_generated_token_ids: torch.Tensor
    requested_contact_count: int
    requested_residues_per_side: int
    requested_contiguous_residues_per_side: int
    automatic_quality: bool
    contact_cutoff: float
    distance_target: float | None
    distance_tolerance: float | None


@dataclass(frozen=True)
class GraphInterfaceTopology:
    """Runtime-resolved output-stage interface targets."""

    edges: tuple[GraphInterfaceEdge, ...]
    generated_atom_mask: torch.Tensor


@dataclass(frozen=True)
class GraphInterfaceEnergy:
    """Joint attractive/repulsive energy over all declared edges."""

    total: torch.Tensor
    attraction: torch.Tensor
    coverage: torch.Tensor
    continuity: torch.Tensor
    clash: torch.Tensor
    distance: torch.Tensor
    minimum_distances: torch.Tensor
    mean_selected_distances: torch.Tensor
    covered_left_residues: torch.Tensor
    covered_right_residues: torch.Tensor
    target_residues_per_side: torch.Tensor
    target_contiguous_residues_per_side: torch.Tensor
    contiguous_left_residues: torch.Tensor
    contiguous_right_residues: torch.Tensor
    per_edge_total: torch.Tensor


def _as_bool_feature(
    features: dict[str, Any],
    name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(features[name], dtype=torch.bool, device=device)
    return value


def build_graph_interface_topology(
    features: dict[str, Any],
    fixed_mask: torch.Tensor,
) -> GraphInterfaceTopology | None:
    """Bind output-stage graph relations to generated atoms on each side."""

    left_value = features.get("assembly_interface_left_membership")
    if left_value is None:
        return None
    device = fixed_mask.device
    fixed = torch.as_tensor(fixed_mask, dtype=torch.bool, device=device)
    left = _as_bool_feature(
        features,
        "assembly_interface_left_membership",
        device=device,
    )
    right = _as_bool_feature(
        features,
        "assembly_interface_right_membership",
        device=device,
    )
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError(
            "Interface memberships must have matching shape [E, L]"
        )
    if left.shape[1] != fixed.numel():
        raise ValueError(
            "Interface memberships must match the atom dimension"
        )
    edge_count = left.shape[0]
    modes = torch.as_tensor(
        features["assembly_interface_mode"],
        dtype=torch.long,
        device=device,
    )
    required = _as_bool_feature(
        features,
        "assembly_interface_required",
        device=device,
    )
    minima = torch.as_tensor(
        features["assembly_interface_minimum_contacts"],
        dtype=torch.long,
        device=device,
    )
    coverage_minima = torch.as_tensor(
        features.get(
            "assembly_interface_minimum_residues_per_side",
            torch.zeros_like(minima),
        ),
        dtype=torch.long,
        device=device,
    )
    contiguous_minima = torch.as_tensor(
        features.get(
            "assembly_interface_minimum_contiguous_residues_per_side",
            torch.zeros_like(minima),
        ),
        dtype=torch.long,
        device=device,
    )
    automatic_quality = torch.as_tensor(
        features.get(
            "assembly_interface_automatic_quality",
            torch.zeros_like(required),
        ),
        dtype=torch.bool,
        device=device,
    )
    contact_cutoffs = torch.as_tensor(
        features["assembly_interface_contact_cutoff"],
        dtype=torch.float32,
        device=device,
    )
    distance_targets = torch.as_tensor(
        features["assembly_interface_distance_target"],
        dtype=torch.float32,
        device=device,
    )
    distance_tolerances = torch.as_tensor(
        features["assembly_interface_distance_tolerance"],
        dtype=torch.float32,
        device=device,
    )
    edge_ids = tuple(features["assembly_interface_ids"])
    source_interface_ids = tuple(
        features.get(
            "assembly_interface_source_ids",
            tuple(str(edge_id).split("@", 1)[0] for edge_id in edge_ids),
        )
    )
    stages = tuple(features["assembly_interface_satisfaction_stages"])
    if not (
        modes.shape
        == required.shape
        == minima.shape
        == coverage_minima.shape
        == contiguous_minima.shape
        == automatic_quality.shape
        == contact_cutoffs.shape
        == distance_targets.shape
        == distance_tolerances.shape
        == (edge_count,)
        and len(edge_ids)
        == len(source_interface_ids)
        == len(stages)
        == edge_count
    ):
        raise ValueError("Interface relation metadata has inconsistent length")

    atom_to_token = torch.as_tensor(
        features["atom_to_token_map"],
        dtype=torch.long,
        device=device,
    )
    asym_id = torch.as_tensor(
        features["asym_id"],
        dtype=torch.long,
        device=device,
    )
    is_ca = _as_bool_feature(features, "is_ca", device=device)
    if atom_to_token.shape != fixed.shape or is_ca.shape != fixed.shape:
        raise ValueError("Atom-level topology features must have shape [L]")
    atom_chain = asym_id[atom_to_token]
    is_virtual = torch.as_tensor(
        features.get("is_virtual", torch.zeros_like(fixed)),
        dtype=torch.bool,
        device=device,
    )
    generated = ~fixed & ~is_virtual
    edges = []
    used_generated = torch.zeros_like(fixed)
    for index in range(edge_count):
        # mode=0 is preserve_input and belongs to the hard constraint path.
        # mode=1 plus output stage is a diffusion-time design objective.
        if int(modes[index].item()) != 1 or stages[index] != "output":
            continue
        if not bool(required[index]):
            continue
        left_chains = torch.unique(atom_chain[left[index]])
        right_chains = torch.unique(atom_chain[right[index]])
        if torch.any(torch.isin(left_chains, right_chains)):
            raise ValueError(
                "Designed-interface guidance currently requires two "
                "distinct output chains; the declared relation resolves "
                f"both sides onto one chain: {edge_ids[index]!r}. Target "
                "a non-identity symmetry neighbour for an inter-subunit "
                "interface."
            )
        left_generated = (
            generated
            & is_ca
            & torch.isin(atom_chain, left_chains)
        )
        right_generated = (
            generated
            & is_ca
            & torch.isin(atom_chain, right_chains)
        )
        if not torch.any(left_generated) or not torch.any(right_generated):
            raise ValueError(
                "Designed interface has no generated CA atoms on both sides: "
                f"{edge_ids[index]!r}"
            )
        edges.append(
            GraphInterfaceEdge(
                edge_id=str(edge_ids[index]),
                source_interface_id=str(source_interface_ids[index]),
                left_generated_ca_mask=left_generated,
                right_generated_ca_mask=right_generated,
                left_generated_token_ids=atom_to_token[left_generated],
                right_generated_token_ids=atom_to_token[right_generated],
                requested_contact_count=max(int(minima[index].item()), 0),
                requested_residues_per_side=max(
                    int(coverage_minima[index].item()), 0
                ),
                requested_contiguous_residues_per_side=max(
                    int(contiguous_minima[index].item()), 0
                ),
                automatic_quality=bool(automatic_quality[index].item()),
                contact_cutoff=float(contact_cutoffs[index].item()),
                distance_target=(
                    float(distance_targets[index].item())
                    if torch.isfinite(distance_targets[index])
                    else None
                ),
                distance_tolerance=(
                    float(distance_tolerances[index].item())
                    if torch.isfinite(distance_tolerances[index])
                    else None
                ),
            )
        )
        used_generated |= left_generated | right_generated
    if not edges:
        return None
    # Guidance translations are token-rigid, so every atom belonging to a
    # selected generated CA token participates in the update.
    selected_tokens = torch.unique(atom_to_token[used_generated])
    generated_atom_mask = generated & torch.isin(
        atom_to_token,
        selected_tokens,
    )
    return GraphInterfaceTopology(
        edges=tuple(edges),
        generated_atom_mask=generated_atom_mask,
    )


def _contiguous_token_runs(token_ids: torch.Tensor) -> tuple[slice, ...]:
    """Return slices for monotonically adjacent token runs."""

    values = [int(value) for value in token_ids.detach().cpu().tolist()]
    if not values:
        return ()
    starts = [0]
    for index, (left, right) in enumerate(zip(values, values[1:]), start=1):
        if right != left + 1:
            starts.append(index)
    starts.append(len(values))
    return tuple(
        slice(start, stop)
        for start, stop in zip(starts, starts[1:])
    )


def _continuity_loss(
    nearest_distances: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    target_distance: float,
    target_count: int,
) -> torch.Tensor:
    """Penalize the best contiguous interface patch on one chain side."""

    excess = torch.relu(nearest_distances - target_distance)
    candidates = []
    fallback = []
    for run in _contiguous_token_runs(token_ids):
        run_excess = excess[run]
        fallback.append(run_excess.mean())
        if run_excess.numel() >= target_count:
            candidates.append(
                run_excess.unfold(0, target_count, 1).mean(dim=-1).min()
            )
    selected = candidates if candidates else fallback
    if not selected:
        return torch.zeros(
            (),
            device=nearest_distances.device,
            dtype=nearest_distances.dtype,
        )
    return torch.stack(selected).min()


def _maximum_contiguous_covered(
    nearest_distances: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    target_distance: float,
) -> torch.Tensor:
    """Measure the longest adjacent residue run already within the target."""

    maximum = 0
    covered = nearest_distances <= target_distance
    for run in _contiguous_token_runs(token_ids):
        current = 0
        for value in covered[run].detach().cpu().tolist():
            current = current + 1 if bool(value) else 0
            maximum = max(maximum, current)
    return torch.tensor(
        maximum,
        device=nearest_distances.device,
        dtype=torch.long,
    )


def _balanced_source_mean(
    values: list[torch.Tensor],
    edges: tuple[GraphInterfaceEdge, ...],
) -> torch.Tensor:
    """Give every declared interface equal weight, independent of orbit size."""

    source_ids = []
    for edge in edges:
        if edge.source_interface_id not in source_ids:
            source_ids.append(edge.source_interface_id)
    source_means = [
        torch.stack(
            [
                value
                for value, edge in zip(values, edges, strict=True)
                if edge.source_interface_id == source_id
            ]
        ).mean()
        for source_id in source_ids
    ]
    return torch.stack(source_means).mean()


def _automatic_interface_targets(
    left_available: int,
    right_available: int,
) -> tuple[int, int]:
    """Derive scale-aware coverage targets without asking the user.

    The square-root rule grows with interfaceable chain length but remains a
    local patch rather than trying to collapse an entire protomer.  The cap
    prevents large cages from receiving disproportionately strong gradients.
    """

    available = min(left_available, right_available)
    if available < 1:
        return 0, 0
    coverage = min(available, min(12, max(3, math.ceil(math.sqrt(available)))))
    continuity = min(coverage, max(2, math.ceil(0.6 * coverage)))
    return coverage, continuity


def graph_interface_energy(
    coordinates: torch.Tensor,
    topology: GraphInterfaceTopology,
    config: GraphInterfaceGuidanceConfig,
) -> GraphInterfaceEnergy:
    """Evaluate local contact attraction plus short-range repulsion."""

    if coordinates.ndim == 3:
        if coordinates.shape[0] != 1:
            raise ValueError("Interface guidance supports one pose batch")
        xyz = coordinates[0]
    else:
        xyz = coordinates
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape [L,3] or [1,L,3]")
    attractions = []
    coverages = []
    continuities = []
    clashes = []
    distance_terms = []
    minima = []
    selected_means = []
    covered_left_counts = []
    covered_right_counts = []
    coverage_targets = []
    continuity_targets = []
    contiguous_left_counts = []
    contiguous_right_counts = []
    for edge in topology.edges:
        distances = torch.cdist(
            xyz[edge.left_generated_ca_mask],
            xyz[edge.right_generated_ca_mask],
        ).flatten()
        if not distances.numel():
            raise ValueError(
                f"Designed interface {edge.edge_id!r} has no atom pairs"
            )
        if (
            edge.requested_contact_count > 0
            or edge.requested_residues_per_side > 0
            or edge.requested_contiguous_residues_per_side > 0
            or edge.automatic_quality
        ):
            pair_count = min(
                distances.numel(),
                max(config.pairs_per_edge, edge.requested_contact_count),
            )
            selected = torch.topk(
                distances,
                k=pair_count,
                largest=False,
            ).values
            # A side-chain heavy-atom cutoff is normally a few Angstrom
            # smaller than its corresponding CA separation. Use it to tighten
            # broad defaults without treating CA distance as a full packing
            # score.
            target_ca_distance = min(
                config.target_ca_distance,
                max(
                    config.clash_ca_distance + 0.5,
                    edge.contact_cutoff + 2.5,
                ),
            )
            excess = torch.relu(selected - target_ca_distance)
            attractions.append(
                functional.smooth_l1_loss(
                    excess,
                    torch.zeros_like(excess),
                    reduction="mean",
                    beta=1.0,
                )
            )
            selected_means.append(selected.mean())

            # Top-k pair attraction alone can satisfy many requested pairs
            # through one residue, creating a point contact rather than an
            # interface.  Require a balanced set of residues on *both* sides
            # to approach the opposing chain.  sqrt(contact_count) is a
            # conservative conversion from pair count to per-side coverage.
            left_nearest = distances.reshape(
                int(edge.left_generated_ca_mask.sum().item()),
                int(edge.right_generated_ca_mask.sum().item()),
            ).min(dim=1).values
            right_nearest = distances.reshape(
                int(edge.left_generated_ca_mask.sum().item()),
                int(edge.right_generated_ca_mask.sum().item()),
            ).min(dim=0).values
            automatic_coverage, automatic_continuity = (
                _automatic_interface_targets(
                    left_nearest.numel(), right_nearest.numel()
                )
            )
            if edge.requested_residues_per_side > 0:
                requested_residues = edge.requested_residues_per_side
            elif edge.requested_contact_count > 0:
                requested_residues = max(
                    2,
                    int(math.ceil(math.sqrt(edge.requested_contact_count))),
                )
            else:
                requested_residues = automatic_coverage
            requested_contiguous = (
                edge.requested_contiguous_residues_per_side
                if edge.requested_contiguous_residues_per_side > 0
                else automatic_continuity
            )
            left_count = min(requested_residues, left_nearest.numel())
            right_count = min(requested_residues, right_nearest.numel())
            left_selected = torch.topk(
                left_nearest,
                k=left_count,
                largest=False,
            ).values
            right_selected = torch.topk(
                right_nearest,
                k=right_count,
                largest=False,
            ).values
            coverage_excess = torch.cat(
                (
                    torch.relu(left_selected - target_ca_distance),
                    torch.relu(right_selected - target_ca_distance),
                )
            )
            coverages.append(
                functional.smooth_l1_loss(
                    coverage_excess,
                    torch.zeros_like(coverage_excess),
                    reduction="mean",
                    beta=1.0,
                )
            )
            covered_left_counts.append(
                torch.count_nonzero(left_nearest <= target_ca_distance)
            )
            covered_right_counts.append(
                torch.count_nonzero(right_nearest <= target_ca_distance)
            )
            coverage_targets.append(
                torch.tensor(
                    min(left_count, right_count),
                    device=xyz.device,
                    dtype=torch.long,
                )
            )
            continuity_target = min(
                requested_contiguous,
                left_nearest.numel(),
                right_nearest.numel(),
            )
            continuity_targets.append(
                torch.tensor(
                    continuity_target,
                    device=xyz.device,
                    dtype=torch.long,
                )
            )
            continuities.append(
                0.5
                * (
                    _continuity_loss(
                        left_nearest,
                        edge.left_generated_token_ids,
                        target_distance=target_ca_distance,
                        target_count=min(continuity_target, left_count),
                    )
                    + _continuity_loss(
                        right_nearest,
                        edge.right_generated_token_ids,
                        target_distance=target_ca_distance,
                        target_count=min(continuity_target, right_count),
                    )
                )
            )
            contiguous_left_counts.append(
                _maximum_contiguous_covered(
                    left_nearest,
                    edge.left_generated_token_ids,
                    target_distance=target_ca_distance,
                )
            )
            contiguous_right_counts.append(
                _maximum_contiguous_covered(
                    right_nearest,
                    edge.right_generated_token_ids,
                    target_distance=target_ca_distance,
                )
            )
        else:
            attractions.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
            )
            selected_means.append(distances.min())
            coverages.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
            )
            covered_left_counts.append(
                torch.zeros((), device=xyz.device, dtype=torch.long)
            )
            covered_right_counts.append(
                torch.zeros((), device=xyz.device, dtype=torch.long)
            )
            coverage_targets.append(
                torch.zeros((), device=xyz.device, dtype=torch.long)
            )
            continuity_targets.append(
                torch.zeros((), device=xyz.device, dtype=torch.long)
            )
            continuities.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
            )
            contiguous_left_counts.append(
                torch.zeros((), device=xyz.device, dtype=torch.long)
            )
            contiguous_right_counts.append(
                torch.zeros((), device=xyz.device, dtype=torch.long)
            )
        overlap = torch.relu(config.clash_ca_distance - distances)
        # Normalize by residues rather than all O(N^2) pairs.  Otherwise one
        # severe collision becomes numerically invisible for large cages.
        clashes.append(
            torch.sum(torch.square(overlap))
            / max(
                int(edge.left_generated_ca_mask.sum().item())
                + int(edge.right_generated_ca_mask.sum().item()),
                1,
            )
        )
        if edge.distance_target is None:
            distance_terms.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
            )
        else:
            centroid_distance = torch.linalg.vector_norm(
                xyz[edge.left_generated_ca_mask].mean(dim=0)
                - xyz[edge.right_generated_ca_mask].mean(dim=0)
            )
            tolerance = edge.distance_tolerance or 0.0
            distance_excess = torch.relu(
                torch.abs(centroid_distance - edge.distance_target)
                - tolerance
            )
            distance_terms.append(
                functional.smooth_l1_loss(
                    distance_excess,
                    torch.zeros_like(distance_excess),
                    reduction="mean",
                    beta=1.0,
                )
            )
        minima.append(distances.min())
    attraction = _balanced_source_mean(attractions, topology.edges)
    coverage = _balanced_source_mean(coverages, topology.edges)
    continuity = _balanced_source_mean(continuities, topology.edges)
    clash = _balanced_source_mean(clashes, topology.edges)
    distance = _balanced_source_mean(distance_terms, topology.edges)
    per_edge_total = torch.stack(
        [
            config.weight * edge_attraction
            + config.coverage_weight * edge_coverage
            + config.continuity_weight * edge_continuity
            + config.clash_weight * edge_clash
            + config.distance_weight * edge_distance
            for (
                edge_attraction,
                edge_coverage,
                edge_continuity,
                edge_clash,
                edge_distance,
            ) in zip(
                attractions,
                coverages,
                continuities,
                clashes,
                distance_terms,
                strict=True,
            )
        ]
    )
    return GraphInterfaceEnergy(
        total=(
            config.weight * attraction
            + config.coverage_weight * coverage
            + config.continuity_weight * continuity
            + config.clash_weight * clash
            + config.distance_weight * distance
        ),
        attraction=attraction,
        coverage=coverage,
        continuity=continuity,
        clash=clash,
        distance=distance,
        minimum_distances=torch.stack(minima),
        mean_selected_distances=torch.stack(selected_means),
        covered_left_residues=torch.stack(covered_left_counts),
        covered_right_residues=torch.stack(covered_right_counts),
        target_residues_per_side=torch.stack(coverage_targets),
        target_contiguous_residues_per_side=torch.stack(continuity_targets),
        contiguous_left_residues=torch.stack(contiguous_left_counts),
        contiguous_right_residues=torch.stack(contiguous_right_counts),
        per_edge_total=per_edge_total,
    )


def _smooth_selected_token_gradients(
    gradients: dict[int, torch.Tensor],
    token_chain_ids: torch.Tensor,
    *,
    weight: float,
    passes: int,
) -> dict[int, torch.Tensor]:
    """Smooth adjacent same-chain token motions without crossing fixed gaps."""

    if not gradients or passes == 0 or weight == 0.0:
        return gradients
    current = dict(gradients)
    selected = set(current)
    for _ in range(passes):
        updated: dict[int, torch.Tensor] = {}
        for token_id, gradient in current.items():
            chain_id = int(token_chain_ids[token_id].item())
            neighbours = [
                current[candidate]
                for candidate in (token_id - 1, token_id + 1)
                if candidate in selected
                and int(token_chain_ids[candidate].item()) == chain_id
            ]
            if not neighbours:
                updated[token_id] = gradient
                continue
            local_mean = torch.stack([gradient, *neighbours]).mean(dim=0)
            updated[token_id] = (1.0 - weight) * gradient + weight * local_mean
        current = updated
    return current


def guidance_window_weight(
    progress: float,
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    """Smoothly open and close the field inside its declared time window."""

    if progress <= start_fraction or progress >= end_fraction:
        return 0.0
    phase = (progress - start_fraction) / (end_fraction - start_fraction)
    return math.sin(math.pi * phase)


def apply_graph_interface_guidance(
    coordinates: torch.Tensor,
    features: dict[str, Any],
    topology: GraphInterfaceTopology,
    *,
    progress: float,
    config: GraphInterfaceGuidanceConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one bounded token-rigid gradient step to generated atoms only."""

    window = guidance_window_weight(
        progress,
        start_fraction=config.start_fraction,
        end_fraction=config.end_fraction,
    )
    if window == 0.0 or (
        config.weight == 0.0
        and config.coverage_weight == 0.0
        and config.continuity_weight == 0.0
        and config.clash_weight == 0.0
        and config.distance_weight == 0.0
    ):
        return coordinates, {"applied": False, "window_weight": window}
    with torch.enable_grad():
        proposal = coordinates.detach().clone().requires_grad_(True)
        energy = graph_interface_energy(proposal, topology, config)
        gradient = torch.autograd.grad(energy.total, proposal)[0]
    if not torch.isfinite(gradient).all():
        raise ValueError("Interface guidance produced a non-finite gradient")

    atom_to_token = torch.as_tensor(
        features["atom_to_token_map"],
        dtype=torch.long,
        device=coordinates.device,
    )
    guided = coordinates.clone()
    maximum_observed_step = 0.0
    selected_tokens = torch.unique(
        atom_to_token[topology.generated_atom_mask]
    )
    token_gradients: dict[int, torch.Tensor] = {}
    for token_id in selected_tokens:
        token_mask = (
            topology.generated_atom_mask
            & (atom_to_token == token_id)
        )
        # A rigid token translation changes every atom slot by the same
        # vector, so its derivative is the *sum* of atom derivatives.  Taking
        # a mean would dilute a CA-only objective by the number of atom slots
        # and make guidance residue-template dependent.
        token_gradients[int(token_id.item())] = gradient[:, token_mask, :].sum(
            dim=1,
            keepdim=True,
        )
    token_chain_ids = torch.as_tensor(
        features["asym_id"],
        dtype=torch.long,
        device=coordinates.device,
    )
    token_gradients = _smooth_selected_token_gradients(
        token_gradients,
        token_chain_ids,
        weight=config.token_smoothing_weight,
        passes=config.token_smoothing_passes,
    )
    observed_steps = []
    for token_id in selected_tokens:
        token_index = int(token_id.item())
        token_mask = (
            topology.generated_atom_mask
            & (atom_to_token == token_id)
        )
        token_gradient = token_gradients[token_index]
        step = -window * token_gradient
        norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
        scale = torch.clamp(
            config.maximum_token_step / norm.clamp_min(1e-8),
            max=1.0,
        )
        step = step * scale
        maximum_observed_step = max(
            maximum_observed_step,
            float(torch.linalg.vector_norm(step).detach().cpu().item()),
        )
        observed_steps.append(torch.linalg.vector_norm(step))
        guided[:, token_mask, :] += step
    return guided.detach(), {
        "applied": True,
        "window_weight": window,
        "edge_count": len(topology.edges),
        "energy": float(energy.total.detach().cpu().item()),
        "attraction": float(energy.attraction.detach().cpu().item()),
        "coverage": float(energy.coverage.detach().cpu().item()),
        "continuity": float(energy.continuity.detach().cpu().item()),
        "clash": float(energy.clash.detach().cpu().item()),
        "distance": float(energy.distance.detach().cpu().item()),
        "minimum_distances": energy.minimum_distances.detach().cpu().tolist(),
        "mean_selected_distances": (
            energy.mean_selected_distances.detach().cpu().tolist()
        ),
        "covered_left_residues": (
            energy.covered_left_residues.detach().cpu().tolist()
        ),
        "covered_right_residues": (
            energy.covered_right_residues.detach().cpu().tolist()
        ),
        "target_residues_per_side": (
            energy.target_residues_per_side.detach().cpu().tolist()
        ),
        "target_contiguous_residues_per_side": (
            energy.target_contiguous_residues_per_side.detach().cpu().tolist()
        ),
        "contiguous_left_residues": (
            energy.contiguous_left_residues.detach().cpu().tolist()
        ),
        "contiguous_right_residues": (
            energy.contiguous_right_residues.detach().cpu().tolist()
        ),
        "per_edge_total": energy.per_edge_total.detach().cpu().tolist(),
        "maximum_token_step": maximum_observed_step,
        "mean_token_step": float(
            torch.stack(observed_steps).mean().detach().cpu().item()
        ),
    }


__all__ = [
    "GraphInterfaceEdge",
    "GraphInterfaceEnergy",
    "GraphInterfaceGuidanceConfig",
    "GraphInterfaceTopology",
    "apply_graph_interface_guidance",
    "build_graph_interface_topology",
    "graph_interface_energy",
    "guidance_window_weight",
]
