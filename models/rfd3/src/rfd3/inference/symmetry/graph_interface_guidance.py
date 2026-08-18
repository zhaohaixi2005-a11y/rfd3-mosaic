"""Symmetry-coupled soft guidance for graph-declared design interfaces.

The fixed-interface and designed-interface cases share one assembly graph and
one sampler.  Input-stage relations are handled by the exact constraint
projector.  Output-stage contact relations are converted here into a soft
field over generated residues.  Every symmetry-expanded edge contributes to
one joint energy, so guidance cannot move one copy independently or collapse
unrelated protomers through an all-to-all compactness force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class GraphInterfaceGuidanceConfig:
    """Trust region and schedule for generated-interface guidance."""

    weight: float = 1.0
    coverage_weight: float = 1.0
    continuity_weight: float = 1.0
    orientation_weight: float = 0.25
    shape_weight: float = 0.5
    backbone_weight: float = 0.1
    interface_balance_weight: float = 0.5
    patch_exclusivity_weight: float = 1.0
    clash_weight: float = 8.0
    distance_weight: float = 0.25
    target_ca_distance: float = 8.0
    clash_ca_distance: float = 3.5
    pairs_per_edge: int = 8
    start_fraction: float = 0.05
    end_fraction: float = 0.80
    # Generated contacts are especially vulnerable to being erased by the
    # denoiser near t=0.  Keep a strong bounded field active through the last
    # diffusion steps; the per-token trust region still caps every move.
    terminal_weight_floor: float = 0.8
    maximum_token_step: float = 0.25
    unsatisfied_step_fraction: float = 0.50
    final_polish_steps: int = 12
    token_smoothing_weight: float = 0.5
    token_smoothing_passes: int = 1
    continuity_softness: float = 0.75
    maximum_tangent_normal_cosine: float = 0.65
    backbone_ca_distance: float = 3.8
    backbone_ca_tolerance: float = 0.5
    # A contact patch is a piece of backbone, not a cloud of independent CA
    # points.  Most of the guidance proposal is therefore projected onto a
    # local rigid-body motion before it is blended into neighbouring tokens.
    # The remaining fraction retains genuine local flexibility for the
    # denoiser and for junction accommodation.
    patch_rigid_weight: float = 1.0
    patch_blend_radius: int = 2
    maximum_patch_rotation_degrees: float = 2.0
    # Patch discovery is intentionally adaptive during early noisy capture,
    # then becomes a stateful contract.  Without this lock an optimizer can
    # appear to improve merely by choosing different residues at every
    # timestep (or even at every line-search trial), producing the scattered
    # interfaces seen in early canaries.
    patch_lock_fraction: float = 0.50
    line_search_steps: int = 5
    line_search_contraction: float = 0.5
    capture_ca_distance: float = 12.0
    maximum_orientation_loss: float = 0.05
    maximum_shape_loss: float = 0.08
    maximum_backbone_loss: float = 0.02
    maximum_patch_exclusivity_loss: float = 0.05
    # A good interface may not pay for a materially worse one.  These small
    # tolerances absorb floating-point and projection noise while preserving
    # the worst declared source interface during a joint move.
    maximum_source_regression_fraction: float = 0.02
    maximum_source_regression_absolute: float = 2.0e-3

    def __post_init__(self) -> None:
        if (
            self.weight < 0.0
            or self.coverage_weight < 0.0
            or self.continuity_weight < 0.0
            or self.orientation_weight < 0.0
            or self.shape_weight < 0.0
            or self.backbone_weight < 0.0
            or self.interface_balance_weight < 0.0
            or self.patch_exclusivity_weight < 0.0
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
        if not 0.0 <= self.terminal_weight_floor <= 1.0:
            raise ValueError(
                "terminal_weight_floor must be between zero and one"
            )
        if self.maximum_token_step <= 0.0:
            raise ValueError("maximum_token_step must be positive")
        if not 0.0 <= self.unsatisfied_step_fraction <= 1.0:
            raise ValueError(
                "unsatisfied_step_fraction must be between zero and one"
            )
        if self.final_polish_steps < 0:
            raise ValueError("final_polish_steps cannot be negative")
        if not 0.0 <= self.token_smoothing_weight <= 1.0:
            raise ValueError(
                "token_smoothing_weight must be between zero and one"
            )
        if self.token_smoothing_passes < 0:
            raise ValueError("token_smoothing_passes cannot be negative")
        if self.continuity_softness <= 0.0:
            raise ValueError("continuity_softness must be positive")
        if not 0.0 <= self.maximum_tangent_normal_cosine <= 1.0:
            raise ValueError(
                "maximum_tangent_normal_cosine must be between zero and one"
            )
        if self.backbone_ca_distance <= 0.0:
            raise ValueError("backbone_ca_distance must be positive")
        if self.backbone_ca_tolerance < 0.0:
            raise ValueError("backbone_ca_tolerance cannot be negative")
        if not 0.0 <= self.patch_rigid_weight <= 1.0:
            raise ValueError("patch_rigid_weight must be between zero and one")
        if self.patch_blend_radius < 0:
            raise ValueError("patch_blend_radius cannot be negative")
        if self.maximum_patch_rotation_degrees < 0.0:
            raise ValueError(
                "maximum_patch_rotation_degrees cannot be negative"
            )
        if not 0.0 <= self.patch_lock_fraction <= 1.0:
            raise ValueError(
                "patch_lock_fraction must be between zero and one"
            )
        if self.line_search_steps < 1:
            raise ValueError("line_search_steps must be positive")
        if not 0.0 < self.line_search_contraction < 1.0:
            raise ValueError(
                "line_search_contraction must be strictly between zero and one"
            )
        if self.capture_ca_distance < self.target_ca_distance:
            raise ValueError(
                "capture_ca_distance cannot be smaller than target_ca_distance"
            )
        for name in (
            "maximum_orientation_loss",
            "maximum_shape_loss",
            "maximum_backbone_loss",
            "maximum_patch_exclusivity_loss",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.maximum_source_regression_fraction < 0.0:
            raise ValueError(
                "maximum_source_regression_fraction cannot be negative"
            )
        if self.maximum_source_regression_absolute < 0.0:
            raise ValueError(
                "maximum_source_regression_absolute cannot be negative"
            )


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
    # Optional global CA safety field.  Rows correspond to guided generated
    # CAs and columns to every real CA in the assembly.  ``safety_exclusions``
    # removes self, same-residue and direct covalent-neighbour pairs.
    guided_ca_mask: torch.Tensor | None = None
    safety_ca_mask: torch.Tensor | None = None
    safety_exclusions: torch.Tensor | None = None
    # Every peptide CA--CA edge touched by a guided generated token.  The
    # historical name is retained for result compatibility, but this now
    # includes generated--generated patch boundaries as well as the original
    # generated--fixed junctions.  A locally rigid interface patch must not
    # tear away from the rest of its generated chain.
    junction_ca_pairs: torch.Tensor | None = None
    capacity_preflight: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GraphInterfaceEnergy:
    """Joint attractive/repulsive energy over all declared edges."""

    total: torch.Tensor
    attraction: torch.Tensor
    coverage: torch.Tensor
    continuity: torch.Tensor
    orientation: torch.Tensor
    shape: torch.Tensor
    backbone: torch.Tensor
    junction: torch.Tensor
    interface_balance: torch.Tensor
    patch_exclusivity: torch.Tensor
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
    per_edge_orientation: torch.Tensor
    per_edge_shape: torch.Tensor
    per_edge_backbone: torch.Tensor
    per_edge_total: torch.Tensor
    per_source_total: torch.Tensor
    global_safety_clash: torch.Tensor
    minimum_global_safety_distance: torch.Tensor


@dataclass(frozen=True)
class _ContiguousPatch:
    """Differentiable score and token-local indices for one contact patch."""

    indices: torch.Tensor
    loss: torch.Tensor


@dataclass(frozen=True)
class _PairedContiguousPatch:
    """One reciprocal pair of sequence-contiguous interface windows.

    ``left_nearest`` and ``right_nearest`` are deliberately measured only
    against the selected opposing window.  This prevents a left-hand patch
    from borrowing contacts from one part of the right chain while the
    right-hand patch borrows contacts from a different part of the left
    chain.  The window choice is discrete, but all returned distance tensors
    retain their autograd connection to the coordinates.
    """

    left_indices: torch.Tensor
    right_indices: torch.Tensor
    distances: torch.Tensor
    left_nearest: torch.Tensor
    right_nearest: torch.Tensor


@dataclass(frozen=True)
class _EdgePatchResolution:
    """The single reciprocal patch and quality targets used by one edge."""

    patch: _PairedContiguousPatch
    target_ca_distance: float
    left_count: int
    right_count: int
    continuity_target: int
    pair_count: int


@dataclass(frozen=True)
class GraphInterfacePatchAssignment:
    """Stable token identities for one reciprocal physical interface."""

    left_token_ids: tuple[int, ...]
    right_token_ids: tuple[int, ...]


@dataclass
class GraphInterfacePatchState:
    """Sampler-owned patch assignments shared across diffusion timesteps."""

    assignments: dict[str, GraphInterfacePatchAssignment]
    locked: bool = False
    lock_reason: str | None = None


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
    has_distance_targets = torch.as_tensor(
        features.get(
            "assembly_interface_has_distance_target",
            torch.isfinite(distance_targets)
            & torch.isfinite(distance_tolerances),
        ),
        dtype=torch.bool,
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
        == has_distance_targets.shape
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
                    if bool(has_distance_targets[index].item())
                    else None
                ),
                distance_tolerance=(
                    float(distance_tolerances[index].item())
                    if bool(has_distance_targets[index].item())
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
    guided_ca_mask = used_generated & is_ca
    safety_ca_mask = is_ca & ~is_virtual
    guided_tokens = atom_to_token[guided_ca_mask]
    safety_tokens = atom_to_token[safety_ca_mask]
    safety_exclusions = guided_tokens[:, None] == safety_tokens[None, :]
    token_count = int(asym_id.numel())
    token_adjacency = torch.zeros(
        (token_count, token_count),
        dtype=torch.bool,
        device=device,
    )
    token_bonds_value = features.get("token_bonds")
    if token_bonds_value is not None:
        token_bonds = torch.as_tensor(
            token_bonds_value,
            dtype=torch.bool,
            device=device,
        )
        if token_bonds.shape != (token_count, token_count):
            raise ValueError(
                "token_bonds must have shape [N_tokens,N_tokens]"
            )
        token_adjacency |= token_bonds | token_bonds.T
    residue_index_value = features.get("residue_index")
    if residue_index_value is None:
        token_positions = torch.arange(token_count, device=device)
    else:
        token_positions = torch.as_tensor(
            residue_index_value,
            dtype=torch.long,
            device=device,
        )
        if token_positions.shape != (token_count,):
            raise ValueError("residue_index must have shape [N_tokens]")
    same_chain_tokens = asym_id[:, None] == asym_id[None, :]
    peptide_neighbours = (
        same_chain_tokens
        & (torch.abs(token_positions[:, None] - token_positions[None, :]) == 1)
    )
    # ``residue_index`` is not guaranteed to remain consecutive across a
    # compiled fixed/generated boundary.  In particular, public contigs can
    # preserve source residue numbering on the fixed fragment while generated
    # tokens use a different numbering interval.  The atom array is still in
    # polymer order, so consecutive CA tokens belonging to the same chain are
    # also covalent neighbours.  Without this fallback a mobile motif can pull
    # away from its generated scaffold while reporting zero junction loss.
    ca_atom_indices = torch.nonzero(
        is_ca & ~is_virtual,
        as_tuple=False,
    ).flatten()
    ca_tokens = atom_to_token[ca_atom_indices]
    ordered_ca_tokens = torch.unique_consecutive(ca_tokens)
    if ordered_ca_tokens.numel() > 1:
        left_tokens = ordered_ca_tokens[:-1]
        right_tokens = ordered_ca_tokens[1:]
        same_ordered_chain = asym_id[left_tokens] == asym_id[right_tokens]
        left_tokens = left_tokens[same_ordered_chain]
        right_tokens = right_tokens[same_ordered_chain]
        peptide_neighbours[left_tokens, right_tokens] = True
        peptide_neighbours[right_tokens, left_tokens] = True
    token_adjacency |= peptide_neighbours
    safety_exclusions |= token_adjacency[
        guided_tokens[:, None], safety_tokens[None, :]
    ]

    guided_token_flags = torch.zeros(
        token_count,
        dtype=torch.bool,
        device=device,
    )
    guided_token_flags[torch.unique(guided_tokens)] = True
    protected_peptide_edges = (
        peptide_neighbours
        & (
            guided_token_flags[:, None]
            | guided_token_flags[None, :]
        )
    )
    protected_token_pairs = torch.nonzero(
        torch.triu(protected_peptide_edges, diagonal=1),
        as_tuple=False,
    )
    junction_pairs = []
    for left_token, right_token in protected_token_pairs:
        left_ca_atoms = ca_atom_indices[ca_tokens == left_token]
        right_ca_atoms = ca_atom_indices[ca_tokens == right_token]
        if left_ca_atoms.numel() == 1 and right_ca_atoms.numel() == 1:
            junction_pairs.append(
                torch.stack((left_ca_atoms[0], right_ca_atoms[0]))
            )
    junction_ca_pairs = (
        torch.stack(junction_pairs)
        if junction_pairs
        else torch.empty((0, 2), dtype=torch.long, device=device)
    )
    topology = GraphInterfaceTopology(
        edges=tuple(edges),
        generated_atom_mask=generated_atom_mask,
        guided_ca_mask=guided_ca_mask,
        safety_ca_mask=safety_ca_mask,
        safety_exclusions=safety_exclusions,
        junction_ca_pairs=junction_ca_pairs,
    )
    return replace(
        topology,
        capacity_preflight=graph_interface_capacity_preflight(topology),
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


def _maximum_contiguous_available(token_ids: torch.Tensor) -> int:
    return max(
        (run.stop - run.start for run in _contiguous_token_runs(token_ids)),
        default=0,
    )


def _contiguous_windows(
    token_ids: torch.Tensor,
    *,
    target_count: int,
) -> torch.Tensor:
    """Enumerate fixed-width windows without crossing sequence gaps.

    If the requested width is unavailable, windows use the longest real run
    rather than concatenating disconnected fragments.  The downstream
    occupancy deficit still records that the requested patch was too large.
    """

    runs = _contiguous_token_runs(token_ids)
    maximum = max((run.stop - run.start for run in runs), default=0)
    if maximum < 1:
        return torch.empty(
            (0, 0),
            device=token_ids.device,
            dtype=torch.long,
        )
    width = min(max(int(target_count), 1), maximum)
    windows = [
        torch.arange(
            start,
            start + width,
            device=token_ids.device,
            dtype=torch.long,
        )
        for run in runs
        for start in range(run.start, run.stop - width + 1)
    ]
    return torch.stack(windows)


def _candidate_occupancy_deficit(
    nearest_distances: torch.Tensor,
    *,
    target_distance: float,
    target_count: int,
    softness: float,
    contiguous: bool,
) -> torch.Tensor:
    """Score candidate-window occupancy along its final dimension."""

    requested = max(int(target_count), 1)
    occupancy = torch.sigmoid(
        (target_distance - nearest_distances) / softness
    )
    available = occupancy.shape[-1]
    selected_count = min(requested, available)
    if contiguous:
        selected_occupancy = occupancy.unfold(
            -1,
            selected_count,
            1,
        ).sum(dim=-1).max(dim=-1).values
    else:
        selected_occupancy = torch.topk(
            occupancy,
            k=selected_count,
            dim=-1,
        ).values.sum(dim=-1)
    return torch.square(
        torch.relu(float(requested) - selected_occupancy)
        / float(requested)
    )


def _best_paired_contiguous_patch(
    distance_matrix: torch.Tensor,
    left_token_ids: torch.Tensor,
    right_token_ids: torch.Tensor,
    *,
    target_distance: float,
    left_window_count: int,
    right_window_count: int,
    coverage_count: int,
    continuity_count: int,
    pair_count: int,
    softness: float,
) -> _PairedContiguousPatch:
    """Select one mutually compatible pair of contiguous sequence windows.

    Candidate scoring is vectorized and performed on detached distances so
    the discrete argmin does not retain a graph for every possible window
    pair.  Once selected, the winning submatrix is gathered again from the
    original distance matrix; attraction, coverage, continuity, orientation
    and shape therefore remain differentiable with respect to the current
    coordinates while sharing exactly one reciprocal patch assignment.
    """

    left_windows = _contiguous_windows(
        left_token_ids,
        target_count=left_window_count,
    )
    right_windows = _contiguous_windows(
        right_token_ids,
        target_count=right_window_count,
    )
    if left_windows.numel() == 0 or right_windows.numel() == 0:
        raise ValueError(
            "Designed-interface paired patch has no contiguous token window"
        )

    with torch.no_grad():
        detached = distance_matrix.detach()
        candidate_distances = detached[
            left_windows[:, None, :, None],
            right_windows[None, :, None, :],
        ]
        flat = candidate_distances.flatten(start_dim=-2)
        selected_pair_count = min(max(int(pair_count), 1), flat.shape[-1])
        selected_pairs = torch.topk(
            flat,
            k=selected_pair_count,
            dim=-1,
            largest=False,
        ).values
        pair_excess = torch.relu(selected_pairs - target_distance)
        attraction_score = functional.smooth_l1_loss(
            pair_excess,
            torch.zeros_like(pair_excess),
            reduction="none",
            beta=1.0,
        ).mean(dim=-1)

        left_nearest = candidate_distances.min(dim=-1).values
        right_nearest = candidate_distances.min(dim=-2).values
        coverage_score = 0.5 * (
            _candidate_occupancy_deficit(
                left_nearest,
                target_distance=target_distance,
                target_count=coverage_count,
                softness=softness,
                contiguous=False,
            )
            + _candidate_occupancy_deficit(
                right_nearest,
                target_distance=target_distance,
                target_count=coverage_count,
                softness=softness,
                contiguous=False,
            )
        )
        continuity_score = 0.5 * (
            _candidate_occupancy_deficit(
                left_nearest,
                target_distance=target_distance,
                target_count=continuity_count,
                softness=softness,
                contiguous=True,
            )
            + _candidate_occupancy_deficit(
                right_nearest,
                target_distance=target_distance,
                target_count=continuity_count,
                softness=softness,
                contiguous=True,
            )
        )
        compatibility = attraction_score + coverage_score + continuity_score
        selected_flat = int(
            torch.argmin(compatibility.flatten()).cpu().item()
        )
        right_candidate_count = right_windows.shape[0]
        left_candidate = selected_flat // right_candidate_count
        right_candidate = selected_flat % right_candidate_count

    left_indices = left_windows[left_candidate]
    right_indices = right_windows[right_candidate]
    selected_distances = distance_matrix[left_indices][:, right_indices]
    return _PairedContiguousPatch(
        left_indices=left_indices,
        right_indices=right_indices,
        distances=selected_distances,
        left_nearest=selected_distances.min(dim=1).values,
        right_nearest=selected_distances.min(dim=0).values,
    )


def _best_contiguous_patch(
    nearest_distances: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    target_distance: float,
    target_count: int,
    softness: float,
) -> _ContiguousPatch:
    """Select one real sequence-contiguous patch and optimize its occupancy.

    The previous proxy averaged distance excess and fell back to an arbitrary
    short run when no run was long enough.  Consequently several disconnected
    2-residue contacts could have zero loss for a requested 6-residue patch.
    Here every candidate is an actual token-adjacent window.  A soft occupancy
    deficit keeps a useful gradient even when residues sit just inside the CA
    cutoff, which is important because the final gate uses stricter heavy-atom
    contacts.
    """

    requested = max(int(target_count), 1)
    candidates: list[_ContiguousPatch] = []
    short_candidates: list[_ContiguousPatch] = []
    for run in _contiguous_token_runs(token_ids):
        run_length = run.stop - run.start
        if run_length < 1:
            continue
        window_length = min(requested, run_length)
        windows = nearest_distances[run].unfold(
            0,
            window_length,
            1,
        )
        occupancy = torch.sigmoid(
            (target_distance - windows) / softness
        )
        occupancy_deficit = torch.relu(
            float(requested) - occupancy.sum(dim=-1)
        ) / float(requested)
        depth_excess = torch.relu(windows - target_distance) / max(
            target_distance,
            1e-6,
        )
        losses = torch.square(occupancy_deficit) + 0.25 * torch.mean(
            torch.square(depth_excess),
            dim=-1,
        )
        selected_window = int(
            torch.argmin(losses.detach()).cpu().item()
        )
        patch = _ContiguousPatch(
            indices=torch.arange(
                run.start + selected_window,
                run.start + selected_window + window_length,
                device=token_ids.device,
                dtype=torch.long,
            ),
            loss=losses[selected_window],
        )
        if run_length >= requested:
            candidates.append(patch)
        else:
            short_candidates.append(patch)
    available = candidates if candidates else short_candidates
    if not available:
        return _ContiguousPatch(
            indices=torch.empty(
                0,
                device=token_ids.device,
                dtype=torch.long,
            ),
            loss=torch.ones(
                (),
                device=nearest_distances.device,
                dtype=nearest_distances.dtype,
            ),
        )
    return min(
        available,
        key=lambda patch: float(patch.loss.detach().cpu().item()),
    )


def _continuity_loss(
    nearest_distances: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    target_distance: float,
    target_count: int,
    softness: float = 0.75,
) -> torch.Tensor:
    """Penalize the best genuine contiguous interface patch on one side."""

    if target_count <= 0:
        return torch.zeros(
            (),
            device=nearest_distances.device,
            dtype=nearest_distances.dtype,
        )
    return _best_contiguous_patch(
        nearest_distances,
        token_ids,
        target_distance=target_distance,
        target_count=target_count,
        softness=softness,
    ).loss


def _paired_window_continuity_loss(
    nearest_distances: torch.Tensor,
    *,
    target_distance: float,
    target_count: int,
    softness: float,
) -> torch.Tensor:
    """Differentiably score continuity inside one selected paired window."""

    if target_count <= 0:
        return torch.zeros(
            (),
            device=nearest_distances.device,
            dtype=nearest_distances.dtype,
        )
    requested = max(int(target_count), 1)
    width = min(requested, nearest_distances.numel())
    windows = nearest_distances.unfold(0, width, 1)
    occupancy = torch.sigmoid((target_distance - windows) / softness)
    occupancy_deficit = torch.relu(
        float(requested) - occupancy.sum(dim=-1)
    ) / float(requested)
    depth_excess = torch.relu(windows - target_distance) / max(
        target_distance,
        1e-6,
    )
    losses = torch.square(occupancy_deficit) + 0.25 * torch.mean(
        torch.square(depth_excess),
        dim=-1,
    )
    return losses.min()


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

    return _source_interface_means(values, edges).mean()


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


def graph_interface_capacity_preflight(
    topology: GraphInterfaceTopology,
) -> tuple[dict[str, Any], ...]:
    """Prove immutable generated-patch capacity before diffusion starts.

    Coordinates are intentionally absent from this check: diffusion may
    change them.  Token count, sequence-contiguous capacity and competing
    interface ownership cannot change, so impossible explicit targets and
    patch-exclusivity over-subscription can be rejected without a GPU run.
    """

    records: list[dict[str, Any]] = []
    pools: list[tuple[str, set[int], int]] = []
    for edge in topology.edges:
        left_available = int(edge.left_generated_token_ids.numel())
        right_available = int(edge.right_generated_token_ids.numel())
        automatic_coverage, automatic_continuity = (
            _automatic_interface_targets(left_available, right_available)
        )
        requested_coverage = (
            edge.requested_residues_per_side
            if edge.requested_residues_per_side > 0
            else (
                min(
                    left_available,
                    right_available,
                    max(
                        2,
                        int(
                            math.ceil(
                                math.sqrt(edge.requested_contact_count)
                            )
                        ),
                    ),
                )
                if edge.requested_contact_count > 0
                else automatic_coverage
            )
        )
        left_contiguous = _maximum_contiguous_available(
            edge.left_generated_token_ids
        )
        right_contiguous = _maximum_contiguous_available(
            edge.right_generated_token_ids
        )
        requested_contiguous = (
            edge.requested_contiguous_residues_per_side
            if edge.requested_contiguous_residues_per_side > 0
            else min(
                automatic_continuity,
                left_contiguous,
                right_contiguous,
            )
        )
        if requested_coverage > min(left_available, right_available):
            raise ValueError(
                f"Designed interface {edge.edge_id!r} requests "
                f"{requested_coverage} generated residues per side but "
                f"only {left_available}/{right_available} exist"
            )
        if requested_contiguous > min(left_contiguous, right_contiguous):
            raise ValueError(
                f"Designed interface {edge.edge_id!r} requires a contiguous "
                f"{requested_contiguous}-residue patch but immutable token "
                f"capacities are {left_contiguous}/{right_contiguous}"
            )
        if edge.requested_contact_count > left_available * right_available:
            raise ValueError(
                f"Designed interface {edge.edge_id!r} requests "
                f"{edge.requested_contact_count} contacts but only "
                f"{left_available * right_available} generated residue "
                "pairs exist"
            )
        records.append(
            {
                "edge_id": edge.edge_id,
                "source_interface_id": edge.source_interface_id,
                "requested_residues_per_side": requested_coverage,
                "requested_contiguous_residues_per_side": (
                    requested_contiguous
                ),
                "available_residues_left": left_available,
                "available_residues_right": right_available,
                "available_contiguous_residues_left": left_contiguous,
                "available_contiguous_residues_right": right_contiguous,
            }
        )
        pools.extend(
            (
                (
                    f"{edge.edge_id}:left",
                    set(
                        int(value)
                        for value in edge.left_generated_token_ids.tolist()
                    ),
                    requested_coverage,
                ),
                (
                    f"{edge.edge_id}:right",
                    set(
                        int(value)
                        for value in edge.right_generated_token_ids.tolist()
                    ),
                    requested_coverage,
                ),
            )
        )

    # Connected overlap components are the exact resource pools within which
    # distinct physical interfaces compete for generated residues.
    remaining = list(range(len(pools)))
    while remaining:
        component = {remaining.pop(0)}
        union = set(pools[next(iter(component))][1])
        changed = True
        while changed:
            changed = False
            for index in tuple(remaining):
                if union.intersection(pools[index][1]):
                    remaining.remove(index)
                    component.add(index)
                    union.update(pools[index][1])
                    changed = True
        demand = sum(pools[index][2] for index in component)
        if demand > len(union):
            participants = sorted(pools[index][0] for index in component)
            raise ValueError(
                "Generated interface patches are over-subscribed: "
                f"participants={participants} require {demand} exclusive "
                f"residues but their shared pool contains {len(union)}"
            )
    return tuple(records)


def _assigned_patch(
    edge: GraphInterfaceEdge,
    distance_matrix: torch.Tensor,
    assignment: GraphInterfacePatchAssignment,
) -> _PairedContiguousPatch:
    """Gather a previously selected patch by stable runtime token identity."""

    def local_indices(
        available: torch.Tensor,
        requested: tuple[int, ...],
        side: str,
    ) -> torch.Tensor:
        positions = {
            int(token_id): index
            for index, token_id in enumerate(
                available.detach().cpu().tolist()
            )
        }
        missing = [token_id for token_id in requested if token_id not in positions]
        if missing:
            raise ValueError(
                f"Locked interface patch {edge.edge_id!r} references missing "
                f"{side} token IDs {missing}"
            )
        return torch.tensor(
            [positions[token_id] for token_id in requested],
            dtype=torch.long,
            device=distance_matrix.device,
        )

    left_indices = local_indices(
        edge.left_generated_token_ids,
        assignment.left_token_ids,
        "left",
    )
    right_indices = local_indices(
        edge.right_generated_token_ids,
        assignment.right_token_ids,
        "right",
    )
    if left_indices.numel() == 0 or right_indices.numel() == 0:
        raise ValueError(
            f"Locked interface patch {edge.edge_id!r} cannot be empty"
        )
    selected = distance_matrix[left_indices][:, right_indices]
    return _PairedContiguousPatch(
        left_indices=left_indices,
        right_indices=right_indices,
        distances=selected,
        left_nearest=selected.min(dim=1).values,
        right_nearest=selected.min(dim=0).values,
    )


def _resolve_edge_patch(
    edge: GraphInterfaceEdge,
    distance_matrix: torch.Tensor,
    config: GraphInterfaceGuidanceConfig,
    *,
    target_ca_distance_override: float | None = None,
    assignment: GraphInterfacePatchAssignment | None = None,
) -> _EdgePatchResolution:
    """Resolve one patch once so scoring and motion use identical residues."""

    final_target_ca_distance = min(
        config.target_ca_distance,
        max(config.clash_ca_distance + 0.5, edge.contact_cutoff + 2.5),
    )
    target_ca_distance = (
        final_target_ca_distance
        if target_ca_distance_override is None
        else final_target_ca_distance
        + max(
            target_ca_distance_override - config.target_ca_distance,
            0.0,
        )
    )
    left_available = distance_matrix.shape[0]
    right_available = distance_matrix.shape[1]
    automatic_coverage, automatic_continuity = (
        _automatic_interface_targets(left_available, right_available)
    )
    automatic_continuity = min(
        automatic_continuity,
        _maximum_contiguous_available(edge.left_generated_token_ids),
        _maximum_contiguous_available(edge.right_generated_token_ids),
    )
    if edge.requested_residues_per_side > min(
        left_available,
        right_available,
    ):
        raise ValueError(
            f"Designed interface {edge.edge_id!r} requests "
            f"{edge.requested_residues_per_side} residues per side but only "
            f"{left_available}/{right_available} generated residues are "
            "available"
        )
    # A disconnected current chain can still become contiguous through the
    # denoiser, so retain its explicit deficit in the energy.  Total residue
    # capacity, unlike current geometry, is immutable and safe to fail here.
    if edge.requested_contact_count > left_available * right_available:
        raise ValueError(
            f"Designed interface {edge.edge_id!r} requests "
            f"{edge.requested_contact_count} residue contacts but its "
            f"generated sides provide at most "
            f"{left_available * right_available} pairs"
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
    left_count = min(requested_residues, left_available)
    right_count = min(requested_residues, right_available)
    continuity_target = min(
        requested_contiguous,
        left_available,
        right_available,
    )
    pair_count = min(
        distance_matrix.numel(),
        max(config.pairs_per_edge, edge.requested_contact_count),
    )
    patch = (
        _assigned_patch(edge, distance_matrix, assignment)
        if assignment is not None
        else _best_paired_contiguous_patch(
            distance_matrix,
            edge.left_generated_token_ids,
            edge.right_generated_token_ids,
            target_distance=target_ca_distance,
            left_window_count=max(left_count, continuity_target, 1),
            right_window_count=max(right_count, continuity_target, 1),
            coverage_count=max(min(left_count, right_count), 1),
            continuity_count=max(continuity_target, 1),
            pair_count=pair_count,
            softness=config.continuity_softness,
        )
    )
    return _EdgePatchResolution(
        patch=patch,
        target_ca_distance=target_ca_distance,
        left_count=left_count,
        right_count=right_count,
        continuity_target=continuity_target,
        pair_count=pair_count,
    )


def resolve_graph_interface_patch_assignments(
    coordinates: torch.Tensor,
    topology: GraphInterfaceTopology,
    config: GraphInterfaceGuidanceConfig,
    *,
    target_ca_distance_override: float | None = None,
) -> dict[str, GraphInterfacePatchAssignment]:
    """Resolve one reciprocal contiguous patch for every quality edge."""

    xyz = coordinates[0] if coordinates.ndim == 3 else coordinates
    assignments: dict[str, GraphInterfacePatchAssignment] = {}
    for edge in topology.edges:
        if not (
            edge.requested_contact_count > 0
            or edge.requested_residues_per_side > 0
            or edge.requested_contiguous_residues_per_side > 0
            or edge.automatic_quality
        ):
            continue
        left_points = xyz[edge.left_generated_ca_mask]
        right_points = xyz[edge.right_generated_ca_mask]
        resolved = _resolve_edge_patch(
            edge,
            torch.cdist(left_points, right_points),
            config,
            target_ca_distance_override=target_ca_distance_override,
        )
        assignments[edge.edge_id] = GraphInterfacePatchAssignment(
            left_token_ids=tuple(
                int(value)
                for value in edge.left_generated_token_ids[
                    resolved.patch.left_indices
                ].detach().cpu().tolist()
            ),
            right_token_ids=tuple(
                int(value)
                for value in edge.right_generated_token_ids[
                    resolved.patch.right_indices
                ].detach().cpu().tolist()
            ),
        )
    return assignments


def _patch_tangent(
    points: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    """Return a stable end-to-end tangent for one selected contact patch."""

    if indices.numel() < 2:
        return None
    vector = points[indices[-1]] - points[indices[0]]
    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().cpu().item()) <= 1e-6:
        return None
    return vector / norm


def _interface_orientation_loss(
    left_points: torch.Tensor,
    right_points: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    *,
    maximum_cosine: float,
) -> torch.Tensor:
    """Discourage end-on contacts using a CA patch-orientation proxy.

    A productive residue patch should generally extend across the interface
    plane rather than point directly through the opposing chain.  This term
    therefore bounds the absolute cosine between each patch tangent and the
    inter-patch approach vector.  It is topology-neutral and does not assume
    parallel helices or a particular secondary structure.
    """

    if left_indices.numel() < 2 or right_indices.numel() < 2:
        return torch.zeros(
            (), device=left_points.device, dtype=left_points.dtype
        )
    left_patch = left_points[left_indices]
    right_patch = right_points[right_indices]
    approach = right_patch.mean(dim=0) - left_patch.mean(dim=0)
    approach_norm = torch.linalg.vector_norm(approach)
    if float(approach_norm.detach().cpu().item()) <= 1e-6:
        return torch.zeros(
            (), device=left_points.device, dtype=left_points.dtype
        )
    normal = approach / approach_norm
    penalties = []
    for tangent in (
        _patch_tangent(left_points, left_indices),
        _patch_tangent(right_points, right_indices),
    ):
        if tangent is None:
            continue
        excess = torch.relu(
            torch.abs(torch.dot(tangent, normal)) - maximum_cosine
        )
        penalties.append(torch.square(excess))
    if not penalties:
        return torch.zeros(
            (), device=left_points.device, dtype=left_points.dtype
        )
    return torch.stack(penalties).mean()


def _contact_shape_loss(
    left_nearest: torch.Tensor,
    right_nearest: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    *,
    target_distance: float,
) -> torch.Tensor:
    """Prefer a continuous, uniformly deep patch over a point contact."""

    selected = torch.cat(
        (left_nearest[left_indices], right_nearest[right_indices])
    )
    if selected.numel() < 2:
        return torch.zeros((), device=selected.device, dtype=selected.dtype)
    # The coverage cutoff is an upper bound, not an optimum.  A shallow
    # target band one Angstrom inside that cutoff prevents the objective from
    # becoming flat as soon as a marginal contact is formed.
    preferred = max(3.5, target_distance - 1.0)
    tolerance = 0.5
    depth_excess = torch.relu(
        torch.abs(selected - preferred) - tolerance
    ) / max(target_distance, 1e-6)
    normalized = (selected - selected.mean()) / max(target_distance, 1e-6)
    return torch.mean(torch.square(normalized)) + torch.mean(
        torch.square(depth_excess)
    )


def _backbone_geometry_loss(
    points: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    target_distance: float,
    tolerance: float,
) -> torch.Tensor:
    """Protect local CA geometry from per-token guidance collapse."""

    losses = []
    for run in _contiguous_token_runs(token_ids):
        run_points = points[run]
        if run_points.shape[0] < 2:
            continue
        distances = torch.linalg.vector_norm(
            run_points[1:] - run_points[:-1],
            dim=-1,
        )
        excess = torch.relu(torch.abs(distances - target_distance) - tolerance)
        losses.append(
            functional.smooth_l1_loss(
                excess,
                torch.zeros_like(excess),
                reduction="mean",
                beta=0.5,
            )
        )
    if not losses:
        return torch.zeros((), device=points.device, dtype=points.dtype)
    return torch.stack(losses).mean()


def _junction_geometry_loss(
    xyz: torch.Tensor,
    pairs: torch.Tensor | None,
    *,
    target_distance: float,
    tolerance: float,
) -> torch.Tensor:
    """Protect every guided peptide edge during interface packing."""

    if pairs is None or pairs.numel() == 0:
        return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    pairs = pairs.to(dtype=torch.long, device=xyz.device)
    distances = torch.linalg.vector_norm(
        xyz[pairs[:, 0]] - xyz[pairs[:, 1]],
        dim=-1,
    )
    excess = torch.relu(torch.abs(distances - target_distance) - tolerance)
    return functional.smooth_l1_loss(
        excess,
        torch.zeros_like(excess),
        reduction="mean",
        beta=0.5,
    )


def _source_interface_means(
    values: list[torch.Tensor],
    edges: tuple[GraphInterfaceEdge, ...],
) -> torch.Tensor:
    source_ids = []
    for edge in edges:
        if edge.source_interface_id not in source_ids:
            source_ids.append(edge.source_interface_id)
    return torch.stack(
        [
            torch.stack(
                [
                    value
                    for value, edge in zip(values, edges, strict=True)
                    if edge.source_interface_id == source_id
                ]
            ).mean()
            for source_id in source_ids
        ]
    )


def graph_interface_energy(
    coordinates: torch.Tensor,
    topology: GraphInterfaceTopology,
    config: GraphInterfaceGuidanceConfig,
    *,
    target_ca_distance_override: float | None = None,
    patch_assignments: (
        dict[str, GraphInterfacePatchAssignment] | None
    ) = None,
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
    orientations = []
    shapes = []
    backbones = []
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
    patch_occupancies: list[tuple[str, int, torch.Tensor]] = []
    for edge in topology.edges:
        left_points = xyz[edge.left_generated_ca_mask]
        right_points = xyz[edge.right_generated_ca_mask]
        distance_matrix = torch.cdist(
            left_points,
            right_points,
        )
        distances = distance_matrix.flatten()
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
            # Top-k pair attraction alone can satisfy many requested pairs
            # through one residue.  Resolve one reciprocal contiguous patch
            # and use that same assignment for every packing term and for the
            # subsequent local-rigid motion proposal.
            resolved_patch = _resolve_edge_patch(
                edge,
                distance_matrix,
                config,
                target_ca_distance_override=target_ca_distance_override,
                assignment=(
                    patch_assignments.get(edge.edge_id)
                    if patch_assignments is not None
                    else None
                ),
            )
            paired_patch = resolved_patch.patch
            target_ca_distance = resolved_patch.target_ca_distance
            left_count = resolved_patch.left_count
            right_count = resolved_patch.right_count
            continuity_target = resolved_patch.continuity_target
            pair_count = resolved_patch.pair_count

            # All local packing terms below use this same reciprocal patch.
            # Only clash detection intentionally remains global.
            selected_pair_count = min(
                pair_count,
                paired_patch.distances.numel(),
            )
            selected = torch.topk(
                paired_patch.distances.flatten(),
                k=selected_pair_count,
                largest=False,
            ).values
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

            left_nearest = paired_patch.left_nearest
            right_nearest = paired_patch.right_nearest
            for token_id, occupancy in zip(
                edge.left_generated_token_ids[
                    paired_patch.left_indices
                ].detach().cpu().tolist(),
                torch.sigmoid(
                    (target_ca_distance - left_nearest)
                    / config.continuity_softness
                ),
                strict=True,
            ):
                patch_occupancies.append(
                    (f"{edge.edge_id}:left", int(token_id), occupancy)
                )
            for token_id, occupancy in zip(
                edge.right_generated_token_ids[
                    paired_patch.right_indices
                ].detach().cpu().tolist(),
                torch.sigmoid(
                    (target_ca_distance - right_nearest)
                    / config.continuity_softness
                ),
                strict=True,
            ):
                patch_occupancies.append(
                    (f"{edge.edge_id}:right", int(token_id), occupancy)
                )
            left_selected = torch.topk(
                left_nearest,
                k=min(left_count, left_nearest.numel()),
                largest=False,
            ).values
            right_selected = torch.topk(
                right_nearest,
                k=min(right_count, right_nearest.numel()),
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
            continuity_targets.append(
                torch.tensor(
                    continuity_target,
                    device=xyz.device,
                    dtype=torch.long,
                )
            )
            left_continuity = _paired_window_continuity_loss(
                left_nearest,
                target_distance=target_ca_distance,
                target_count=min(continuity_target, left_count),
                softness=config.continuity_softness,
            )
            right_continuity = _paired_window_continuity_loss(
                right_nearest,
                target_distance=target_ca_distance,
                target_count=min(continuity_target, right_count),
                softness=config.continuity_softness,
            )
            continuities.append(
                0.5 * (left_continuity + right_continuity)
            )
            contiguous_left_counts.append(
                _maximum_contiguous_covered(
                    left_nearest,
                    edge.left_generated_token_ids[
                        paired_patch.left_indices
                    ],
                    target_distance=target_ca_distance,
                )
            )
            contiguous_right_counts.append(
                _maximum_contiguous_covered(
                    right_nearest,
                    edge.right_generated_token_ids[
                        paired_patch.right_indices
                    ],
                    target_distance=target_ca_distance,
                )
            )
            orientations.append(
                _interface_orientation_loss(
                    left_points,
                    right_points,
                    paired_patch.left_indices,
                    paired_patch.right_indices,
                    maximum_cosine=(
                        config.maximum_tangent_normal_cosine
                    ),
                )
            )
            shapes.append(
                _contact_shape_loss(
                    left_nearest,
                    right_nearest,
                    torch.arange(
                        left_nearest.numel(),
                        device=left_nearest.device,
                    ),
                    torch.arange(
                        right_nearest.numel(),
                        device=right_nearest.device,
                    ),
                    target_distance=target_ca_distance,
                )
            )
            backbones.append(
                0.5
                * (
                    _backbone_geometry_loss(
                        left_points,
                        edge.left_generated_token_ids,
                        target_distance=config.backbone_ca_distance,
                        tolerance=config.backbone_ca_tolerance,
                    )
                    + _backbone_geometry_loss(
                        right_points,
                        edge.right_generated_token_ids,
                        target_distance=config.backbone_ca_distance,
                        tolerance=config.backbone_ca_tolerance,
                    )
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
            orientations.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
            )
            shapes.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
            )
            backbones.append(
                torch.zeros((), device=xyz.device, dtype=xyz.dtype)
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
    participant_token_values: dict[tuple[str, int], list[torch.Tensor]] = {}
    for participant_id, token_id, occupancy in patch_occupancies:
        participant_token_values.setdefault(
            (participant_id, token_id),
            [],
        ).append(
            occupancy
        )
    token_participant_values: dict[int, list[torch.Tensor]] = {}
    for (
        (_participant_id, token_id),
        occupancies,
    ) in participant_token_values.items():
        token_participant_values.setdefault(token_id, []).append(
            torch.stack(occupancies).max()
        )
    exclusivity_penalties = [
        torch.square(torch.relu(torch.stack(values).sum() - 1.0))
        for values in token_participant_values.values()
        if len(values) > 1
    ]
    patch_exclusivity = (
        torch.stack(exclusivity_penalties).mean()
        if exclusivity_penalties
        else torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    )

    global_safety_clash = torch.zeros(
        (), device=xyz.device, dtype=xyz.dtype
    )
    minimum_global_safety_distance = torch.tensor(
        float("inf"), device=xyz.device, dtype=xyz.dtype
    )
    if (
        topology.guided_ca_mask is not None
        and topology.safety_ca_mask is not None
        and topology.safety_exclusions is not None
    ):
        safety_distances = torch.cdist(
            xyz[topology.guided_ca_mask],
            xyz[topology.safety_ca_mask],
        )
        exclusions = topology.safety_exclusions.to(
            dtype=torch.bool,
            device=xyz.device,
        )
        if exclusions.shape != safety_distances.shape:
            raise ValueError(
                "Global interface safety exclusions have inconsistent shape"
            )
        eligible = ~exclusions
        if torch.any(eligible):
            eligible_distances = safety_distances[eligible]
            minimum_global_safety_distance = eligible_distances.min()
            global_overlap = torch.relu(
                config.clash_ca_distance - eligible_distances
            )
            global_safety_clash = torch.sum(torch.square(global_overlap)) / max(
                int(topology.guided_ca_mask.sum().item()),
                1,
            )

    attraction = _balanced_source_mean(attractions, topology.edges)
    coverage = _balanced_source_mean(coverages, topology.edges)
    continuity = _balanced_source_mean(continuities, topology.edges)
    orientation = _balanced_source_mean(orientations, topology.edges)
    shape = _balanced_source_mean(shapes, topology.edges)
    junction = _junction_geometry_loss(
        xyz,
        topology.junction_ca_pairs,
        target_distance=config.backbone_ca_distance,
        tolerance=config.backbone_ca_tolerance,
    )
    backbone = _balanced_source_mean(backbones, topology.edges) + junction
    clash = (
        _balanced_source_mean(clashes, topology.edges)
        + global_safety_clash
    )
    distance = _balanced_source_mean(distance_terms, topology.edges)
    per_edge_total = torch.stack(
        [
            config.weight * edge_attraction
            + config.coverage_weight * edge_coverage
            + config.continuity_weight * edge_continuity
            + config.orientation_weight * edge_orientation
            + config.shape_weight * edge_shape
            + config.backbone_weight * edge_backbone
            + config.clash_weight * edge_clash
            + config.distance_weight * edge_distance
            for (
                edge_attraction,
                edge_coverage,
                edge_continuity,
                edge_orientation,
                edge_shape,
                edge_backbone,
                edge_clash,
                edge_distance,
            ) in zip(
                attractions,
                coverages,
                continuities,
                orientations,
                shapes,
                backbones,
                clashes,
                distance_terms,
                strict=True,
            )
        ]
    )
    source_totals = _source_interface_means(
        list(per_edge_total),
        topology.edges,
    )
    interface_balance = (
        torch.logsumexp(source_totals, dim=0)
        - math.log(source_totals.numel())
        if source_totals.numel() > 1
        else torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    )
    return GraphInterfaceEnergy(
        total=(
            config.weight * attraction
            + config.coverage_weight * coverage
            + config.continuity_weight * continuity
            + config.orientation_weight * orientation
            + config.shape_weight * shape
            + config.backbone_weight * backbone
            + config.interface_balance_weight * interface_balance
            + config.patch_exclusivity_weight * patch_exclusivity
            + config.clash_weight * clash
            + config.distance_weight * distance
        ),
        attraction=attraction,
        coverage=coverage,
        continuity=continuity,
        orientation=orientation,
        shape=shape,
        backbone=backbone,
        junction=junction,
        interface_balance=interface_balance,
        patch_exclusivity=patch_exclusivity,
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
        per_edge_orientation=torch.stack(orientations),
        per_edge_shape=torch.stack(shapes),
        per_edge_backbone=torch.stack(backbones),
        per_edge_total=per_edge_total,
        per_source_total=source_totals,
        global_safety_clash=global_safety_clash,
        minimum_global_safety_distance=minimum_global_safety_distance,
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
    terminal_weight_floor: float = 0.0,
) -> float:
    """Ramp in smoothly, then retain a terminal hold for required packing.

    A zero terminal field allowed the denoiser to erase an interface during
    the final diffusion steps.  ``terminal_weight_floor`` preserves the old
    schedule when explicitly set to zero, while required-interface defaults
    now keep a bounded correction active through the final state.
    """

    if progress <= start_fraction:
        return 0.0
    if progress >= end_fraction:
        return terminal_weight_floor
    phase = (progress - start_fraction) / (end_fraction - start_fraction)
    pulse = math.sin(math.pi * phase)
    if phase <= 0.5:
        return pulse
    return terminal_weight_floor + (1.0 - terminal_weight_floor) * pulse


def scheduled_interface_ca_distance(
    progress: float,
    config: GraphInterfaceGuidanceConfig,
) -> float:
    """Coarse-to-fine capture radius used inside the sampler.

    Early denoising receives a broad, gentle basin that can find the correct
    neighbouring surface.  The basin contracts continuously to the final
    packing target before the terminal hold/polish phase.  This is internal
    model behaviour; users do not have to guess a capture radius.
    """

    if progress <= config.start_fraction:
        return config.capture_ca_distance
    if progress >= config.end_fraction:
        return config.target_ca_distance
    fraction = (progress - config.start_fraction) / (
        config.end_fraction - config.start_fraction
    )
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return (
        config.capture_ca_distance
        + smooth
        * (config.target_ca_distance - config.capture_ca_distance)
    )


def adaptive_graph_interface_phase(
    energy: GraphInterfaceEnergy,
    config: GraphInterfaceGuidanceConfig,
) -> str:
    """Choose capture, expansion or polish from observed interface quality.

    Time remains a coarse annealing envelope, but is no longer authoritative:
    a distant patch stays in capture, a narrow patch stays in expansion, and
    only a reciprocal patch meeting its residue-scale contract enters polish.
    """

    if bool(
        torch.any(
            energy.mean_selected_distances
            > config.target_ca_distance + 2.0
        ).item()
    ):
        return "capture"
    coverage_satisfied = bool(
        torch.all(
            energy.covered_left_residues
            >= energy.target_residues_per_side
        ).item()
        and torch.all(
            energy.covered_right_residues
            >= energy.target_residues_per_side
        ).item()
    )
    continuity_satisfied = bool(
        torch.all(
            energy.contiguous_left_residues
            >= energy.target_contiguous_residues_per_side
        ).item()
        and torch.all(
            energy.contiguous_right_residues
            >= energy.target_contiguous_residues_per_side
        ).item()
    )
    if not (coverage_satisfied and continuity_satisfied):
        return "expand"
    return "polish"


def _phase_guidance_config(
    config: GraphInterfaceGuidanceConfig,
    phase: str,
) -> GraphInterfaceGuidanceConfig:
    """Resolve internal objective weights for the observed packing phase."""

    if phase == "capture":
        return replace(
            config,
            coverage_weight=0.75 * config.coverage_weight,
            continuity_weight=0.60 * config.continuity_weight,
            orientation_weight=0.50 * config.orientation_weight,
            shape_weight=0.50 * config.shape_weight,
        )
    if phase == "expand":
        return replace(
            config,
            weight=0.85 * config.weight,
            coverage_weight=1.25 * config.coverage_weight,
            continuity_weight=1.35 * config.continuity_weight,
            shape_weight=0.80 * config.shape_weight,
        )
    if phase == "polish":
        return replace(
            config,
            weight=0.60 * config.weight,
            continuity_weight=1.15 * config.continuity_weight,
            orientation_weight=1.50 * config.orientation_weight,
            shape_weight=1.50 * config.shape_weight,
            maximum_token_step=0.50 * config.maximum_token_step,
            maximum_patch_rotation_degrees=(
                0.50 * config.maximum_patch_rotation_degrees
            ),
        )
    raise ValueError(f"Unknown graph-interface phase {phase!r}")


def _phase_target_ca_distance(
    phase: str,
    scheduled_target: float,
    config: GraphInterfaceGuidanceConfig,
) -> float:
    if phase == "capture":
        return max(scheduled_target, config.target_ca_distance + 2.0)
    if phase == "expand":
        return min(
            max(scheduled_target, config.target_ca_distance),
            config.target_ca_distance + 2.0,
        )
    return config.target_ca_distance


def graph_interface_quality_satisfied(
    energy: GraphInterfaceEnergy,
    *,
    clash_ca_distance: float | None = None,
    config: GraphInterfaceGuidanceConfig | None = None,
) -> bool:
    """Return whether every runtime packing contract is met.

    Coverage alone is not a packed interface.  The terminal contract also
    rejects end-on, corrugated, locally damaged and multiply re-used patches.
    These thresholds are internal defaults, so ordinary users are not asked
    to invent geometric cutoffs.
    """

    effective = config or GraphInterfaceGuidanceConfig()
    clash_threshold = (
        clash_ca_distance
        if clash_ca_distance is not None
        else effective.clash_ca_distance
    )
    return bool(
        torch.all(
            energy.covered_left_residues
            >= energy.target_residues_per_side
        ).item()
        and torch.all(
            energy.covered_right_residues
            >= energy.target_residues_per_side
        ).item()
        and torch.all(
            energy.contiguous_left_residues
            >= energy.target_contiguous_residues_per_side
        ).item()
        and torch.all(
            energy.contiguous_right_residues
            >= energy.target_contiguous_residues_per_side
        ).item()
        and torch.all(
            energy.per_edge_orientation
            <= effective.maximum_orientation_loss
        ).item()
        and torch.all(
            energy.per_edge_shape <= effective.maximum_shape_loss
        ).item()
        and torch.all(
            energy.per_edge_backbone <= effective.maximum_backbone_loss
        ).item()
        and bool(energy.junction <= effective.maximum_backbone_loss)
        and bool(
            energy.patch_exclusivity
            <= effective.maximum_patch_exclusivity_loss
        )
        and torch.all(
            energy.minimum_distances >= clash_threshold
        ).item()
        and bool(
            energy.minimum_global_safety_distance >= clash_threshold
        )
    )


def graph_interface_patch_capture_satisfied(
    energy: GraphInterfaceEnergy,
) -> bool:
    """Return whether a reciprocal patch is ready for identity locking.

    A timestep alone cannot establish patch identity: before contact forms,
    the denoiser may move the physically relevant sequence window.  Lock only
    after the selected window meets the final residue-coverage and continuity
    contracts on both sides.  Orientation and shape remain optimization
    targets after capture and are intentionally not prerequisites here.
    """

    return bool(
        torch.all(
            energy.covered_left_residues
            >= energy.target_residues_per_side
        ).item()
        and torch.all(
            energy.covered_right_residues
            >= energy.target_residues_per_side
        ).item()
        and torch.all(
            energy.contiguous_left_residues
            >= energy.target_contiguous_residues_per_side
        ).item()
        and torch.all(
            energy.contiguous_right_residues
            >= energy.target_contiguous_residues_per_side
        ).item()
    )


def graph_interface_proposal_acceptable(
    before: GraphInterfaceEnergy,
    after: GraphInterfaceEnergy,
    config: GraphInterfaceGuidanceConfig,
) -> bool:
    """Hard acceptance contract for a timestep packing proposal.

    A lower weighted energy is insufficient if it buys attraction by creating
    a new backbone collision.  Safe states must stay above the CA exclusion
    radius; already unsafe states must monotonically improve both their worst
    distance and aggregate global clash energy.
    """

    if not (
        torch.isfinite(after.total)
        and float(after.total.detach().cpu().item())
        < float(before.total.detach().cpu().item()) - 1e-10
    ):
        return False

    def minimum_not_worse(
        initial: torch.Tensor,
        candidate: torch.Tensor,
    ) -> bool:
        initial_minimum = float(initial.min().detach().cpu().item())
        candidate_minimum = float(candidate.min().detach().cpu().item())
        required = (
            config.clash_ca_distance
            if initial_minimum >= config.clash_ca_distance
            else initial_minimum
        )
        return candidate_minimum >= required - 1e-6

    if not minimum_not_worse(
        before.minimum_distances,
        after.minimum_distances,
    ):
        return False
    if not minimum_not_worse(
        before.minimum_global_safety_distance.reshape(1),
        after.minimum_global_safety_distance.reshape(1),
    ):
        return False
    if before.per_source_total.shape != after.per_source_total.shape:
        return False
    before_sources = before.per_source_total.detach()
    after_sources = after.per_source_total.detach()
    worst_before = float(before_sources.max().cpu().item())
    worst_after = float(after_sources.max().cpu().item())
    allowed_regression = max(
        config.maximum_source_regression_absolute,
        abs(worst_before) * config.maximum_source_regression_fraction,
    )
    if (
        worst_after
        > worst_before + config.maximum_source_regression_absolute
    ):
        return False
    if bool(torch.any(after_sources - before_sources > allowed_regression)):
        return False
    before_junction = float(before.junction.detach().cpu().item())
    after_junction = float(after.junction.detach().cpu().item())
    junction_limit = (
        config.maximum_backbone_loss
        if before_junction <= config.maximum_backbone_loss
        else before_junction
    )
    if after_junction > junction_limit + 1e-8:
        return False
    before_exclusivity = float(
        before.patch_exclusivity.detach().cpu().item()
    )
    after_exclusivity = float(
        after.patch_exclusivity.detach().cpu().item()
    )
    exclusivity_limit = (
        config.maximum_patch_exclusivity_loss
        if before_exclusivity <= config.maximum_patch_exclusivity_loss
        else before_exclusivity
    )
    if after_exclusivity > exclusivity_limit + 1e-8:
        return False
    return bool(
        after.global_safety_clash
        <= before.global_safety_clash + 1e-8
    )

def graph_interface_energy_diagnostics(
    energy: GraphInterfaceEnergy,
) -> dict[str, Any]:
    """Serialize one energy evaluated on an actual coordinate state.

    Guidance-step diagnostics are evaluated before applying that step.  The
    sampler also needs this state-only representation after finalization so
    audits never mistake the last proposal's pre-update energy for the
    structure that was written to disk.
    """

    scalar_fields = (
        "total",
        "attraction",
        "coverage",
        "continuity",
        "orientation",
        "shape",
        "backbone",
        "junction",
        "interface_balance",
        "patch_exclusivity",
        "clash",
        "global_safety_clash",
        "minimum_global_safety_distance",
        "distance",
    )
    vector_fields = (
        "minimum_distances",
        "mean_selected_distances",
        "covered_left_residues",
        "covered_right_residues",
        "target_residues_per_side",
        "target_contiguous_residues_per_side",
        "contiguous_left_residues",
        "contiguous_right_residues",
        "per_edge_orientation",
        "per_edge_shape",
        "per_edge_backbone",
        "per_edge_total",
        "per_source_total",
    )
    payload = {
        name: float(getattr(energy, name).detach().cpu().item())
        for name in scalar_fields
    }
    payload["energy"] = payload.pop("total")
    payload.update(
        {
            name: getattr(energy, name).detach().cpu().tolist()
            for name in vector_fields
        }
    )
    return payload


def _rotation_from_vector(vector: torch.Tensor) -> torch.Tensor:
    """Convert one axis-angle vector to a proper rotation matrix."""

    angle = torch.linalg.vector_norm(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    if float(angle.detach().cpu().item()) <= 1e-10:
        return identity
    axis = vector / angle
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    return (
        identity
        + torch.sin(angle) * skew
        + (1.0 - torch.cos(angle)) * (skew @ skew)
    )


def _fit_patch_rigid_descent(
    points: torch.Tensor,
    gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit an infinitesimal SE(3) descent field to one CA patch.

    The fit is a least-squares projection of ``-dE/dx`` onto translation plus
    rotation around the patch centroid.  It keeps the selected backbone patch
    coherent while still allowing the whole patch to tilt toward its partner.
    """

    center = points.mean(dim=0)
    descent = -gradients
    translation = descent.mean(dim=0)
    centered = points - center
    residual = descent - translation
    identity = torch.eye(3, dtype=points.dtype, device=points.device)
    inertia = torch.zeros((3, 3), dtype=points.dtype, device=points.device)
    torque = torch.zeros(3, dtype=points.dtype, device=points.device)
    for radius, target in zip(centered, residual, strict=True):
        inertia = inertia + (
            torch.dot(radius, radius) * identity
            - torch.outer(radius, radius)
        )
        torque = torque + torch.linalg.cross(radius, target)
    if points.shape[0] < 2 or float(
        torch.linalg.matrix_norm(inertia).detach().cpu().item()
    ) <= 1e-10:
        rotation_vector = torch.zeros_like(translation)
    else:
        rotation_vector = torch.linalg.pinv(inertia) @ torque
    return center, translation, rotation_vector


def _selected_patch_token_groups(
    coordinates: torch.Tensor,
    topology: GraphInterfaceTopology,
    config: GraphInterfaceGuidanceConfig,
    *,
    target_ca_distance_override: float,
    patch_assignments: (
        dict[str, GraphInterfacePatchAssignment] | None
    ) = None,
) -> tuple[torch.Tensor, ...]:
    """Recover the exact reciprocal windows used by the current energy."""

    xyz = coordinates[0]
    groups = []
    for edge in topology.edges:
        if not (
            edge.requested_contact_count > 0
            or edge.requested_residues_per_side > 0
            or edge.requested_contiguous_residues_per_side > 0
            or edge.automatic_quality
        ):
            continue
        left_points = xyz[edge.left_generated_ca_mask]
        right_points = xyz[edge.right_generated_ca_mask]
        resolved = _resolve_edge_patch(
            edge,
            torch.cdist(left_points, right_points),
            config,
            target_ca_distance_override=target_ca_distance_override,
            assignment=(
                patch_assignments.get(edge.edge_id)
                if patch_assignments is not None
                else None
            ),
        )
        groups.extend(
            (
                edge.left_generated_token_ids[
                    resolved.patch.left_indices
                ],
                edge.right_generated_token_ids[
                    resolved.patch.right_indices
                ],
            )
        )
    return tuple(groups)


def _patch_rigid_token_displacements(
    coordinates: torch.Tensor,
    gradient: torch.Tensor,
    features: dict[str, Any],
    topology: GraphInterfaceTopology,
    config: GraphInterfaceGuidanceConfig,
    *,
    window: float,
    gradient_boost: float,
    target_ca_distance_override: float,
    patch_assignments: (
        dict[str, GraphInterfacePatchAssignment] | None
    ) = None,
) -> tuple[dict[int, torch.Tensor], int, float]:
    """Build simultaneous local-rigid proposals for all reciprocal patches."""

    atom_to_token = torch.as_tensor(
        features["atom_to_token_map"],
        dtype=torch.long,
        device=coordinates.device,
    )
    token_chain_ids = torch.as_tensor(
        features["asym_id"],
        dtype=torch.long,
        device=coordinates.device,
    )
    token_positions = torch.as_tensor(
        features.get(
            "residue_index",
            torch.arange(
                token_chain_ids.numel(), device=coordinates.device
            ),
        ),
        dtype=torch.long,
        device=coordinates.device,
    )
    guided_tokens = torch.unique(
        atom_to_token[topology.generated_atom_mask]
    )
    ca_mask = topology.guided_ca_mask
    if ca_mask is None:
        # Manually constructed unit-test topologies predate the global safety
        # fields.  Their generated atoms are all CA-only.
        ca_mask = topology.generated_atom_mask
    groups = _selected_patch_token_groups(
        coordinates,
        topology,
        config,
        target_ca_distance_override=target_ca_distance_override,
        patch_assignments=patch_assignments,
    )
    proposals: dict[int, list[torch.Tensor]] = {}
    maximum_rotation_degrees = 0.0
    patch_count = 0
    for group in groups:
        group = torch.unique(group)
        group_ca_mask = ca_mask & torch.isin(atom_to_token, group)
        if not torch.any(group_ca_mask):
            continue
        points = coordinates[0, group_ca_mask]
        point_gradients = gradient[0, group_ca_mask]
        center, translation, rotation_vector = _fit_patch_rigid_descent(
            points,
            point_gradients,
        )
        translation = translation * (window * gradient_boost)
        rotation_vector = rotation_vector * (window * gradient_boost)
        rotation_norm = torch.linalg.vector_norm(rotation_vector)
        maximum_rotation = math.radians(
            config.maximum_patch_rotation_degrees
        )
        if float(rotation_norm.detach().cpu().item()) > maximum_rotation > 0:
            rotation_vector = rotation_vector * (
                maximum_rotation / rotation_norm
            )
        elif maximum_rotation == 0.0:
            rotation_vector = torch.zeros_like(rotation_vector)
        group_chain_ids = torch.unique(token_chain_ids[group])
        if group_chain_ids.numel() != 1:
            raise ValueError(
                "One reciprocal interface patch spans multiple chains"
            )
        chain_id = group_chain_ids[0]
        same_chain_tokens = guided_tokens[
            token_chain_ids[guided_tokens] == chain_id
        ]
        group_positions = token_positions[group]
        patch_tokens: list[tuple[int, float, torch.Tensor]] = []
        for token_id_tensor in same_chain_tokens:
            token_id = int(token_id_tensor.item())
            sequence_distance = int(
                torch.min(
                    torch.abs(
                        token_positions[token_id_tensor] - group_positions
                    )
                ).item()
            )
            if sequence_distance > config.patch_blend_radius:
                continue
            blend = 1.0 - sequence_distance / float(
                config.patch_blend_radius + 1
            )
            token_mask = (
                topology.generated_atom_mask
                & (atom_to_token == token_id_tensor)
            )
            original = coordinates[0, token_mask]
            if original.numel() == 0:
                continue
            patch_tokens.append((token_id, blend, original))

        # One patch is one local rigid-body proposal.  Its trust-region scale
        # must therefore be shared by every residue: independently clipping
        # end residues of a rotating patch would silently bend the backbone.
        common_scale = 1.0
        for _ in range(3):
            maximum_displacement = coordinates.new_zeros(())
            for _, blend, original in patch_tokens:
                rotation = _rotation_from_vector(
                    rotation_vector * (blend * common_scale)
                )
                transformed = (
                    (original - center) @ rotation.T
                    + center
                    + translation * (blend * common_scale)
                )
                maximum_displacement = torch.maximum(
                    maximum_displacement,
                    torch.linalg.vector_norm(
                        transformed - original,
                        dim=-1,
                    ).max(),
                )
            observed = float(
                maximum_displacement.detach().cpu().item()
            )
            if observed <= config.maximum_token_step + 1e-8:
                break
            common_scale *= config.maximum_token_step / observed

        maximum_rotation_degrees = max(
            maximum_rotation_degrees,
            math.degrees(
                float(
                    (torch.linalg.vector_norm(rotation_vector) * common_scale)
                    .detach()
                    .cpu()
                    .item()
                )
            ),
        )
        for token_id, blend, original in patch_tokens:
            rotation = _rotation_from_vector(
                rotation_vector * (blend * common_scale)
            )
            transformed = (
                (original - center) @ rotation.T
                + center
                + translation * (blend * common_scale)
            )
            displacement = transformed - original
            proposals.setdefault(token_id, []).append(displacement)
        patch_count += 1
    return (
        {
            token_id: torch.stack(displacements).mean(dim=0)
            for token_id, displacements in proposals.items()
        },
        patch_count,
        maximum_rotation_degrees,
    )


def apply_graph_interface_guidance(
    coordinates: torch.Tensor,
    features: dict[str, Any],
    topology: GraphInterfaceTopology,
    *,
    progress: float,
    config: GraphInterfaceGuidanceConfig,
    projector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    patch_state: GraphInterfacePatchState | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one bounded local-rigid step to the true projected state."""

    window = guidance_window_weight(
        progress,
        start_fraction=config.start_fraction,
        end_fraction=config.end_fraction,
        terminal_weight_floor=config.terminal_weight_floor,
    )
    if window == 0.0 or (
        config.weight == 0.0
        and config.coverage_weight == 0.0
        and config.continuity_weight == 0.0
        and config.orientation_weight == 0.0
        and config.shape_weight == 0.0
        and config.backbone_weight == 0.0
        and config.interface_balance_weight == 0.0
        and config.patch_exclusivity_weight == 0.0
        and config.clash_weight == 0.0
        and config.distance_weight == 0.0
    ):
        return coordinates, {"applied": False, "window_weight": window}
    scheduled_target = scheduled_interface_ca_distance(progress, config)
    if (
        patch_state is not None
        and patch_state.locked
        and patch_state.assignments
    ):
        patch_assignments = patch_state.assignments
    else:
        patch_assignments = resolve_graph_interface_patch_assignments(
            coordinates,
            topology,
            config,
            target_ca_distance_override=scheduled_target,
        )
        if patch_state is not None:
            patch_state.assignments = patch_assignments
    with torch.no_grad():
        phase_probe = graph_interface_energy(
            coordinates,
            topology,
            config,
            patch_assignments=patch_assignments,
        )
    adaptive_phase = adaptive_graph_interface_phase(phase_probe, config)
    effective_config = _phase_guidance_config(config, adaptive_phase)
    scheduled_target = _phase_target_ca_distance(
        adaptive_phase,
        scheduled_target,
        config,
    )
    patch_capture_satisfied = False
    with torch.enable_grad():
        proposal = coordinates.detach().clone().requires_grad_(True)
        energy = graph_interface_energy(
            proposal,
            topology,
            effective_config,
            target_ca_distance_override=scheduled_target,
            patch_assignments=patch_assignments,
        )
        if (
            patch_state is not None
            and not patch_state.locked
            and progress >= config.patch_lock_fraction
        ):
            # Judge capture using the final packing radius rather than the
            # wider coarse-capture target that drives this timestep.  A
            # distant early window must not become permanent simply because
            # diffusion passed a time threshold.
            with torch.no_grad():
                lock_energy = graph_interface_energy(
                    proposal.detach(),
                    topology,
                    effective_config,
                    patch_assignments=patch_assignments,
                )
            patch_capture_satisfied = (
                graph_interface_patch_capture_satisfied(lock_energy)
            )
            if patch_capture_satisfied:
                patch_state.locked = True
                patch_state.lock_reason = "quality_capture"
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
    quality_targets_satisfied_before = graph_interface_quality_satisfied(
        energy,
        clash_ca_distance=config.clash_ca_distance,
        config=config,
    )
    gradient_boost = 1.0
    if (
        not quality_targets_satisfied_before
        and config.unsatisfied_step_fraction > 0
    ):
        maximum_raw_step = max(
            (
                float(
                    torch.linalg.vector_norm(window * token_gradient)
                    .detach()
                    .cpu()
                    .item()
                )
                for token_gradient in token_gradients.values()
            ),
            default=0.0,
        )
        desired_step = (
            effective_config.maximum_token_step
            * effective_config.unsatisfied_step_fraction
        )
        if 1e-8 < maximum_raw_step < desired_step:
            gradient_boost = desired_step / maximum_raw_step
    rigid_displacements, rigid_patch_count, maximum_patch_rotation = (
        _patch_rigid_token_displacements(
            coordinates,
            gradient,
            features,
            topology,
            effective_config,
            window=window,
            gradient_boost=gradient_boost,
            target_ca_distance_override=scheduled_target,
            patch_assignments=patch_assignments,
        )
    )
    observed_steps = []
    for token_id in selected_tokens:
        token_index = int(token_id.item())
        token_mask = (
            topology.generated_atom_mask
            & (atom_to_token == token_id)
        )
        token_gradient = token_gradients[token_index]
        local_step = -window * gradient_boost * token_gradient
        local_norm = torch.linalg.vector_norm(
            local_step,
            dim=-1,
            keepdim=True,
        )
        local_step = local_step * torch.clamp(
            effective_config.maximum_token_step
            / local_norm.max().clamp_min(1e-8),
            max=1.0,
        )
        step = local_step
        rigid_displacement = rigid_displacements.get(token_index)
        if rigid_displacement is not None:
            step = (
                effective_config.patch_rigid_weight
                * rigid_displacement.unsqueeze(0)
                + (1.0 - effective_config.patch_rigid_weight) * local_step
            )
        # Rigid displacements were already clipped once with one shared
        # patch-wide scale.  A second per-token clip here would destroy that
        # rigid transform.  Convex blending with the bounded local step also
        # remains inside the same trust region.
        token_maximum = torch.linalg.vector_norm(step, dim=-1).max()
        maximum_observed_step = max(
            maximum_observed_step,
            float(token_maximum.detach().cpu().item()),
        )
        observed_steps.append(torch.linalg.vector_norm(step, dim=-1).mean())
        guided[:, token_mask, :] += step

    # Smoothing, overlapping interface patches and local-rigid projection can
    # rotate the raw gradient enough that a full trust-region step is not a
    # descent step.  Accept the whole multi-interface proposal atomically or
    # backtrack it; never let one edge improve by damaging another unnoticed.
    displacement = guided - coordinates
    accepted_scale = 0.0
    accepted_energy = energy
    accepted = coordinates
    for line_search_index in range(effective_config.line_search_steps):
        scale = effective_config.line_search_contraction**line_search_index
        candidate = coordinates + scale * displacement
        if projector is not None:
            candidate = projector(candidate)
        candidate_energy = graph_interface_energy(
            candidate,
            topology,
            effective_config,
            target_ca_distance_override=scheduled_target,
            patch_assignments=patch_assignments,
        )
        if graph_interface_proposal_acceptable(
            energy,
            candidate_energy,
            effective_config,
        ):
            accepted_scale = scale
            accepted_energy = candidate_energy
            accepted = candidate.detach()
            break
    maximum_observed_step *= accepted_scale
    quality_targets_satisfied = graph_interface_quality_satisfied(
        accepted_energy,
        clash_ca_distance=config.clash_ca_distance,
        config=config,
    )
    return accepted, {
        "applied": accepted_scale > 0.0,
        "proposal_accepted": accepted_scale > 0.0,
        "line_search_scale": accepted_scale,
        "window_weight": window,
        "adaptive_phase": adaptive_phase,
        "time_scheduled_target_ca_distance": (
            scheduled_interface_ca_distance(progress, config)
        ),
        "scheduled_target_ca_distance": scheduled_target,
        "patch_locked": bool(patch_state and patch_state.locked),
        "patch_lock_reason": (
            patch_state.lock_reason if patch_state is not None else None
        ),
        "patch_capture_satisfied": patch_capture_satisfied,
        "patch_assignments": {
            edge_id: {
                "left_token_ids": list(assignment.left_token_ids),
                "right_token_ids": list(assignment.right_token_ids),
            }
            for edge_id, assignment in sorted(patch_assignments.items())
        },
        "edge_count": len(topology.edges),
        "rigid_patch_count": rigid_patch_count,
        "maximum_patch_rotation_degrees": (
            maximum_patch_rotation * accepted_scale
        ),
        "quality_targets_satisfied_before": (
            quality_targets_satisfied_before
        ),
        "quality_targets_satisfied": quality_targets_satisfied,
        "gradient_boost": gradient_boost,
        "energy_before": float(energy.total.detach().cpu().item()),
        "energy": float(accepted_energy.total.detach().cpu().item()),
        "energy_after": float(
            accepted_energy.total.detach().cpu().item()
        ),
        "attraction": float(
            accepted_energy.attraction.detach().cpu().item()
        ),
        "coverage": float(accepted_energy.coverage.detach().cpu().item()),
        "continuity": float(
            accepted_energy.continuity.detach().cpu().item()
        ),
        "orientation": float(
            accepted_energy.orientation.detach().cpu().item()
        ),
        "shape": float(accepted_energy.shape.detach().cpu().item()),
        "backbone": float(accepted_energy.backbone.detach().cpu().item()),
        "junction": float(accepted_energy.junction.detach().cpu().item()),
        "interface_balance": float(
            accepted_energy.interface_balance.detach().cpu().item()
        ),
        "patch_exclusivity": float(
            accepted_energy.patch_exclusivity.detach().cpu().item()
        ),
        "clash": float(accepted_energy.clash.detach().cpu().item()),
        "global_safety_clash": float(
            accepted_energy.global_safety_clash.detach().cpu().item()
        ),
        "minimum_global_safety_distance": float(
            accepted_energy.minimum_global_safety_distance.detach().cpu().item()
        ),
        "distance": float(accepted_energy.distance.detach().cpu().item()),
        "minimum_distances": (
            accepted_energy.minimum_distances.detach().cpu().tolist()
        ),
        "mean_selected_distances": (
            accepted_energy.mean_selected_distances.detach().cpu().tolist()
        ),
        "covered_left_residues": (
            accepted_energy.covered_left_residues.detach().cpu().tolist()
        ),
        "covered_right_residues": (
            accepted_energy.covered_right_residues.detach().cpu().tolist()
        ),
        "target_residues_per_side": (
            accepted_energy.target_residues_per_side.detach().cpu().tolist()
        ),
        "target_contiguous_residues_per_side": (
            accepted_energy.target_contiguous_residues_per_side
            .detach()
            .cpu()
            .tolist()
        ),
        "contiguous_left_residues": (
            accepted_energy.contiguous_left_residues.detach().cpu().tolist()
        ),
        "contiguous_right_residues": (
            accepted_energy.contiguous_right_residues.detach().cpu().tolist()
        ),
        "per_edge_orientation": (
            accepted_energy.per_edge_orientation.detach().cpu().tolist()
        ),
        "per_edge_shape": (
            accepted_energy.per_edge_shape.detach().cpu().tolist()
        ),
        "per_edge_backbone": (
            accepted_energy.per_edge_backbone.detach().cpu().tolist()
        ),
        "per_edge_total": (
            accepted_energy.per_edge_total.detach().cpu().tolist()
        ),
        "per_source_total": (
            accepted_energy.per_source_total.detach().cpu().tolist()
        ),
        "maximum_token_step": maximum_observed_step,
        "mean_token_step": float(
            torch.stack(observed_steps).mean().detach().cpu().item()
        ) * accepted_scale,
    }


__all__ = [
    "GraphInterfaceEdge",
    "GraphInterfaceEnergy",
    "GraphInterfaceGuidanceConfig",
    "GraphInterfaceTopology",
    "apply_graph_interface_guidance",
    "build_graph_interface_topology",
    "graph_interface_energy",
    "graph_interface_energy_diagnostics",
    "graph_interface_patch_capture_satisfied",
    "graph_interface_proposal_acceptable",
    "graph_interface_quality_satisfied",
    "guidance_window_weight",
    "resolve_graph_interface_patch_assignments",
    "scheduled_interface_ca_distance",
]
