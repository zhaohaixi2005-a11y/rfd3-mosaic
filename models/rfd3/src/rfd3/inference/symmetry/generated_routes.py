"""Spatial ownership of generated polymer runs, independent of atom clashes.

Reference corridors follow the current fixed-anchor poses. Positive clearance
and ALL competing corridors remove the zero-loss tie at a symmetric centre.
This is an explicit spatial design constraint, not a knot/link invariant.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import torch


def point_segment_distances(points, starts, ends):
    direction = ends - starts
    relative = points[:, None, :] - starts[None, :, :]
    fraction = (relative * direction[None, :, :]).sum(-1)
    fraction = fraction / direction.square().sum(-1).clamp_min(1e-12)
    closest = starts[None, :, :] + fraction.clamp(0, 1)[..., None] * direction[None, :, :]
    return torch.linalg.vector_norm(points[:, None, :] - closest, dim=-1)


def generated_route_deficits(coordinates, topology, *, clearance=3.2, anchor_taper=2.0):
    """Return Å deficits at generated CAs and all adjoining bond midpoints.

    s is sequence position from the left fixed anchor (0), right anchor n+1.
    m(s) = clearance * min(1, s/taper, (n+1-s)/taper).
    v(s,j) = max(0, m(s) + distance(x,S_own) - distance(x,S_j)).
    Fixed endpoints are excluded. Sampling midpoints also checks between CAs;
    it does not certify the complete continuous curve or an all-atom surface.
    """
    terms = []
    for index, run in enumerate(topology.generated_runs):
        competitors = [other for j, other in enumerate(topology.generated_runs)
                       if j != index and other.asym_id != run.asym_id]
        if not competitors or not len(run.generated_ca_atom_indices):
            continue
        left = coordinates[run.left_anchor_ca_atom_index]
        right = coordinates[run.right_anchor_ca_atom_index]
        generated = coordinates[run.generated_ca_atom_indices]
        path = torch.cat((left[None], generated, right[None]))
        midpoints = (path[:-1] + path[1:]) * 0.5
        points = torch.cat((generated, midpoints))
        n = len(generated)
        positions = torch.cat((
            torch.arange(1, n + 1, dtype=coordinates.dtype, device=coordinates.device),
            torch.arange(n + 1, dtype=coordinates.dtype, device=coordinates.device) + 0.5,
        ))
        margin = clearance * torch.minimum(positions, n + 1 - positions).div(anchor_taper).clamp(max=1)
        own = point_segment_distances(points, left[None], right[None])
        starts = torch.stack([coordinates[r.left_anchor_ca_atom_index] for r in competitors])
        ends = torch.stack([coordinates[r.right_anchor_ca_atom_index] for r in competitors])
        others = point_segment_distances(points, starts, ends)
        terms.append(torch.relu(margin[:, None] + own - others).flatten())
    return torch.cat(terms) if terms else coordinates.new_empty(0)


def route_deficits_from_config(coordinates, topology, config):
    return generated_route_deficits(
        coordinates, topology, clearance=config.routing_clearance,
        anchor_taper=config.routing_anchor_taper_residues,
    )


def apply_generated_route_guidance(
    coordinates, topology, *, progress, config,
    projector: Callable[[torch.Tensor], torch.Tensor], iterations=2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Correct each CLEAN prediction, with fixed seeds and geometry protected.

    Normalize the route-only direction before bounding displacement: averaging
    over a longer chain or more symmetry copies must not erase its effect.
    Early corrections have at most one Å/residue; the final prediction uses
    the core's 0.2 Å bound. No clean-geometry projection is applied to noisy X_t.
    """
    from .scaffold_core_guidance import scaffold_geometry_guard

    if not 0 <= progress <= 1 or iterations < 0:
        raise ValueError("Invalid generated route guidance schedule")
    if config.routing_ownership_weight <= 0:
        return coordinates, {"applied": False, "reason": "disabled", "steps": []}
    result = coordinates.detach()
    maximum_step = config.maximum_token_step + (1.0 - progress) * (1.0 - config.maximum_token_step)
    maximum_step = max(config.maximum_token_step, maximum_step)

    def values(x):
        return route_deficits_from_config(x[0], topology, config)

    def summary(v):
        return {
            "checked_constraints": v.numel(),
            "violated_constraints": int((v > config.routing_tolerance).sum()),
            "maximum_excess_angstrom": float(v.max()) if v.numel() else 0.0,
            "squared_excess_angstrom2": float(v.square().sum()),
        }

    with torch.no_grad():
        initial = summary(values(result))
    steps = []
    for _ in range(iterations):
        with torch.enable_grad():
            source = result.detach().requires_grad_(True)
            before = values(source)
            if not before.numel() or not torch.any(before > config.routing_tolerance):
                break
            loss = before.square().sum()
            gradient = torch.autograd.grad(loss, source)[0][0]
        before = before.detach()
        token_step = result.new_zeros((len(topology.generated_token_mask), 3))
        token_step.index_add_(0, topology.atom_to_token, -gradient)
        token_step[~topology.generated_token_mask] = 0
        # Smooth translations along the polymer, always restoring fixed tokens.
        for _ in range(2):
            pairs = topology.adjacent_token_pairs
            accumulated = token_step.clone()
            counts = torch.ones_like(token_step[:, 0])
            if len(pairs):
                a, b = pairs.unbind(1)
                accumulated.index_add_(0, a, token_step[b])
                accumulated.index_add_(0, b, token_step[a])
                counts.index_add_(0, a, torch.ones_like(a, dtype=counts.dtype))
                counts.index_add_(0, b, torch.ones_like(b, dtype=counts.dtype))
            token_step = accumulated / counts[:, None]
            token_step[~topology.generated_token_mask] = 0
        norm = torch.linalg.vector_norm(token_step, dim=-1).max()
        if not torch.isfinite(norm) or float(norm) <= 1e-12:
            steps.append({"accepted": False, "reason": "no_route_direction", "line_search_trials": []})
            break
        token_step *= maximum_step / norm
        # Limit the change to peptide bond vectors, not only the motion of
        # individual residues. Alternating edge colours avoid double-writing
        # a residue within a sweep. A final scale enforces the bound exactly.
        pairs = topology.adjacent_token_pairs
        for _ in range(8):
            for colour in (0, 1):
                selected = pairs[topology.adjacent_pair_colors == colour]
                if not len(selected):
                    continue
                a, b = selected.unbind(1)
                difference = token_step[a] - token_step[b]
                length = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
                excess = difference * (1 - (config.maximum_adjacent_token_step_difference / length.clamp_min(1e-12)).clamp(max=1))
                movable_a = topology.generated_token_mask[a].to(token_step.dtype)[:, None]
                movable_b = topology.generated_token_mask[b].to(token_step.dtype)[:, None]
                count = (movable_a + movable_b).clamp_min(1)
                token_step[a] -= excess * movable_a / count
                token_step[b] += excess * movable_b / count
        if len(pairs):
            difference = torch.linalg.vector_norm(token_step[pairs[:, 0]] - token_step[pairs[:, 1]], dim=-1).max()
            token_step *= (config.maximum_adjacent_token_step_difference / difference.clamp_min(1e-12)).clamp(max=1)
        atom_step = token_step[topology.atom_to_token][None]
        # Route residuals use category-wise descent below; geometry remains
        # pairwise protected, including after the exact symmetry projection.
        geometry_guard = scaffold_geometry_guard(
            result, topology, replace(config, routing_ownership_weight=0.0)
        )
        accepted = False
        trials = []
        for attempt in range(config.line_search_steps):
            scale = config.line_search_contraction ** attempt
            with torch.no_grad():
                candidate = projector(result + scale * atom_step)
                geometry = geometry_guard(candidate)
                after = values(candidate)
                descent = bool(torch.isfinite(after).all() and
                               after.square().sum() < before.square().sum() - 1e-8)
                route_safe = bool(torch.all(after[before <= config.routing_tolerance] <= config.routing_tolerance)
                                  and after.max() <= before.max() + 1e-6)
                actual_step = torch.linalg.vector_norm(candidate - result, dim=-1).max()
                bounded = bool(actual_step <= maximum_step + 1e-5)
                accepted = geometry["accepted"] and descent and route_safe and bounded
                trials.append({"scale": scale, "accepted": accepted,
                               "geometry_guard": geometry, "route_descent": descent,
                               "route_nonregression": route_safe,
                               "maximum_actual_atom_step": float(actual_step),
                               "bounded_after_projection": bounded})
                if accepted:
                    result = candidate.detach()
                    break
        steps.append({"accepted": accepted, "line_search_trials": trials})
        if not accepted:
            break
    with torch.no_grad():
        final = summary(values(result))
    return result, {
        "applied": any(s["accepted"] for s in steps), "steps": steps,
        "progress": progress, "coordinate_space": "denoised_prediction",
        "maximum_token_step": maximum_step, "initial": initial, "final": final,
        "within_tolerance": final["violated_constraints"] == 0,
    }
