"""Bounded final feasibility restoration, with explicit residual failures.

This operates on generated residue translations, followed by the caller's
exact motif/symmetry projector. It is not an all-atom relaxation or a proof
that an arbitrary fixed-anchor problem is feasible.
"""

from typing import Any, Callable

import torch

from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig,
    ScaffoldCoreTopology,
    scaffold_geometry_deficits,
)


def restore_generated_geometry(
    coordinates: torch.Tensor,
    topology: ScaffoldCoreTopology,
    config: ScaffoldCoreGuidanceConfig,
    *,
    projector: Callable[[torch.Tensor], torch.Tensor],
    iterations: int = 64,
    segment_distance: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reduce violations without introducing new ones or trading categories.

    Each accepted trial must reduce the squared residual, preserve every
    currently satisfied constraint, and not increase either the maximum or
    squared sum in ANY constraint category. Already violated pairs may trade
    within these bounds; otherwise intersecting/broken chains can deadlock
    under a strictly pairwise monotonic correction. All residuals are logged.
    """
    if coordinates.ndim != 3 or coordinates.shape[0] != 1:
        raise ValueError("Geometry restoration requires [1, atoms, 3]")
    if iterations < 0 or segment_distance <= 0:
        raise ValueError("Invalid geometry restoration limits")

    def evaluate(value):
        return scaffold_geometry_deficits(
            value, topology, config, segment_distance=segment_distance
        )

    def summary(values):
        return {
            name: {
                "violated_constraints": int((v > 1e-6).sum()),
                "maximum_violation": float(v.max()) if v.numel() else 0.0,
                "squared_violation": float(v.square().sum()),
            }
            for name, v in values.items()
        }

    result = coordinates.detach().clone()
    with torch.no_grad():
        initial = summary(evaluate(result))
    steps = []
    for _ in range(iterations):
        with torch.enable_grad():
            source = result.detach().requires_grad_(True)
            before = evaluate(source)
            if all(not torch.any(v > 1e-6) for v in before.values()):
                break
            energy = sum(v.square().sum() for v in before.values())
            gradient = torch.autograd.grad(energy, source)[0][0]
        before = {k: v.detach() for k, v in before.items()}
        token_gradient = result.new_zeros((len(topology.generated_token_mask), 3))
        token_gradient.index_add_(0, topology.atom_to_token, gradient)
        token_gradient[~topology.generated_token_mask] = 0
        maximum = torch.linalg.vector_norm(token_gradient, dim=-1).max()
        if not torch.isfinite(maximum) or float(maximum) <= 1e-12:
            break
        delta = -token_gradient * (config.maximum_token_step / maximum)
        atom_delta = delta[topology.atom_to_token][None]
        accepted = False
        trials = []
        for attempt in range(config.line_search_steps):
            scale = config.line_search_contraction ** attempt
            with torch.no_grad():
                candidate = projector(result + scale * atom_delta)
                if not torch.isfinite(candidate).all():
                    trials.append({"scale": scale, "accepted": False, "reason": "nonfinite"})
                    continue
                after = evaluate(candidate)
                checks = {}
                for name, old in before.items():
                    new = after[name]
                    checks[name] = bool(
                        torch.all(new[old <= 1e-6] <= 1e-6)
                        and new.square().sum() <= old.square().sum() + 1e-10
                        and (not old.numel() or new.max() <= old.max() + 1e-6)
                    )
                descent = float(sum(v.square().sum() for v in after.values())) < float(energy) - 1e-10
                accepted = descent and all(checks.values())
                trials.append({"scale": scale, "accepted": accepted, "descent": descent, "categories": checks})
                if accepted:
                    result = candidate.detach()
                    break
        steps.append({"accepted": accepted, "line_search_trials": trials})
        if not accepted:
            break
    with torch.no_grad():
        final = summary(evaluate(result))
    return result, {
        "phase": "final_geometry_restoration",
        "applied": any(s["accepted"] for s in steps),
        "initial": initial, "final": final, "steps": steps,
        "within_tolerance": all(v["violated_constraints"] == 0 for v in final.values()),
        "maximum_token_step": config.maximum_token_step,
        "segment_distance": segment_distance,
        "ca_distance": config.clash_distance,
        "backbone_distance": config.backbone_distance,
        "backbone_tolerance": config.backbone_tolerance,
    }
