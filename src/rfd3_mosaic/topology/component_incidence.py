"""Finite-group incidence plans for one supplied interface type.

One chemically distinct interface can occur on building blocks with different
valencies.  A tetrahedral C2--C3 design, for example, has twelve physical
interface instances, six two-valent components and four three-valent
components.  This module derives those counts from exact stabilizer/coset
actions; it never invents a new interface geometry.

The result is architecture evidence.  Public lowering can turn it into an
executable mixed-orbit RFD3 input only after participant geometry is bound to
the selected component actions and interface incidence is proved after
expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.topology.stabilizer_cosets import (
    StabilizerCosetHypothesis,
    stabilizer_coset_hypotheses,
)
from rfd3_mosaic.topology.symmetry_connectivity import finite_symmetry_spec


@dataclass(frozen=True)
class ParticipantOrbitPlan:
    participant: str
    valency: int
    physical_component_count: int
    action: StabilizerCosetHypothesis


@dataclass(frozen=True)
class BinaryInterfaceIncidencePlan:
    """One exact bipartite component/interface incidence graph."""

    symmetry: str
    interface_id: str
    physical_interface_count: int
    left: ParticipantOrbitPlan
    right: ParticipantOrbitPlan
    physical_edges: tuple[tuple[str, str], ...]
    executable: bool = False
    evidence_scope: str = "finite_group_component_incidence_only"

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.left.valency,
            self.right.valency,
            self.left.action.stabilizer_transform_ids,
            self.right.action.stabilizer_transform_ids,
            self.physical_edges,
        )


def _action_map(
    action: StabilizerCosetHypothesis,
) -> dict[str, str]:
    return dict(action.transform_to_coset_representative)


def _constant_degree(
    edges: tuple[tuple[str, str], ...],
    *,
    side: int,
) -> int | None:
    opposite = 1 - side
    neighbours: dict[str, set[str]] = {}
    for edge in edges:
        neighbours.setdefault(edge[side], set()).add(edge[opposite])
    degrees = {len(values) for values in neighbours.values()}
    return next(iter(degrees)) if len(degrees) == 1 else None


def enumerate_binary_interface_incidence_plans(
    *,
    symmetry: str,
    interface_id: str,
    left_participant: str,
    right_participant: str,
    physical_interface_count: int | None = None,
    minimum_valency: int = 2,
    maximum_valency: int = 5,
    max_candidates: int = 4096,
) -> tuple[BinaryInterfaceIncidencePlan, ...]:
    """Enumerate mixed-stabilizer component orbits for one fixed interface.

    The first executable slice uses one free interface-edge orbit, so its
    physical multiplicity equals the full finite-group order.  Component
    counts may be smaller because each participant can have a non-trivial
    stabilizer.  Every full-group transform must map to a unique left/right
    component pair; plans with repeated physical edges or non-uniform valency
    are rejected.
    """

    if not interface_id or not left_participant or not right_participant:
        raise ValueError("Interface and participant identifiers cannot be empty")
    if left_participant == right_participant:
        raise ValueError("Binary incidence participants must be distinct")
    if minimum_valency < 1:
        raise ValueError("minimum_valency must be positive")
    if maximum_valency < minimum_valency:
        raise ValueError("maximum_valency cannot be smaller than minimum")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    registry = build_transform_registry(finite_symmetry_spec(symmetry))
    transform_ids = registry.transform_ids
    group_order = len(transform_ids)
    requested_edges = (
        group_order
        if physical_interface_count is None
        else physical_interface_count
    )
    if requested_edges != group_order:
        raise NotImplementedError(
            "The first component-incidence solver requires one free "
            f"interface orbit ({group_order} physical instances for "
            f"{symmetry}); requested {requested_edges}. Quotient interface "
            "edges require separate geometric stabilizer evidence"
        )

    actions_by_valency: dict[
        int,
        tuple[StabilizerCosetHypothesis, ...],
    ] = {}
    for valency in range(minimum_valency, maximum_valency + 1):
        if group_order % valency:
            continue
        component_count = group_order // valency
        actions = tuple(
            action
            for action in stabilizer_coset_hypotheses(
                symmetry,
                component_count,
            )
            if action.stabilizer_order == valency
        )
        if actions:
            actions_by_valency[valency] = actions

    plans: dict[tuple[object, ...], BinaryInterfaceIncidencePlan] = {}
    for left_valency, left_actions in sorted(actions_by_valency.items()):
        for right_valency, right_actions in sorted(actions_by_valency.items()):
            for left_action in left_actions:
                left_map = _action_map(left_action)
                for right_action in right_actions:
                    right_map = _action_map(right_action)
                    edges = tuple(sorted({
                        (left_map[transform_id], right_map[transform_id])
                        for transform_id in transform_ids
                    }))
                    if len(edges) != requested_edges:
                        continue
                    if _constant_degree(edges, side=0) != left_valency:
                        continue
                    if _constant_degree(edges, side=1) != right_valency:
                        continue
                    plan = BinaryInterfaceIncidencePlan(
                        symmetry=symmetry,
                        interface_id=interface_id,
                        physical_interface_count=requested_edges,
                        left=ParticipantOrbitPlan(
                            participant=left_participant,
                            valency=left_valency,
                            physical_component_count=(
                                len(left_action.coset_representative_ids)
                            ),
                            action=left_action,
                        ),
                        right=ParticipantOrbitPlan(
                            participant=right_participant,
                            valency=right_valency,
                            physical_component_count=(
                                len(right_action.coset_representative_ids)
                            ),
                            action=right_action,
                        ),
                        physical_edges=edges,
                    )
                    plans[plan.canonical_key] = plan
                    if len(plans) > max_candidates:
                        raise ValueError(
                            "Component-incidence enumeration exceeds "
                            f"max_candidates={max_candidates}; narrow the "
                            "symmetry or valency range"
                        )
    if not plans:
        raise ValueError(
            f"No exact component-incidence plan exists for {symmetry} "
            f"within valency {minimum_valency}..{maximum_valency}"
        )
    return tuple(plans[key] for key in sorted(plans))


__all__ = [
    "BinaryInterfaceIncidencePlan",
    "ParticipantOrbitPlan",
    "enumerate_binary_interface_incidence_plans",
]
