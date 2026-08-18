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
    # Order of the component stabilizer (for example 2 for a C2 building
    # block).  This is intentionally distinct from ``interface_degree``:
    # a quotient edge orbit may use one face of a C2 component while another
    # interface type uses its other face.
    valency: int
    interface_degree: int
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
    physical_edge_actions: tuple[
        tuple[str, str, tuple[str, ...]], ...
    ]
    edge_stabilizer_order: int
    executable: bool = False
    evidence_scope: str = "finite_group_component_incidence_only"

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.left.valency,
            self.right.valency,
            self.left.interface_degree,
            self.right.interface_degree,
            self.left.action.stabilizer_transform_ids,
            self.right.action.stabilizer_transform_ids,
            self.physical_edge_actions,
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

    Every group action is mapped onto a concrete left/right component pair.
    Actions that map onto the same pair form the stabilizer of one physical
    interface edge and are represented once.  This supports both free edge
    orbits (one physical edge per group action) and quotient edge orbits
    (fewer physical interfaces than ``|G|``) without confusing component
    stabilizer order with interface-specific graph degree.
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
    if requested_edges < 1 or requested_edges > group_order:
        raise ValueError(
            "physical_interface_count must be between one and the finite "
            f"group order ({group_order} for {symmetry})"
        )
    if group_order % requested_edges:
        raise ValueError(
            f"{requested_edges} physical interfaces cannot form one "
            f"uniform transitive edge orbit under {symmetry} order "
            f"{group_order}"
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
                    pair_actions: dict[tuple[str, str], list[str]] = {}
                    for transform_id in transform_ids:
                        pair = (
                            left_map[transform_id],
                            right_map[transform_id],
                        )
                        pair_actions.setdefault(pair, []).append(transform_id)
                    edges = tuple(sorted(pair_actions))
                    if len(edges) != requested_edges:
                        continue
                    left_degree = _constant_degree(edges, side=0)
                    right_degree = _constant_degree(edges, side=1)
                    if left_degree is None or right_degree is None:
                        continue
                    edge_stabilizer_orders = {
                        len(actions) for actions in pair_actions.values()
                    }
                    if len(edge_stabilizer_orders) != 1:
                        continue
                    edge_stabilizer_order = next(iter(edge_stabilizer_orders))
                    if edge_stabilizer_order * requested_edges != group_order:
                        raise RuntimeError(
                            "Interface action classes do not partition the "
                            "finite group"
                        )
                    physical_edge_actions = tuple(
                        (
                            left,
                            right,
                            tuple(pair_actions[(left, right)]),
                        )
                        for left, right in edges
                    )
                    plan = BinaryInterfaceIncidencePlan(
                        symmetry=symmetry,
                        interface_id=interface_id,
                        physical_interface_count=requested_edges,
                        left=ParticipantOrbitPlan(
                            participant=left_participant,
                            valency=left_valency,
                            interface_degree=left_degree,
                            physical_component_count=(
                                len(left_action.coset_representative_ids)
                            ),
                            action=left_action,
                        ),
                        right=ParticipantOrbitPlan(
                            participant=right_participant,
                            valency=right_valency,
                            interface_degree=right_degree,
                            physical_component_count=(
                                len(right_action.coset_representative_ids)
                            ),
                            action=right_action,
                        ),
                        physical_edges=edges,
                        physical_edge_actions=physical_edge_actions,
                        edge_stabilizer_order=edge_stabilizer_order,
                        evidence_scope=(
                            "finite_group_free_interface_incidence"
                            if edge_stabilizer_order == 1
                            else "finite_group_quotient_interface_incidence"
                        ),
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
