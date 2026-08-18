"""Conservative architecture compatibility for ordinary cage intents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rfd3_mosaic.schema import SimpleCageIntentSpec
from rfd3_mosaic.topology.stabilizer_cosets import (
    stabilizer_coset_hypotheses,
    supported_orbit_sizes,
)

_CYCLIC = re.compile(r"^C(?P<order>[1-9][0-9]*)$")
_DIHEDRAL = re.compile(r"^D(?P<order>[2-9][0-9]*)$")
_POLYHEDRAL_ORDERS = {"T": 12, "O": 24, "I": 60}


@dataclass(frozen=True)
class SimpleArchitectureHypothesis:
    """One finite-group compatibility result, not a frozen design."""

    symmetry: str
    group_action_count: int
    accepted: bool
    interface_physical_instances: dict[str, int]
    interface_orbit_actions: dict[str, dict[str, object]]
    rejection_reasons: tuple[str, ...]
    unresolved: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def symmetry_group_action_count(symmetry_id: str) -> int:
    cyclic = _CYCLIC.fullmatch(symmetry_id)
    if cyclic is not None:
        return int(cyclic.group("order"))
    dihedral = _DIHEDRAL.fullmatch(symmetry_id)
    if dihedral is not None:
        return 2 * int(dihedral.group("order"))
    try:
        return _POLYHEDRAL_ORDERS[symmetry_id]
    except KeyError as error:
        raise ValueError(f"Unsupported symmetry {symmetry_id!r}") from error


def candidate_symmetries(intent: SimpleCageIntentSpec) -> tuple[str, ...]:
    if intent.goal.symmetry != "auto":
        return tuple(intent.goal.symmetry)
    cyclic = tuple(f"C{order}" for order in range(2, 13))
    dihedral = tuple(f"D{order}" for order in range(2, 7))
    polyhedral = ("T", "O", "I")
    if intent.goal.architecture == "ring":
        return cyclic
    if intent.goal.architecture == "cage":
        return dihedral + polyhedral
    return cyclic + dihedral + polyhedral


def analyze_simple_architectures(
    intent: SimpleCageIntentSpec,
) -> tuple[SimpleArchitectureHypothesis, ...]:
    """Filter finite-group orbit actions by user-visible intent.

    Interface multiplicity is resolved using actual subgroup/coset
    decompositions.  A numerical divisor of the group order is not accepted
    unless the registered group contains the required stabilizer subgroup.
    Geometry is still fail-closed: a non-trivial stabilizer remains unresolved
    until the supplied seed is proven invariant under that subgroup.
    """

    results: list[SimpleArchitectureHypothesis] = []
    for symmetry_id in candidate_symmetries(intent):
        count = symmetry_group_action_count(symmetry_id)
        reasons: list[str] = []
        if (
            intent.goal.subunits is not None
            and intent.goal.composition == "homomer"
            and not (
                intent.goal.subunits.minimum
                <= count
                <= intent.goal.subunits.maximum
            )
        ):
            reasons.append(
                "generic full-orbit copy count "
                f"{count} is outside requested subunits "
                f"{intent.goal.subunits.minimum}.."
                f"{intent.goal.subunits.maximum}"
            )
        instances: dict[str, int] = {}
        orbit_actions: dict[str, dict[str, object]] = {}
        available_orbit_sizes = supported_orbit_sizes(symmetry_id)
        for interface_id, interface in intent.interface_seeds.items():
            accepted_sizes = tuple(
                orbit_size
                for orbit_size in available_orbit_sizes
                if interface.use.accepts(orbit_size)
            )
            if not accepted_sizes:
                reasons.append(
                    f"interface {interface_id} requests "
                    f"{interface.use.description}, but {symmetry_id} has "
                    "no exact subgroup/coset orbit with that physical "
                    f"multiplicity; supported orbit sizes are "
                    f"{list(available_orbit_sizes)}"
                )
                continue
            # Prefer the least-special (largest) compatible orbit.  Exact
            # requests naturally select their one requested multiplicity.
            orbit_size = max(accepted_sizes)
            hypotheses = stabilizer_coset_hypotheses(
                symmetry_id,
                orbit_size,
            )
            if not hypotheses:
                raise RuntimeError(
                    "supported_orbit_sizes returned an unrealizable orbit"
                )
            canonical = hypotheses[0]
            instances[interface_id] = orbit_size
            orbit_actions[interface_id] = {
                "orbit_size": orbit_size,
                "stabilizer_order": canonical.stabilizer_order,
                "stabilizer_transform_ids": list(
                    canonical.stabilizer_transform_ids
                ),
                "coset_representative_ids": list(
                    canonical.coset_representative_ids
                ),
                "subgroup_hypothesis_count": len(hypotheses),
                "requires_geometric_stabilizer_validation": (
                    canonical.stabilizer_order > 1
                ),
            }
        unresolved = [
            "interface-side ownership by polymer units",
            "directed scaffold connection order",
            "symmetry-neighbour relation for each interface",
            "continuous radius/orientation/axial pose",
        ]
        if intent.goal.composition == "heteromer":
            unresolved.append("heteromeric asymmetric-unit composition")
        elif (
            intent.goal.composition == "auto"
            and intent.goal.subunits is not None
        ):
            unresolved.append(
                "physical subunit count depends on unresolved polymer-unit "
                "ownership/composition"
            )
        if intent.goal.diameter_angstrom is not None:
            unresolved.append(
                "continuous pose satisfying requested assembly diameter"
            )
        if intent.goal.cavity_diameter_angstrom is not None:
            unresolved.append(
                "continuous pose satisfying requested cavity diameter"
            )
        usage_descriptions = {
            item.use.description
            for item in intent.interface_seeds.values()
        }
        if len(usage_descriptions) > 1:
            unresolved.append(
                "joint compatibility of mixed interface orbit actions"
            )
        if any(
            action["requires_geometric_stabilizer_validation"]
            for action in orbit_actions.values()
        ):
            unresolved.append(
                "geometric invariance of supplied seeds under selected "
                "stabilizers"
            )
        results.append(
            SimpleArchitectureHypothesis(
                symmetry=symmetry_id,
                group_action_count=count,
                accepted=not reasons,
                interface_physical_instances=instances,
                interface_orbit_actions=orbit_actions,
                rejection_reasons=tuple(reasons),
                unresolved=tuple(unresolved),
            )
        )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                not item.accepted,
                item.group_action_count,
                item.symmetry,
            ),
        )
    )


__all__ = [
    "SimpleArchitectureHypothesis",
    "analyze_simple_architectures",
    "candidate_symmetries",
    "symmetry_group_action_count",
]
