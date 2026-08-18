"""Finite-group connectivity primitives for ordinary assembly resolution."""

from __future__ import annotations

import re
from itertools import combinations

from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.schema import SymmetryTransformSetSpec

_CYCLIC = re.compile(r"^C(?P<order>[1-9][0-9]*)$")
_DIHEDRAL = re.compile(r"^D(?P<order>[2-9][0-9]*)$")
_POLYHEDRAL = {
    "T": ("tetrahedral", 12),
    "O": ("octahedral", 24),
    "I": ("icosahedral", 60),
}


def finite_symmetry_spec(symmetry_id: str) -> SymmetryTransformSetSpec:
    """Build the canonical typed transform-set declaration for a group."""

    cyclic = _CYCLIC.fullmatch(symmetry_id)
    if cyclic is not None:
        return SymmetryTransformSetSpec(
            type="cyclic",
            order=int(cyclic.group("order")),
        )
    dihedral = _DIHEDRAL.fullmatch(symmetry_id)
    if dihedral is not None:
        return SymmetryTransformSetSpec(
            type="dihedral",
            order=int(dihedral.group("order")),
            secondary_axis=(1.0, 0.0, 0.0),
        )
    try:
        group_type, order = _POLYHEDRAL[symmetry_id]
    except KeyError as error:
        raise ValueError(
            f"Unsupported finite symmetry {symmetry_id!r}"
        ) from error
    return SymmetryTransformSetSpec(type=group_type, order=order)


def generated_transform_ids(
    symmetry_id: str,
    relation_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the subgroup reached by repeatedly applying relations."""

    registry = build_transform_registry(finite_symmetry_spec(symmetry_id))
    unknown = set(relation_ids) - set(registry.transform_ids)
    if unknown:
        raise ValueError(
            f"Unknown relations for {symmetry_id}: {sorted(unknown)}"
        )
    reached = {registry.identity_id}
    frontier = [registry.identity_id]
    while frontier:
        current = frontier.pop()
        for relation_id in relation_ids:
            for candidate in (
                registry.compose_ids(relation_id, current),
                registry.compose_ids(current, relation_id),
            ):
                if candidate in reached:
                    continue
                reached.add(candidate)
                frontier.append(candidate)
    return tuple(
        transform_id
        for transform_id in registry.transform_ids
        if transform_id in reached
    )


def minimal_group_relations(
    symmetry_id: str,
    *,
    maximum_generators: int = 3,
) -> tuple[str, ...]:
    """Find the canonical smallest relation set generating the full group."""

    if maximum_generators < 1:
        raise ValueError("maximum_generators must be positive")
    registry = build_transform_registry(finite_symmetry_spec(symmetry_id))
    candidates = tuple(
        transform_id
        for transform_id in registry.transform_ids
        if transform_id != registry.identity_id
    )
    for size in range(1, min(maximum_generators, len(candidates)) + 1):
        for relations in combinations(candidates, size):
            if len(generated_transform_ids(symmetry_id, relations)) == len(
                registry.transform_ids
            ):
                return relations
    raise ValueError(
        f"Could not find at most {maximum_generators} relations generating "
        f"the complete {symmetry_id} transform registry"
    )


__all__ = [
    "finite_symmetry_spec",
    "generated_transform_ids",
    "minimal_group_relations",
]
