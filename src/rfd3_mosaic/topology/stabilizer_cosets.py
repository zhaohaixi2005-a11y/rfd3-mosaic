"""Exact stabilizer/coset decompositions for finite symmetry registries.

An orbit with ``m`` physical instances in a finite group ``G`` is only
possible when there is a subgroup ``H`` with ``m = |G| / |H|``.  This module
computes that statement from the actual registered transform multiplication
table; divisibility alone is not treated as sufficient evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.topology.symmetry_connectivity import finite_symmetry_spec


@dataclass(frozen=True)
class StabilizerCosetHypothesis:
    """One exact transitive group action represented by left cosets."""

    symmetry: str
    group_order: int
    orbit_size: int
    stabilizer_order: int
    stabilizer_transform_ids: tuple[str, ...]
    coset_representative_ids: tuple[str, ...]
    cosets: tuple[tuple[str, ...], ...]
    transform_to_coset_representative: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=None)
def _multiplication_table(
    symmetry_id: str,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    registry = build_transform_registry(finite_symmetry_spec(symmetry_id))
    transform_ids = registry.transform_ids
    indices = {
        transform_id: index
        for index, transform_id in enumerate(transform_ids)
    }
    table = tuple(
        tuple(
            indices[registry.compose_ids(left_id, right_id)]
            for right_id in transform_ids
        )
        for left_id in transform_ids
    )
    return transform_ids, table


def _generated_subgroup(
    generators: frozenset[int],
    table: tuple[tuple[int, ...], ...],
) -> frozenset[int]:
    reached = {0, *generators}
    changed = True
    while changed:
        changed = False
        current = tuple(sorted(reached))
        for left in current:
            for right in current:
                product = table[left][right]
                if product not in reached:
                    reached.add(product)
                    changed = True
    return frozenset(reached)


@lru_cache(maxsize=None)
def subgroup_indices(symmetry_id: str) -> tuple[tuple[int, ...], ...]:
    """Enumerate every subgroup deterministically from the registry table."""

    transform_ids, table = _multiplication_table(symmetry_id)
    discovered: set[frozenset[int]] = {frozenset({0})}
    frontier = [frozenset({0})]
    while frontier:
        subgroup = frontier.pop(0)
        for candidate in range(1, len(transform_ids)):
            if candidate in subgroup:
                continue
            generated = _generated_subgroup(
                subgroup | frozenset({candidate}),
                table,
            )
            if generated in discovered:
                continue
            discovered.add(generated)
            frontier.append(generated)
    return tuple(
        tuple(sorted(subgroup))
        for subgroup in sorted(
            discovered,
            key=lambda item: (len(item), tuple(sorted(item))),
        )
    )


def stabilizer_coset_hypotheses(
    symmetry_id: str,
    orbit_size: int,
) -> tuple[StabilizerCosetHypothesis, ...]:
    """Return all exact subgroup/coset realizations of ``orbit_size``.

    The returned cosets are left cosets ``gH`` in canonical registry order.
    Every result is checked to be disjoint and to partition the full group.
    """

    if orbit_size < 1:
        raise ValueError("orbit_size must be positive")
    transform_ids, table = _multiplication_table(symmetry_id)
    group_order = len(transform_ids)
    if group_order % orbit_size:
        return ()
    stabilizer_order = group_order // orbit_size
    hypotheses: list[StabilizerCosetHypothesis] = []
    for subgroup_tuple in subgroup_indices(symmetry_id):
        if len(subgroup_tuple) != stabilizer_order:
            continue
        subgroup = frozenset(subgroup_tuple)
        covered: set[int] = set()
        representatives: list[int] = []
        cosets: list[tuple[int, ...]] = []
        for representative in range(group_order):
            if representative in covered:
                continue
            coset = tuple(
                sorted(
                    {table[representative][item] for item in subgroup}
                )
            )
            if len(coset) != stabilizer_order or covered.intersection(coset):
                raise RuntimeError(
                    f"Invalid coset partition while resolving {symmetry_id}"
                )
            representatives.append(representative)
            cosets.append(coset)
            covered.update(coset)
        if len(cosets) != orbit_size or len(covered) != group_order:
            raise RuntimeError(
                f"Cosets do not partition complete {symmetry_id} registry"
            )
        representative_by_transform = {
            transform_index: representatives[coset_index]
            for coset_index, coset in enumerate(cosets)
            for transform_index in coset
        }
        hypotheses.append(
            StabilizerCosetHypothesis(
                symmetry=symmetry_id,
                group_order=group_order,
                orbit_size=orbit_size,
                stabilizer_order=stabilizer_order,
                stabilizer_transform_ids=tuple(
                    transform_ids[index] for index in subgroup_tuple
                ),
                coset_representative_ids=tuple(
                    transform_ids[index] for index in representatives
                ),
                cosets=tuple(
                    tuple(transform_ids[index] for index in coset)
                    for coset in cosets
                ),
                transform_to_coset_representative=tuple(
                    (
                        transform_ids[index],
                        transform_ids[representative_by_transform[index]],
                    )
                    for index in range(group_order)
                ),
            )
        )
    return tuple(hypotheses)


def supported_orbit_sizes(symmetry_id: str) -> tuple[int, ...]:
    """Return every transitive orbit size supported by an actual subgroup."""

    transform_ids, _ = _multiplication_table(symmetry_id)
    group_order = len(transform_ids)
    return tuple(
        sorted(
            {
                group_order // len(subgroup)
                for subgroup in subgroup_indices(symmetry_id)
            }
        )
    )


__all__ = [
    "StabilizerCosetHypothesis",
    "stabilizer_coset_hypotheses",
    "subgroup_indices",
    "supported_orbit_sizes",
]
