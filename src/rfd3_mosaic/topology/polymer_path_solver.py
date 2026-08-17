"""Bounded topology search for interleaved interface hyperedges.

This module answers one deliberately narrow combinatorial question.  Given
several disjoint supplied interface seeds, which scaffold links can connect
their participant sides so that every side is used exactly once?  The legacy
binary-cycle API is retained for replay stability; the hyperedge API is the
general form used for interfaces with two or more participants.

The result is topology-only evidence.  A hypothesis does not establish chain
terminus compatibility, linker reachability, geometric closure, absence of
clashes, symmetry compatibility, or RFD3 executability.  Those checks belong
to later compiler and geometry stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable


@dataclass(frozen=True, order=True)
class InterfaceHyperedgeSeed:
    """One supplied interface containing two or more physical sides."""

    seed_id: str
    side_ids: tuple[str, ...]


@dataclass(frozen=True, order=True)
class BinaryInterfaceSeed:
    """One supplied binary interface and its two physical sides."""

    seed_id: str
    left_side_id: str
    right_side_id: str


@dataclass(frozen=True, order=True)
class DirectedPolymerLink:
    """One proposed directed scaffold link between different seeds."""

    source_side_id: str
    target_side_id: str


@dataclass(frozen=True)
class PolymerPathCoverHypothesis:
    """One canonical single-cycle cover of all supplied seed sides.

    ``canonical_key`` records the cyclic traversal as
    ``(seed_id, entry_side_id, exit_side_id)`` tuples.  Global cycle rotations
    and traversal reversal are canonicalized to the same key.
    """

    canonical_key: tuple[tuple[str, str, str], ...]
    ordered_links: tuple[DirectedPolymerLink, ...]
    evidence_scope: str = "disjoint_binary_seed_topology_only"
    executable: bool = False


@dataclass(frozen=True)
class PolymerHyperedgeCoverHypothesis:
    """One canonical scaffold matching across interface hyperedges.

    Every supplied participant side occurs exactly once.  Links within the
    same supplied interface are forbidden: those sides are related by the
    non-covalent interface geometry, not by a generated peptide connection.
    """

    canonical_key: tuple[tuple[str, str], ...]
    ordered_links: tuple[DirectedPolymerLink, ...]
    evidence_scope: str = "disjoint_interface_hyperedge_topology_only"
    executable: bool = False


@dataclass(frozen=True)
class PolymerUnitPathCoverHypothesis:
    """One cover by linear protein units carrying arbitrary interface faces."""

    canonical_key: tuple[tuple[str, ...], ...]
    ordered_paths: tuple[tuple[str, ...], ...]
    ordered_links: tuple[DirectedPolymerLink, ...]
    unit_count: int
    search_complete: bool = True
    evidence_scope: str = "multi_face_polymer_unit_topology_only"
    executable: bool = False


def _validate_hyperedges(
    seeds: Iterable[InterfaceHyperedgeSeed],
) -> tuple[InterfaceHyperedgeSeed, ...]:
    ordered = tuple(sorted(seeds, key=lambda seed: seed.seed_id))
    if len(ordered) < 2:
        raise ValueError(
            "Polymer hyperedge-cover search requires at least two "
            "interface seeds"
        )
    seed_ids = [seed.seed_id for seed in ordered]
    if any(not seed_id for seed_id in seed_ids):
        raise ValueError("Interface seed IDs cannot be empty")
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("Interface seed IDs must be unique")

    side_owner: dict[str, str] = {}
    overlapping: set[str] = set()
    for seed in ordered:
        if len(seed.side_ids) < 2:
            raise ValueError(
                f"Interface seed {seed.seed_id!r} requires at least two "
                "participant sides"
            )
        if len(seed.side_ids) != len(set(seed.side_ids)):
            raise ValueError(
                f"Interface seed {seed.seed_id!r} repeats participant sides"
            )
        for side_id in seed.side_ids:
            if not side_id:
                raise ValueError("Interface side IDs cannot be empty")
            if side_id in side_owner:
                overlapping.add(side_id)
            else:
                side_owner[side_id] = seed.seed_id
    if overlapping:
        raise ValueError(
            "Polymer hyperedge-cover search requires disjoint interface "
            f"sides; overlapping side IDs: {sorted(overlapping)}"
        )
    return ordered


def enumerate_polymer_hyperedge_covers(
    seeds: Iterable[InterfaceHyperedgeSeed],
    *,
    max_candidates: int = 4096,
) -> tuple[PolymerHyperedgeCoverHypothesis, ...]:
    """Enumerate complete cross-seed scaffold matchings deterministically.

    This is the general form of the binary path-cover problem.  For two-side
    interface seeds it produces the same number of unique covers as
    :func:`enumerate_directed_polymer_path_covers`; for multi-participant
    interfaces it preserves the hyperedge while assigning every participant
    to exactly one proposed polymer unit.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    ordered = _validate_hyperedges(seeds)
    total_side_count = sum(len(seed.side_ids) for seed in ordered)
    if total_side_count % 2:
        raise ValueError(
            "Polymer hyperedge-cover search requires an even total number "
            f"of participant sides, observed {total_side_count}"
        )
    owner = {
        side_id: seed.seed_id
        for seed in ordered
        for side_id in seed.side_ids
    }
    all_sides = tuple(sorted(owner))
    matchings: list[tuple[tuple[str, str], ...]] = []

    def visit(
        remaining: tuple[str, ...],
        pairs: tuple[tuple[str, str], ...],
    ) -> None:
        if not remaining:
            if len(matchings) >= max_candidates:
                raise ValueError(
                    "Polymer hyperedge-cover search exceeds "
                    f"max_candidates={max_candidates}; narrow the supplied "
                    "seed set instead of accepting a partial enumeration"
                )
            matchings.append(tuple(sorted(pairs)))
            return
        left = remaining[0]
        for index, right in enumerate(remaining[1:], start=1):
            if owner[left] == owner[right]:
                continue
            next_remaining = remaining[1:index] + remaining[index + 1 :]
            visit(next_remaining, pairs + ((left, right),))

    visit(all_sides, ())
    if not matchings:
        raise ValueError(
            "No complete cross-seed scaffold matching exists for the "
            "declared participant counts"
        )
    return tuple(
        PolymerHyperedgeCoverHypothesis(
            canonical_key=matching,
            ordered_links=tuple(
                DirectedPolymerLink(source, target)
                for source, target in matching
            ),
        )
        for matching in sorted(set(matchings))
    )


def _canonical_linear_path(path: tuple[str, ...]) -> tuple[str, ...]:
    reversed_path = tuple(reversed(path))
    return min(path, reversed_path)


def enumerate_polymer_unit_path_covers(
    seeds: Iterable[InterfaceHyperedgeSeed],
    *,
    minimum_faces_per_unit: int,
    maximum_faces_per_unit: int,
    require_equal_unit_sizes: bool = False,
    max_candidates: int = 4096,
) -> tuple[PolymerUnitPathCoverHypothesis, ...]:
    """Enumerate linear protein units with two or more interface faces.

    Every participant side appears exactly once in one ordered protein path.
    No path may contain two sides of the same supplied interface: that would
    turn an intended intermolecular interface into an intramolecular contact.
    Adjacent path members become generated C-to-N scaffold connections.
    Protein-unit labels, path reversal and unit ordering are canonicalized.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if minimum_faces_per_unit < 2:
        raise ValueError("minimum_faces_per_unit must be at least two")
    if maximum_faces_per_unit < minimum_faces_per_unit:
        raise ValueError(
            "maximum_faces_per_unit cannot be smaller than minimum"
        )
    ordered = _validate_hyperedges(seeds)
    side_count = sum(len(seed.side_ids) for seed in ordered)
    minimum_units = max(
        max(len(seed.side_ids) for seed in ordered),
        (side_count + maximum_faces_per_unit - 1)
        // maximum_faces_per_unit,
    )
    maximum_units = min(
        side_count // minimum_faces_per_unit,
        side_count // 2,
    )
    if minimum_units > maximum_units:
        raise ValueError(
            "No polymer-unit count can satisfy the requested interface "
            "faces per unit while keeping both sides of every supplied "
            "interface on different units"
        )

    partitions: set[tuple[tuple[str, ...], ...]] = set()
    partition_search_complete = True
    unit_count_options = tuple(range(minimum_units, maximum_units + 1))
    per_unit_count_limit = max(1, max_candidates // len(unit_count_options))
    for unit_count in unit_count_options:
        unit_partitions: set[tuple[tuple[str, ...], ...]] = set()
        # Protein-unit labels are arbitrary.  Anchor the first supplied
        # interface to the first k unit labels so globally permuting unit
        # labels is not explored repeatedly.
        seed_assignments = [
            (tuple(range(len(ordered[0].side_ids))),),
            *[
                tuple(permutations(range(unit_count), len(seed.side_ids)))
                for seed in ordered[1:]
            ],
        ]
        assignment_attempt_limit = max_candidates * 20
        for assignment_index, assignment_set in enumerate(
            product(*seed_assignments),
            start=1,
        ):
            if assignment_index > assignment_attempt_limit:
                partition_search_complete = False
                break
            bins: list[list[str]] = [[] for _ in range(unit_count)]
            for seed, assignment in zip(
                ordered,
                assignment_set,
                strict=True,
            ):
                for side_id, unit_index in zip(
                    seed.side_ids,
                    assignment,
                    strict=True,
                ):
                    bins[unit_index].append(side_id)
            sizes = tuple(len(values) for values in bins)
            if any(
                size < minimum_faces_per_unit
                or size > maximum_faces_per_unit
                for size in sizes
            ):
                continue
            if require_equal_unit_sizes and len(set(sizes)) != 1:
                continue
            unit_partitions.add(
                tuple(sorted(tuple(sorted(values)) for values in bins))
            )
            if len(unit_partitions) >= per_unit_count_limit:
                partition_search_complete = False
                break
        partitions.update(unit_partitions)

    hypotheses: dict[
        tuple[tuple[str, ...], ...],
        PolymerUnitPathCoverHypothesis,
    ] = {}
    search_complete = partition_search_complete
    per_unit_hypothesis_limit = max(
        1,
        max_candidates // len(unit_count_options),
    )
    for unit_count in unit_count_options:
        unit_hypothesis_count = 0
        unit_budget_exhausted = False
        for partition in sorted(
            partition for partition in partitions
            if len(partition) == unit_count
        ):
            ordering_options = [
                tuple(
                    sorted(
                        {
                            _canonical_linear_path(tuple(ordering))
                            for ordering in permutations(unit_sides)
                        }
                    )
                )
                for unit_sides in partition
            ]
            for ordered_paths in product(*ordering_options):
                canonical_paths = tuple(sorted(ordered_paths))
                if canonical_paths in hypotheses:
                    continue
                if unit_hypothesis_count >= per_unit_hypothesis_limit:
                    search_complete = False
                    unit_budget_exhausted = True
                    break
                links = tuple(
                    DirectedPolymerLink(source, target)
                    for path in canonical_paths
                    for source, target in zip(path, path[1:])
                )
                hypotheses[canonical_paths] = PolymerUnitPathCoverHypothesis(
                    canonical_key=canonical_paths,
                    ordered_paths=canonical_paths,
                    ordered_links=links,
                    unit_count=len(canonical_paths),
                )
                unit_hypothesis_count += 1
            if unit_budget_exhausted:
                break
    if not hypotheses:
        raise ValueError(
            "No multi-face polymer unit path cover satisfies the requested "
            "interface valency"
        )
    return tuple(
        PolymerUnitPathCoverHypothesis(
            canonical_key=hypothesis.canonical_key,
            ordered_paths=hypothesis.ordered_paths,
            ordered_links=hypothesis.ordered_links,
            unit_count=hypothesis.unit_count,
            search_complete=search_complete,
        )
        for hypothesis in (
            hypotheses[key] for key in sorted(hypotheses)
        )
    )


def _validate_seeds(
    seeds: Iterable[BinaryInterfaceSeed],
) -> tuple[BinaryInterfaceSeed, ...]:
    ordered = tuple(
        sorted(
            seeds,
            key=lambda seed: (
                seed.seed_id,
                seed.left_side_id,
                seed.right_side_id,
            ),
        )
    )
    if len(ordered) < 2:
        raise ValueError(
            "Polymer path-cover search requires at least two binary "
            "interface seeds"
        )

    seed_ids = [seed.seed_id for seed in ordered]
    if any(not seed_id for seed_id in seed_ids):
        raise ValueError("Binary interface seed IDs cannot be empty")
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("Binary interface seed IDs must be unique")

    side_owner: dict[str, str] = {}
    overlapping: set[str] = set()
    for seed in ordered:
        if not seed.left_side_id or not seed.right_side_id:
            raise ValueError("Binary interface side IDs cannot be empty")
        if seed.left_side_id == seed.right_side_id:
            raise ValueError(
                f"Binary interface seed {seed.seed_id!r} must have two "
                "distinct sides"
            )
        for side_id in (seed.left_side_id, seed.right_side_id):
            if side_id in side_owner:
                overlapping.add(side_id)
            else:
                side_owner[side_id] = seed.seed_id
    if overlapping:
        raise ValueError(
            "Polymer path-cover search requires disjoint binary interface "
            f"seeds; overlapping side IDs: {sorted(overlapping)}"
        )
    return ordered


def _rotations(
    traversal: tuple[tuple[str, str, str], ...],
) -> tuple[tuple[tuple[str, str, str], ...], ...]:
    return tuple(
        traversal[offset:] + traversal[:offset]
        for offset in range(len(traversal))
    )


def _canonical_cycle(
    traversal: tuple[tuple[str, str, str], ...],
) -> tuple[tuple[str, str, str], ...]:
    reversed_traversal = tuple(
        (seed_id, exit_side_id, entry_side_id)
        for seed_id, entry_side_id, exit_side_id in reversed(traversal)
    )
    return min((*_rotations(traversal), *_rotations(reversed_traversal)))


def _links_from_cycle(
    cycle: tuple[tuple[str, str, str], ...],
) -> tuple[DirectedPolymerLink, ...]:
    return tuple(
        DirectedPolymerLink(
            source_side_id=exit_side_id,
            target_side_id=cycle[(index + 1) % len(cycle)][1],
        )
        for index, (_, _, exit_side_id) in enumerate(cycle)
    )


def enumerate_directed_polymer_path_covers(
    seeds: Iterable[BinaryInterfaceSeed],
    *,
    max_candidates: int = 4096,
) -> tuple[PolymerPathCoverHypothesis, ...]:
    """Enumerate unique single-cycle covers of disjoint binary seed sides.

    Each seed is traversed from one side to the other through its supplied
    non-covalent interface.  A directed polymer link then joins that exit side
    to the entry side of a *different* seed.  Consequently every side occurs
    exactly once as one scaffold-link endpoint and the alternating
    interface/scaffold graph is one closed cycle.

    Enumeration is deterministic and independent of input order.  Rotations
    of a cycle and reversal of the whole traversal are equivalent.  The
    search fails instead of silently returning a partial candidate set when
    ``max_candidates`` would be exceeded.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    ordered = _validate_seeds(seeds)

    # Fix the lexicographically first seed at the beginning.  This removes
    # rotation duplicates before enumeration; canonicalization below also
    # removes reversal duplicates and acts as a defensive invariant.
    anchor = ordered[0]
    remainder = ordered[1:]
    hypotheses: dict[
        tuple[tuple[str, str, str], ...],
        PolymerPathCoverHypothesis,
    ] = {}
    for tail in permutations(remainder):
        seed_order = (anchor, *tail)
        for orientation_bits in product((0, 1), repeat=len(seed_order)):
            traversal = tuple(
                (
                    seed.seed_id,
                    seed.left_side_id if bit == 0 else seed.right_side_id,
                    seed.right_side_id if bit == 0 else seed.left_side_id,
                )
                for seed, bit in zip(seed_order, orientation_bits, strict=True)
            )
            canonical_key = _canonical_cycle(traversal)
            if canonical_key in hypotheses:
                continue
            if len(hypotheses) >= max_candidates:
                raise ValueError(
                    "Polymer path-cover search exceeds "
                    f"max_candidates={max_candidates}; narrow the supplied "
                    "seed set instead of accepting an incomplete topology "
                    "enumeration"
                )
            hypotheses[canonical_key] = PolymerPathCoverHypothesis(
                canonical_key=canonical_key,
                ordered_links=_links_from_cycle(canonical_key),
            )

    return tuple(hypotheses[key] for key in sorted(hypotheses))


__all__ = [
    "BinaryInterfaceSeed",
    "DirectedPolymerLink",
    "InterfaceHyperedgeSeed",
    "PolymerHyperedgeCoverHypothesis",
    "PolymerUnitPathCoverHypothesis",
    "PolymerPathCoverHypothesis",
    "enumerate_directed_polymer_path_covers",
    "enumerate_polymer_hyperedge_covers",
    "enumerate_polymer_unit_path_covers",
]
