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
    if len(side_owner) % 2:
        raise ValueError(
            "Polymer hyperedge-cover search requires an even total number "
            f"of participant sides, observed {len(side_owner)}"
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
    "PolymerPathCoverHypothesis",
    "enumerate_directed_polymer_path_covers",
    "enumerate_polymer_hyperedge_covers",
]
