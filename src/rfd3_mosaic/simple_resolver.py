"""Resolve a conservative ordinary cage intent into public designs.

The ordinary and expert authoring surfaces deliberately meet at
``UserDesignSpec``.  This module is therefore an architecture frontend, not a
second compiler or sampler.  Every selected result is compiled and strictly
replayed by :func:`rfd3_mosaic.graph_search.rank_design_candidates` before it
is advertised as executable.

The executable Cn contract accepts several disjoint supplied interface seeds,
including interfaces with more than two participants.  Every supplied seed is
one exact rigid hyperedge; an internal contact spanning tree exposes it to the
pairwise RFD3 runtime without changing its multiplicity or fragment geometry.
Topology hypotheses become executable only after real termini, symmetry
winding, linker feasibility, clashes and strict replay are bound through the
normal compiler.  Homomer-equivalence inference and stabilizer/coset orbits
remain explicit fail-closed boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any, Iterable

import yaml

from rfd3_mosaic.design_compiler import (
    lower_user_design,
    parse_public_selector,
)
from rfd3_mosaic.graph_search import rank_design_candidates
from rfd3_mosaic.pose_optimizer import (
    initialize_global_seed_layout,
    optimize_candidate_subset,
)
from rfd3_mosaic.seed_library import materialize_seed_library
from rfd3_mosaic.seed_stabilizer import resolve_seed_stabilizer
from rfd3_mosaic.schema import SimpleCageIntentSpec, UserDesignSpec
from rfd3_mosaic.simple_architecture import (
    analyze_simple_architectures,
    candidate_symmetries,
    symmetry_group_action_count,
)
from rfd3_mosaic.structure import read_structure_atoms
from rfd3_mosaic.structure_inspection import (
    inspect_declared_interface_relation,
)
from rfd3_mosaic.topology.polymer_path_solver import (
    BinaryInterfaceSeed,
    InterfaceHyperedgeSeed,
    enumerate_directed_polymer_path_covers,
    enumerate_polymer_hyperedge_covers,
)
from rfd3_mosaic.topology.interface_seed_graph import (
    analyze_interleaved_interface_seed_topology,
)
from rfd3_mosaic.topology.symmetry_connectivity import (
    minimal_group_relations,
)


_CYCLIC = re.compile(r"^C(?P<order>[2-9][0-9]*)$")


@dataclass(frozen=True)
class SimpleDesignCandidate:
    """One deterministic ordinary-resolution hypothesis."""

    candidate_id: str
    symmetry: str
    topology_id: str
    design: UserDesignSpec
    connection_order: tuple[str, str]
    connection_orbit_offset: int
    connection_group_relation: str | None = None
    resolution_frontend: str = "simple_binary_cn_ring_v1"
    polymer_links: tuple[tuple[str, str, int | str], ...] = ()
    polymer_units_per_copy: int = 1
    physical_polymer_unit_count: int | None = None
    expanded_topology_status: str | None = None
    unresolved_variables: tuple[str, ...] = ()
    preflight_failures: tuple[str, ...] = ()
    pose_sample_index: int = 0
    global_pose_initialization: bool = False
    interface_hyperedges: tuple[tuple[str, tuple[str, ...]], ...] = ()
    stabilizer_evidence: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        if self.polymer_links:
            neighbour_transforms = {
                f"polymer_link_{index + 1:03d}": (
                    f"orbit_offset:{relation:+d}"
                    if isinstance(relation, int)
                    else f"transform:{relation}"
                )
                for index, (_, _, relation) in enumerate(self.polymer_links)
            }
        else:
            neighbour_transforms = {
                "polymer_unit": (
                    f"orbit_offset:{self.connection_orbit_offset:+d}"
                )
            }
        return {
            "resolution_frontend": self.resolution_frontend,
            "topology_id": self.topology_id,
            "connection_order": list(self.connection_order),
            "connection_orbit_offset": self.connection_orbit_offset,
            "connection_group_relation": self.connection_group_relation,
            "polymer_links": [list(link) for link in self.polymer_links],
            "polymer_units_per_copy": self.polymer_units_per_copy,
            "physical_polymer_unit_count": (
                self.physical_polymer_unit_count
            ),
            "expanded_topology_status": self.expanded_topology_status,
            "unresolved_variables": list(self.unresolved_variables),
            "preflight_failures": list(self.preflight_failures),
            "pose_sample_index": self.pose_sample_index,
            "global_pose_initialization": self.global_pose_initialization,
            "interface_hyperedges": {
                interface_id: list(member_ids)
                for interface_id, member_ids in self.interface_hyperedges
            },
            "stabilizer_evidence": self.stabilizer_evidence,
            "neighbour_transforms": neighbour_transforms,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_seed_contract(interface_id: str, seed) -> None:
    if len(seed.participants) < 2:
        raise ValueError(
            f"Interface {interface_id!r} requires at least two participants"
        )
    if seed.geometry != "preserve_exact":
        raise NotImplementedError(
            "The first ordinary resolver supports geometry=preserve_exact; "
            "bounded supplied-interface geometry is not frozen yet"
        )
    for participant in seed.participants:
        selector = seed.selectors[participant]
        if len(parse_public_selector(selector)) != 1:
            raise NotImplementedError(
                f"Interface {interface_id!r} selector {selector!r} contains "
                "several disjoint ranges. Ordered multi-fragment paths must "
                "be resolved explicitly before execution"
            )


def _supported_seed(intent: SimpleCageIntentSpec):
    if len(intent.interface_seeds) != 1:
        raise ValueError("Expected exactly one supplied interface seed")
    interface_id, seed = next(iter(intent.interface_seeds.items()))
    _validate_seed_contract(interface_id, seed)
    return interface_id, seed


def _supported_interface_seeds(
    intent: SimpleCageIntentSpec,
) -> tuple[tuple[str, Any], ...]:
    seeds = tuple(sorted(intent.interface_seeds.items()))
    for interface_id, seed in seeds:
        _validate_seed_contract(interface_id, seed)

    selected_ranges: list[tuple[str, str, int, int]] = []
    for interface_id, seed in seeds:
        for participant in seed.participants:
            segment = parse_public_selector(seed.selectors[participant])[0]
            for (
                previous_interface,
                previous_chain,
                previous_start,
                previous_end,
            ) in selected_ranges:
                if segment.chain_id != previous_chain:
                    continue
                if not (
                    segment.residue_end < previous_start
                    or segment.residue_start > previous_end
                ):
                    raise ValueError(
                        "Multi-seed executable resolution requires disjoint "
                        "selected fragments; "
                        f"{interface_id!r}:{participant} overlaps "
                        f"{previous_interface!r} on chain "
                        f"{segment.chain_id}"
                    )
            selected_ranges.append(
                (
                    interface_id,
                    segment.chain_id,
                    segment.residue_start,
                    segment.residue_end,
                )
            )
    return seeds


# Internal compatibility alias for older callers/tests.  The implementation
# is now hyperedge-capable; retaining the name avoids a migration-only API
# break in this research branch.
_supported_binary_seeds = _supported_interface_seeds


def _validate_resolution_contract(
    intent: SimpleCageIntentSpec,
    *,
    allow_continuous_geometry: bool = False,
) -> None:
    if intent.goal.composition == "heteromer":
        raise NotImplementedError(
            "Automatic heteromer ownership is not implemented; use expert "
            "components or keep composition=auto"
        )
    if (
        intent.goal.diameter_angstrom is not None
        and not allow_continuous_geometry
    ):
        raise NotImplementedError(
            "Requested assembly diameter is not yet part of the executable "
            "continuous-pose objective"
        )
    if intent.goal.cavity_diameter_angstrom is not None:
        raise NotImplementedError(
            "Requested cavity diameter is not yet backed by an executable "
            "cavity evaluator"
        )


def _validated_input_evidence(
    intent: SimpleCageIntentSpec,
    interface_id: str,
    seed,
) -> dict[str, Any]:
    evidence = inspect_declared_interface_relation(
        intent.input,
        interface_id=interface_id,
        participants=seed.participants,
        selectors=seed.selectors,
        contact_cutoff=intent.inspection.contact_cutoff,
        minimum_atom_contacts=intent.inspection.minimum_atom_contacts,
        minimum_contact_residues_per_side=(
            intent.inspection.minimum_contact_residues_per_side
        ),
    )
    if not evidence.contact_graph_connected:
        raise ValueError(
            f"Supplied interface {interface_id!r} does not form a connected "
            "contact graph under the frozen inspection thresholds"
        )
    return {
        "interface_id": interface_id,
        "participants": list(seed.participants),
        "active_contact_pairs": [
            list(pair) for pair in evidence.active_contact_pairs
        ],
        "contact_cutoff": intent.inspection.contact_cutoff,
        "minimum_atom_contacts": intent.inspection.minimum_atom_contacts,
        "minimum_contact_residues_per_side": (
            intent.inspection.minimum_contact_residues_per_side
        ),
    }


def _validate_multi_seed_backbone_anchors(
    intent: SimpleCageIntentSpec,
) -> None:
    """Require peptide anchor atoms at both ends of every selected fragment."""

    atoms = read_structure_atoms(
        intent.input,
        mmcif_identifier_namespace="label",
    )
    atom_names = {
        (atom.chain_id, atom.residue_number, atom.atom_name.upper())
        for atom in atoms
    }
    failures: list[str] = []
    for interface_id, seed in _supported_binary_seeds(intent):
        for participant in seed.participants:
            selector = seed.selectors[participant]
            segment = parse_public_selector(selector)[0]
            for boundary_name, residue in (
                ("N-boundary", segment.residue_start),
                ("C-boundary", segment.residue_end),
            ):
                missing = [
                    atom_name
                    for atom_name in ("N", "CA", "C")
                    if (
                        segment.chain_id,
                        residue,
                        atom_name,
                    )
                    not in atom_names
                ]
                if missing:
                    failures.append(
                        f"{interface_id}:{participant} {boundary_name} "
                        f"residue {segment.chain_id}{residue} missing "
                        + "/".join(missing)
                    )
    if failures:
        raise ValueError(
            "Multi-seed polymer termini cannot be bound to complete peptide "
            "backbone anchors: " + "; ".join(failures)
        )


def _resolver_symmetry_order(
    intent: SimpleCageIntentSpec,
    symmetry_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    requested = tuple(symmetry_ids) if symmetry_ids is not None else None
    if requested is not None:
        if not requested:
            raise ValueError("At least one resolver symmetry is required")
        if len(requested) != len(set(requested)):
            raise ValueError("Resolver symmetry IDs must be unique")
    explicitly_requested = (
        requested is not None or intent.goal.symmetry != "auto"
    )
    analyses = analyze_simple_architectures(intent)
    compatible = {item.symmetry: item for item in analyses if item.accepted}
    symmetry_order = requested or tuple(compatible)
    rejected_requested = [
        symmetry_id
        for symmetry_id in symmetry_order
        if symmetry_id not in compatible
    ]
    if rejected_requested:
        reasons = {
            item.symmetry: list(item.rejection_reasons)
            for item in analyses
            if item.symmetry in rejected_requested
        }
        raise ValueError(
            "Requested resolver symmetries are incompatible with the "
            f"ordinary intent: {reasons or rejected_requested}"
        )
    partial_orbits = {
        symmetry_id: {
            interface_id: observed
            for interface_id, observed in (
                compatible[symmetry_id].interface_physical_instances.items()
            )
            if observed != compatible[symmetry_id].group_action_count
        }
        for symmetry_id in symmetry_order
        if symmetry_id in compatible
    }
    partial_orbits = {
        symmetry_id: interfaces
        for symmetry_id, interfaces in partial_orbits.items()
        if interfaces
    }
    validated_partial_orbits: set[str] = set()
    if partial_orbits and len(intent.interface_seeds) == 1:
        interface_id, seed = next(iter(intent.interface_seeds.items()))
        for symmetry_id, interface_counts in partial_orbits.items():
            orbit_size = interface_counts[interface_id]
            try:
                resolve_seed_stabilizer(
                    source=intent.input,
                    interface_id=interface_id,
                    participants=seed.participants,
                    selectors=seed.selectors,
                    symmetry=symmetry_id,
                    orbit_size=orbit_size,
                )
            except ValueError:
                continue
            validated_partial_orbits.add(symmetry_id)
    executable_symmetry_order = tuple(
        symmetry_id
        for symmetry_id in symmetry_order
        if symmetry_id not in partial_orbits
        or symmetry_id in validated_partial_orbits
    )
    if partial_orbits and not executable_symmetry_order:
        raise NotImplementedError(
            "The requested multiplicities have exact stabilizer/coset "
            "solutions at the architecture-planning layer, but executable "
            "lowering requires geometric validation that each supplied "
            "seed is invariant under its selected stabilizer. Mosaic will "
            f"not emit a fake full orbit: {partial_orbits}"
        )
    unsupported = [
        symmetry_id
        for symmetry_id in executable_symmetry_order
        if _CYCLIC.fullmatch(symmetry_id) is None
    ]
    if unsupported and explicitly_requested:
        raise NotImplementedError(
            "The executable ordinary resolver currently supports only Cn; "
            f"explicit unsupported choices: {unsupported}"
        )
    return tuple(
        symmetry_id
        for symmetry_id in executable_symmetry_order
        if _CYCLIC.fullmatch(symmetry_id) is not None
    )


def _single_seed_finite_action(
    intent: SimpleCageIntentSpec,
    *,
    symmetry_id: str,
    interface_id: str,
    seed,
):
    analysis = next(
        item
        for item in analyze_simple_architectures(intent)
        if item.symmetry == symmetry_id
    )
    orbit_size = analysis.interface_physical_instances[interface_id]
    if orbit_size == analysis.group_action_count:
        return None
    return resolve_seed_stabilizer(
        source=intent.input,
        interface_id=interface_id,
        participants=seed.participants,
        selectors=seed.selectors,
        symmetry=symmetry_id,
        orbit_size=orbit_size,
    )


def _multi_seed_side_records(
    seeds: tuple[tuple[str, Any], ...],
) -> tuple[
    tuple[InterfaceHyperedgeSeed, ...],
    dict[str, tuple[str, str, str]],
]:
    topology_seeds: list[InterfaceHyperedgeSeed] = []
    sides: dict[str, tuple[str, str, str]] = {}
    for interface_id, seed in seeds:
        side_ids = tuple(
            f"{interface_id}:side_{index}"
            for index in range(len(seed.participants))
        )
        topology_seeds.append(
            InterfaceHyperedgeSeed(
                seed_id=interface_id,
                side_ids=side_ids,
            )
        )
        for side_id, participant in zip(
            side_ids,
            seed.participants,
            strict=True,
        ):
            sides[side_id] = (
                interface_id,
                participant,
                seed.selectors[participant],
            )
    return tuple(topology_seeds), sides


def _contact_spanning_tree(
    participants: tuple[str, ...],
    active_contact_pairs: Iterable[Iterable[str]],
) -> tuple[tuple[str, str], ...]:
    """Return a deterministic contact-supported tree for one hyperedge."""

    parent = {participant: participant for participant in participants}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    selected: list[tuple[str, str]] = []
    normalized = sorted(
        {
            tuple(sorted((str(pair[0]), str(pair[1]))))
            for pair in active_contact_pairs
        }
    )
    for left, right in normalized:
        if left not in parent or right not in parent:
            raise ValueError(
                "Interface inspection returned a contact pair outside its "
                f"declared participants: {(left, right)}"
            )
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        keep, merge = sorted((left_root, right_root))
        parent[merge] = keep
        selected.append((left, right))
    if len(selected) != len(participants) - 1:
        raise ValueError(
            "Supplied multi-participant interface does not have a connected "
            "contact-supported spanning tree"
        )
    return tuple(selected)


def _multi_seed_symmetry_order(
    intent: SimpleCageIntentSpec,
    symmetry_ids: Iterable[str] | None,
    *,
    polymer_units_per_copy: int,
    topological_cycle_rank: int,
) -> tuple[str, ...]:
    explicitly_requested = (
        symmetry_ids is not None or intent.goal.symmetry != "auto"
    )
    requested = (
        tuple(symmetry_ids)
        if symmetry_ids is not None
        else candidate_symmetries(intent)
    )
    if not requested:
        raise ValueError("At least one resolver symmetry is required")
    if len(requested) != len(set(requested)):
        raise ValueError("Resolver symmetry IDs must be unique")

    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    for symmetry_id in requested:
        reasons: list[str] = []
        try:
            order = symmetry_group_action_count(symmetry_id)
            generators = minimal_group_relations(symmetry_id)
        except ValueError as error:
            reasons.append(
                f"finite-group connectivity is unavailable: {error}"
            )
        else:
            if len(generators) > topological_cycle_rank:
                reasons.append(
                    f"{symmetry_id} needs {len(generators)} independent "
                    "group generators, but the supplied interface/polymer "
                    f"base graph has cycle rank {topological_cycle_rank}"
                )
            for interface_id, seed in intent.interface_seeds.items():
                if not seed.use.accepts(order):
                    reasons.append(
                        f"interface {interface_id} requests "
                        f"{seed.use.description}, but {symmetry_id} produces "
                        f"{order} "
                        "physical instances"
                    )
            physical_polymer_units = polymer_units_per_copy * order
            if (
                intent.goal.subunits is not None
                and not (
                    intent.goal.subunits.minimum
                    <= physical_polymer_units
                    <= intent.goal.subunits.maximum
                )
            ):
                reasons.append(
                    f"{polymer_units_per_copy} polymer units per copy x "
                    f"{order} group actions = {physical_polymer_units}, "
                    "outside requested subunit range "
                    f"{intent.goal.subunits.minimum}.."
                    f"{intent.goal.subunits.maximum}"
                )
        if reasons:
            rejected[symmetry_id] = reasons
        else:
            accepted.append(symmetry_id)
    if rejected and explicitly_requested:
        raise ValueError(
            "Explicit resolver symmetries are incompatible with the "
            f"pre-positioned multi-seed contract: {rejected}"
        )
    if not accepted:
        raise ValueError(
            "No requested symmetry satisfies the pre-positioned multi-seed "
            f"contract: {rejected}"
        )
    return tuple(accepted)


def _reverse_polymer_links(
    links: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((target, source) for source, target in reversed(links))


def _connection_relation_plans(
    symmetry_id: str,
    link_count: int,
) -> tuple[dict[str, Any], ...]:
    """Assign enough group relations to connect every symmetry copy."""

    if link_count < 1:
        raise ValueError("At least one polymer link is required")
    cyclic = _CYCLIC.fullmatch(symmetry_id)
    if cyclic is not None:
        order = int(cyclic.group("order"))
        offsets = (1,) if order == 2 else (1, -1)
        return tuple(
            {
                "label": (
                    f"seam_{seam_index:02d}__winding_"
                    + ("plus1" if offset > 0 else "minus1")
                ),
                "relations": tuple(
                    offset if index == seam_index else 0
                    for index in range(link_count)
                ),
                "primary_index": seam_index,
                "primary_relation": offset,
            }
            for seam_index in range(link_count)
            for offset in offsets
        )

    generators = minimal_group_relations(symmetry_id)
    if len(generators) > link_count:
        raise ValueError(
            f"{symmetry_id} requires {len(generators)} generating "
            f"relations but only {link_count} polymer links are available"
        )
    plans: list[dict[str, Any]] = []
    for link_indices in permutations(range(link_count), len(generators)):
        relations: list[int | str] = [0] * link_count
        for link_index, generator in zip(
            link_indices,
            generators,
            strict=True,
        ):
            relations[link_index] = generator
        assignment = "__".join(
            f"link_{link_index:02d}_{generator.replace(':', '_')}"
            for link_index, generator in zip(
                link_indices,
                generators,
                strict=True,
            )
        )
        plans.append(
            {
                "label": f"generators__{assignment}",
                "relations": tuple(relations),
                "primary_index": link_indices[0],
                "primary_relation": generators[0],
            }
        )
    return tuple(plans)


def _source_chain_ownership_failures(
    directed_links: tuple[tuple[str, str], ...],
    side_records: dict[str, tuple[str, str, str]],
) -> tuple[str, ...]:
    """Reject ordinary candidates that split one known source polymer.

    A source chain reused by several declared interface seeds is evidence
    that those seed faces belong to one physical component.  The current
    binary path-cover frontend can preserve that ownership only when the two
    sides from that chain are joined into one polymer unit.  Treating the
    faces as unrelated components can place a generated terminus directly
    into the occupied continuation of the same source backbone (the failure
    observed for the 7mwr two-patch engineering canary).

    Expert assembly graphs remain free to request a different chain
    topology explicitly.  Ordinary resolution must not silently discard the
    component identity already present in the input file.
    """

    sides_by_chain: dict[str, list[str]] = {}
    for side_id, (_, _, selector) in side_records.items():
        segment = parse_public_selector(selector)[0]
        sides_by_chain.setdefault(segment.chain_id, []).append(side_id)

    linked_pairs = {
        frozenset((source_side, target_side))
        for source_side, target_side in directed_links
    }
    failures: list[str] = []
    for chain_id, side_ids in sorted(sides_by_chain.items()):
        if len(side_ids) < 2:
            continue
        if len(side_ids) > 2:
            failures.append(
                f"source chain {chain_id!r} contributes {len(side_ids)} "
                "interface faces; the binary ordinary resolver cannot "
                "represent that multi-face component without splitting it"
            )
            continue
        if frozenset(side_ids) not in linked_pairs:
            failures.append(
                f"source chain {chain_id!r} is split across different "
                "polymer units instead of preserving its two supplied "
                "interface faces as one component"
            )
    return tuple(failures)


def _enumerate_multi_seed_candidates(
    intent: SimpleCageIntentSpec,
    *,
    symmetry_ids: Iterable[str] | None,
    seed_start: int,
    timesteps: int,
    max_candidates: int,
    global_placement: bool = False,
    pose_samples: int = 1,
) -> tuple[SimpleDesignCandidate, ...]:
    """Freeze supplied interface hyperedges into finite-group hypotheses.

    The input coordinates are authoritative for the relative pose of every
    seed. The topology enumerator chooses which seed sides form polymer
    units. For Cn, one cycle holonomy carries the familiar +/-1 winding. For
    non-cyclic finite groups, independent base-graph cycles carry a minimal
    set of registry-derived generators. The fully expanded incidence graph,
    rather than generator count alone, decides whether a candidate is truly
    connected. Static compilation remains responsible for linker
    reachability, clashes, group closure and exact replay.
    """

    if intent.goal.composition == "homomer":
        raise NotImplementedError(
            "The bounded multi-seed path cover currently emits several "
            "distinct polymer-unit paths per asymmetric unit. Mosaic cannot "
            "claim composition=homomer until those paths are proven "
            "sequence/topology equivalent; use composition=auto for this "
            "pre-positioned multicomponent slice"
        )
    from rfd3_mosaic.compile import expand_symmetry_instances

    seeds = _supported_interface_seeds(intent)
    topology_seeds, side_records = _multi_seed_side_records(seeds)
    if all(len(seed.side_ids) == 2 for seed in topology_seeds):
        # Preserve the historical binary ordering and candidate identities.
        binary_seeds = tuple(
            BinaryInterfaceSeed(
                seed_id=seed.seed_id,
                left_side_id=seed.side_ids[0],
                right_side_id=seed.side_ids[1],
            )
            for seed in topology_seeds
        )
        path_covers = enumerate_directed_polymer_path_covers(
            binary_seeds,
            max_candidates=max_candidates,
        )
        hyperedge_frontend = False
    else:
        path_covers = enumerate_polymer_hyperedge_covers(
            topology_seeds,
            max_candidates=max_candidates,
        )
        hyperedge_frontend = True
    polymer_units_per_copy = sum(
        len(seed.side_ids) for seed in topology_seeds
    ) // 2
    topological_cycle_rank = (
        polymer_units_per_copy - len(topology_seeds) + 1
    )
    symmetry_order = _multi_seed_symmetry_order(
        intent,
        symmetry_ids,
        polymer_units_per_copy=polymer_units_per_copy,
        topological_cycle_rank=topological_cycle_rank,
    )
    total_candidate_count = 0
    for symmetry_id in symmetry_order:
        total_candidate_count += (
            len(path_covers)
            * 2  # both chemical chain directions
            * len(
                _connection_relation_plans(
                    symmetry_id,
                    polymer_units_per_copy,
                )
            )
        )
    if total_candidate_count > max_candidates:
        raise ValueError(
            "Ordinary multi-seed resolution would create "
            f"{total_candidate_count} candidates, exceeding "
            f"max_candidates={max_candidates}; narrow the seed or symmetry "
            "choices"
        )

    components: dict[str, dict[str, Any]] = {}
    ports: dict[str, dict[str, Any]] = {}
    interfaces: list[dict[str, Any]] = []
    hyperedge_members: dict[str, list[str]] = {}
    for interface_id, seed in seeds:
        component_id = f"seed__{interface_id}"
        components[component_id] = {
            "selectors": [
                seed.selectors[participant]
                for participant in seed.participants
            ],
            "geometry": "joint_rigid",
            "pose": {"mode": "fixed"},
        }
        interface_ports: list[str] = []
        for side_index, participant in enumerate(seed.participants):
            port_id = f"port__{interface_id}__{side_index + 1:02d}"
            interface_ports.append(port_id)
            ports[port_id] = {
                "component": component_id,
                "selectors": [seed.selectors[participant]],
            }
        participant_ports = dict(
            zip(seed.participants, interface_ports, strict=True)
        )
        evidence = _validated_input_evidence(intent, interface_id, seed)
        tree = _contact_spanning_tree(
            seed.participants,
            evidence["active_contact_pairs"],
        )
        member_ids: list[str] = []
        for pair_index, (left, right) in enumerate(tree, start=1):
            edge_id = (
                interface_id
                if len(tree) == 1
                else f"{interface_id}__member_{pair_index:02d}"
            )
            member_ids.append(edge_id)
            interfaces.append({
                "id": edge_id,
                "hyperedge_id": interface_id,
                "between": [
                    participant_ports[left],
                    participant_ports[right],
                ],
                "copy_relation": {"orbit_offset": 0},
                "relation": {
                    "mode": "preserve_input",
                    "cutoff": intent.inspection.contact_cutoff,
                    "minimum_heavy_atom_contacts": (
                        intent.inspection.minimum_atom_contacts
                    ),
                },
                "use": seed.use.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "required": True,
            })
        hyperedge_members[interface_id] = member_ids

    candidates: list[SimpleDesignCandidate] = []
    for symmetry_id in symmetry_order:
        order = symmetry_group_action_count(symmetry_id)
        relation_plans = _connection_relation_plans(
            symmetry_id,
            polymer_units_per_copy,
        )
        for path_index, hypothesis in enumerate(path_covers):
            canonical_links = tuple(
                (link.source_side_id, link.target_side_id)
                for link in hypothesis.ordered_links
            )
            directions = (
                ("forward", canonical_links),
                ("reverse", _reverse_polymer_links(canonical_links)),
            )
            for direction_label, directed_links in directions:
                for relation_plan in relation_plans:
                        if len(candidates) >= max_candidates:
                            raise ValueError(
                                "Ordinary multi-seed resolution exceeds "
                                f"max_candidates={max_candidates}; narrow "
                                "the seed or symmetry choices"
                            )
                        candidate_index = len(candidates)
                        candidate_id = f"candidate_{candidate_index:06d}"
                        topology_id = (
                            f"{symmetry_id.lower()}__path_{path_index:04d}__"
                            f"{direction_label}__{relation_plan['label']}"
                        )
                        connection_payloads: list[dict[str, Any]] = []
                        recorded_links: list[
                            tuple[str, str, int | str]
                        ] = []
                        for link_index, (
                            source_side,
                            target_side,
                        ) in enumerate(directed_links):
                            source_interface, _, source_selector = side_records[
                                source_side
                            ]
                            target_interface, _, target_selector = side_records[
                                target_side
                            ]
                            if source_interface == target_interface:
                                raise RuntimeError(
                                    "Path-cover invariant violated: a "
                                    "polymer edge joins the two sides of "
                                    "one supplied interface seed"
                                )
                            relation = relation_plan["relations"][link_index]
                            copy_relation = (
                                {"orbit_offset": relation}
                                if isinstance(relation, int)
                                else {"transform": relation}
                            )
                            connection_payloads.append(
                                {
                                    "id": (
                                        f"polymer_link_{link_index + 1:03d}"
                                    ),
                                    "from": {
                                        "component": (
                                            f"seed__{source_interface}"
                                        ),
                                        "selector": source_selector,
                                        "terminus": "c",
                                    },
                                    "to": {
                                        "component": (
                                            f"seed__{target_interface}"
                                        ),
                                        "selector": target_selector,
                                        "terminus": "n",
                                    },
                                    "length": intent.generation.length,
                                    "copy_relation": copy_relation,
                                }
                            )
                            recorded_links.append(
                                (source_side, target_side, relation)
                            )
                        payload: dict[str, Any] = {
                            "schema_version": 1,
                            "name": (
                                f"{intent.name}-{symmetry_id.lower()}-"
                                f"{candidate_index:04d}"
                            ),
                            "input": str(intent.input),
                            "symmetry": symmetry_id,
                            "components": components,
                            "ports": ports,
                            "interfaces": interfaces,
                            "connections": connection_payloads,
                            "sampling": {
                                "timesteps": timesteps,
                                "seed": seed_start + candidate_index,
                                "preset": "exact_mosaic",
                                "low_memory_mode": True,
                                "execution_backend": "explicit_all_copy",
                            },
                            "resources": intent.resources.model_dump(
                                mode="json",
                                exclude_none=True,
                            ),
                        }
                        if intent.output is not None:
                            payload["output"] = intent.output.model_dump(
                                mode="json",
                                exclude_none=True,
                            )
                        design = UserDesignSpec.model_validate(payload)
                        lowered = lower_user_design(design)
                        topology = analyze_interleaved_interface_seed_topology(
                            expand_symmetry_instances(lowered.specification)
                        )
                        topology_valid = (
                            topology.is_valid_interface_unit_graph
                            if hyperedge_frontend
                            else topology.is_closed_alternating_cycle
                        )
                        if not topology_valid:
                            if _CYCLIC.fullmatch(symmetry_id):
                                raise RuntimeError(
                                    "Generated multi-seed candidate violates "
                                    "the connected interface/unit graph "
                                    "contract: "
                                    + "; ".join(topology.violations)
                                )
                            # A finite-group voltage assignment is only a
                            # hypothesis until the fully expanded incidence
                            # graph proves that its cycle holonomies generate
                            # one connected lift.  Reject this assignment and
                            # continue enumerating the remaining placements.
                            continue
                        expected_unit_count = polymer_units_per_copy * order
                        observed_unit_count = len(topology.polymer_units)
                        if observed_unit_count != expected_unit_count:
                            raise RuntimeError(
                                "Generated multi-seed candidate produced "
                                f"{observed_unit_count} physical polymer "
                                f"units; expected {expected_unit_count}"
                            )
                        primary_index = int(
                            relation_plan["primary_index"]
                        )
                        seam_source, seam_target, primary_relation = (
                            recorded_links[primary_index]
                        )
                        preflight_failures = (
                            _source_chain_ownership_failures(
                                directed_links,
                                side_records,
                            )
                        )
                        candidates.append(
                            SimpleDesignCandidate(
                                candidate_id=candidate_id,
                                symmetry=symmetry_id,
                                topology_id=topology_id,
                                design=design,
                                connection_order=(
                                    seam_source,
                                    seam_target,
                                ),
                                connection_orbit_offset=(
                                    primary_relation
                                    if isinstance(primary_relation, int)
                                    else 0
                                ),
                                connection_group_relation=(
                                    primary_relation
                                    if isinstance(primary_relation, str)
                                    else None
                                ),
                                resolution_frontend=(
                                    "prepositioned_multi_interface_"
                                    "hyperedge_v1"
                                    if hyperedge_frontend
                                    else (
                                        "prepositioned_multi_binary_cn_"
                                        "experimental"
                                        if _CYCLIC.fullmatch(symmetry_id)
                                        else
                                        "prepositioned_multi_interface_"
                                        "finite_group_v1"
                                    )
                                ),
                                polymer_links=tuple(recorded_links),
                                polymer_units_per_copy=(
                                    polymer_units_per_copy
                                ),
                                physical_polymer_unit_count=(
                                    observed_unit_count
                                ),
                                expanded_topology_status=topology.status,
                                preflight_failures=preflight_failures,
                                unresolved_variables=(),
                                interface_hyperedges=tuple(
                                    (
                                        interface_id,
                                        tuple(member_ids),
                                    )
                                    for interface_id, member_ids in sorted(
                                        hyperedge_members.items()
                                    )
                                ),
                            )
                        )
    if not candidates:
        raise ValueError(
            "No connected finite-group multi-seed hypothesis remains after "
            "expanding and validating all group-relation assignments. The "
            "supplied interface/polymer topology may not contain enough "
            "independent cycles for the requested symmetry"
        )
    if not global_placement:
        return tuple(candidates)
    if pose_samples < 1:
        raise ValueError("global pose_samples must be positive")
    expanded_count = len(candidates) * pose_samples
    if expanded_count > max_candidates:
        raise ValueError(
            "Global multi-seed placement would create "
            f"{expanded_count} topology/pose candidates, exceeding "
            f"max_candidates={max_candidates}; reduce pose samples or "
            "narrow symmetry choices"
        )
    diameter_range = (
        (
            intent.goal.diameter_angstrom.minimum,
            intent.goal.diameter_angstrom.maximum,
        )
        if intent.goal.diameter_angstrom is not None
        else None
    )
    expanded: list[SimpleDesignCandidate] = []
    for candidate in candidates:
        for pose_sample_index in range(pose_samples):
            index = len(expanded)
            design = initialize_global_seed_layout(
                candidate.design,
                sample_index=pose_sample_index,
                sample_count=pose_samples,
                diameter_range=diameter_range,
            )
            sampling = design.sampling.model_copy(
                update={"seed": seed_start + index}
            )
            design = design.model_copy(
                update={
                    "name": f"{candidate.design.name}-p{pose_sample_index:03d}",
                    "sampling": sampling,
                }
            )
            expanded.append(
                replace(
                    candidate,
                    candidate_id=f"candidate_{index:06d}",
                    topology_id=(
                        f"{candidate.topology_id}__pose_"
                        f"{pose_sample_index:03d}"
                    ),
                    design=design,
                    pose_sample_index=pose_sample_index,
                    global_pose_initialization=True,
                    resolution_frontend=(
                        "independent_multi_interface_global_pose_v1"
                        if hyperedge_frontend
                        else (
                            "independent_multi_binary_cn_global_pose_v1"
                            if _CYCLIC.fullmatch(candidate.symmetry)
                            else
                            "independent_multi_interface_"
                            "finite_group_global_pose_v1"
                        )
                    ),
                )
            )
    return tuple(expanded)


def enumerate_simple_design_candidates(
    intent: SimpleCageIntentSpec,
    *,
    symmetry_ids: Iterable[str] | None = None,
    seed_start: int = 0,
    timesteps: int = 200,
    max_candidates: int = 4096,
    global_placement: bool = False,
    pose_samples: int = 1,
) -> tuple[SimpleDesignCandidate, ...]:
    """Enumerate deterministic executable hypotheses for supported Cn seeds.

    No candidate is silently chosen.  The returned designs still have to pass
    static compilation and strict replay before they are publishable.
    """

    if seed_start < 0:
        raise ValueError("seed_start cannot be negative")
    if not 2 <= timesteps <= 200:
        raise ValueError("timesteps must be between 2 and 200")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    _validate_resolution_contract(
        intent,
        allow_continuous_geometry=global_placement,
    )
    if len(intent.interface_seeds) > 1:
        return _enumerate_multi_seed_candidates(
            intent,
            symmetry_ids=symmetry_ids,
            seed_start=seed_start,
            timesteps=timesteps,
            max_candidates=max_candidates,
            global_placement=global_placement,
            pose_samples=pose_samples,
        )
    interface_id, seed = _supported_seed(intent)
    if len(seed.participants) != 2:
        raise NotImplementedError(
            f"A single {len(seed.participants)}-participant interface "
            "hyperedge "
            "does not determine a unique polymer connectivity by itself. "
            "Provide at least one additional supplied interface seed or use "
            "expert connections; Mosaic will not invent covalent links "
            "between participants of the same supplied interface"
        )
    symmetry_order = _resolver_symmetry_order(intent, symmetry_ids)

    left, right = seed.participants
    selectors = seed.selectors
    directions = ((right, left), (left, right))
    candidates: list[SimpleDesignCandidate] = []
    for symmetry_id in symmetry_order:
        match = _CYCLIC.fullmatch(symmetry_id)
        if match is None:
            continue
        order = int(match.group("order"))
        finite_action = _single_seed_finite_action(
            intent,
            symmetry_id=symmetry_id,
            interface_id=interface_id,
            seed=seed,
        )
        physical_copy_count = (
            len(finite_action.coset_representative_ids)
            if finite_action is not None
            else order
        )
        offsets = (1,) if physical_copy_count == 2 else (1, -1)
        for from_participant, to_participant in directions:
            for offset in offsets:
                candidate_index = len(candidates)
                direction_label = (
                    f"{from_participant}_to_{to_participant}"
                )
                offset_label = "plus1" if offset > 0 else "minus1"
                topology_id = f"{direction_label}__{offset_label}"
                candidate_id = f"candidate_{candidate_index:06d}"
                component_id = f"seed__{interface_id}"
                left_port = f"port__{interface_id}__left"
                right_port = f"port__{interface_id}__right"
                payload: dict[str, Any] = {
                    "schema_version": 1,
                    "name": (
                        f"{intent.name}-{symmetry_id.lower()}-"
                        f"{candidate_index:04d}"
                    ),
                    "input": str(intent.input),
                    "symmetry": (
                        {
                            "id": symmetry_id,
                            "axis": finite_action.symmetry_axis,
                            "center": finite_action.symmetry_center,
                        }
                        if finite_action is not None
                        else symmetry_id
                    ),
                    "finite_orbit_action": (
                        finite_action.finite_action_payload
                        if finite_action is not None
                        else None
                    ),
                    "components": {
                        component_id: {
                            "selectors": [selectors[left], selectors[right]],
                            "geometry": "joint_rigid",
                            "pose": {"mode": "fixed"},
                        }
                    },
                    "ports": {
                        left_port: {
                            "component": component_id,
                            "selectors": [selectors[left]],
                        },
                        right_port: {
                            "component": component_id,
                            "selectors": [selectors[right]],
                        },
                    },
                    "interfaces": [
                        {
                            "id": interface_id,
                            "between": [left_port, right_port],
                            "copy_relation": {"orbit_offset": 0},
                            "relation": {
                                "mode": "preserve_input",
                                "cutoff": (
                                    intent.inspection.contact_cutoff
                                ),
                                "minimum_heavy_atom_contacts": (
                                    intent.inspection.minimum_atom_contacts
                                ),
                            },
                            "use": seed.use.model_dump(
                                mode="json", exclude_none=True
                            ),
                            "required": True,
                        }
                    ],
                    "connections": [
                        {
                            "id": "polymer_unit",
                            "from": {
                                "component": component_id,
                                "selector": selectors[from_participant],
                                "terminus": "c",
                            },
                            "to": {
                                "component": component_id,
                                "selector": selectors[to_participant],
                                "terminus": "n",
                            },
                            "length": intent.generation.length,
                            "tie_group": "polymer_unit_length",
                            "copy_relation": {"orbit_offset": offset},
                        }
                    ],
                    "sampling": {
                        "timesteps": timesteps,
                        "seed": seed_start + candidate_index,
                        "preset": "exact_mosaic",
                        "low_memory_mode": True,
                        "execution_backend": "explicit_all_copy",
                    },
                    "resources": intent.resources.model_dump(
                        mode="json", exclude_none=True
                    ),
                }
                if intent.output is not None:
                    payload["output"] = intent.output.model_dump(
                        mode="json", exclude_none=True
                    )
                design = UserDesignSpec.model_validate(payload)
                candidates.append(
                    SimpleDesignCandidate(
                        candidate_id=candidate_id,
                        symmetry=symmetry_id,
                        topology_id=topology_id,
                        design=design,
                        connection_order=(
                            from_participant,
                            to_participant,
                        ),
                        connection_orbit_offset=offset,
                        stabilizer_evidence=(
                            finite_action.to_dict()
                            if finite_action is not None
                            else None
                        ),
                    )
                )
                if len(candidates) > max_candidates:
                    raise ValueError(
                        "Ordinary resolution exceeds "
                        f"max_candidates={max_candidates}"
                    )
    if not candidates:
        raise NotImplementedError(
            "No executable Cn ring hypothesis remains. The current resolver "
            "does not yet freeze Dn/T/O/I cage connection transforms"
        )
    return tuple(candidates)


def resolve_simple_intent(
    intent: SimpleCageIntentSpec,
    output_directory: str | Path,
    *,
    source_path: str | Path | None = None,
    symmetry_ids: Iterable[str] | None = None,
    pose_samples: int = 1,
    seed_start: int = 0,
    timesteps: int = 200,
    top_count: int = 20,
    max_candidates: int = 4096,
    optimize_poses: bool = False,
    pose_optimize_top: int = 4,
    pose_optimization_levels: int = 3,
    pose_maximum_translation: float = 12.0,
    pose_maximum_rotation_deg: float = 25.0,
) -> dict[str, Any]:
    """Resolve, rank and freeze ordinary candidates through the expert path."""

    if pose_samples < 1:
        raise ValueError("pose_samples must be positive")
    output = Path(output_directory).expanduser().resolve()
    materialized = materialize_seed_library(
        intent,
        output / "_inputs",
    )
    working_intent = materialized.intent
    if pose_samples != 1 and not materialized.independent_frames:
        raise NotImplementedError(
            "The first ordinary resolver keeps the supplied seed pose. "
            "Multiple global pose starts apply only to independently "
            "supplied seed files"
        )
    effective_pose_samples = (
        max(8, pose_samples)
        if materialized.independent_frames
        else 1
    )
    seed_records = _supported_binary_seeds(working_intent)
    evidence_records = [
        _validated_input_evidence(working_intent, interface_id, seed)
        for interface_id, seed in seed_records
    ]
    if len(seed_records) > 1:
        _validate_multi_seed_backbone_anchors(working_intent)
    candidates = enumerate_simple_design_candidates(
        working_intent,
        symmetry_ids=symmetry_ids,
        seed_start=seed_start,
        timesteps=timesteps,
        max_candidates=max_candidates,
        global_placement=materialized.independent_frames,
        pose_samples=effective_pose_samples,
    )
    candidate_payloads = tuple(
        (candidate.candidate_id, candidate.design, candidate.metadata())
        for candidate in candidates
    )
    pose_optimization_applied = (
        (optimize_poses or materialized.independent_frames)
        and len(seed_records) > 1
    )
    if pose_optimization_applied:
        candidate_payloads = optimize_candidate_subset(
            candidate_payloads,
            top_count=pose_optimize_top,
            levels=pose_optimization_levels,
            maximum_translation=pose_maximum_translation,
            maximum_rotation_deg=pose_maximum_rotation_deg,
        )
    ranked = rank_design_candidates(
        candidate_payloads,
        output_directory,
        top_count=top_count,
    )
    normalized_intent_path = output / "normalized_intent.yaml"
    normalized_intent_path.write_text(
        yaml.safe_dump(
            working_intent.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else None
    )
    for item in ranked["ranking"]:
        resolved = item.get("resolved_design")
        if resolved:
            item["resolved_design_sha256"] = _sha256(Path(resolved))
    recommended = next(
        (
            item
            for item in ranked["ranking"]
            if item.get("resolved_design")
        ),
        None,
    )
    multi_seed = len(seed_records) > 1
    hyperedge_seed = any(
        len(seed.participants) > 2 for _, seed in seed_records
    )
    manifest = {
        "schema_version": 1,
        "resolver": (
            "rfd3_mosaic.independent_multi_interface_finite_group_v1"
            if multi_seed
            and materialized.independent_frames
            and hyperedge_seed
            else "rfd3_mosaic.independent_multi_seed_global_cn_v1"
            if multi_seed and materialized.independent_frames
            else "rfd3_mosaic.prepositioned_multi_interface_finite_group_v1"
            if multi_seed and hyperedge_seed
            else "rfd3_mosaic.prepositioned_multi_binary_cn_v1"
            if multi_seed
            else "rfd3_mosaic.simple_binary_cn_ring_v1"
        ),
        "source_intent": str(source) if source is not None else None,
        "source_intent_sha256": (
            _sha256(source)
            if source is not None and source.is_file()
            else None
        ),
        "normalized_intent": str(normalized_intent_path),
        "normalized_intent_sha256": _sha256(normalized_intent_path),
        "input_structure": str(working_intent.input),
        "input_structure_sha256": _sha256(working_intent.input),
        "seed_library": materialized.manifest,
        "input_evidence": (
            evidence_records
            if multi_seed
            else evidence_records[0]
        ),
        "authoring_mode": "ordinary",
        "execution_path": (
            "UserDesignSpec -> AssemblySpecification -> Mosaic-RFD3"
        ),
        "automatic_selection": recommended is not None,
        "recommendation_scope": (
            "best_cpu_feasible_topology_and_initial_pose_before_diffusion"
        ),
        "recommended_design": (
            recommended.get("resolved_design")
            if recommended is not None
            else None
        ),
        "recommended_candidate_id": (
            recommended.get("candidate_id")
            if recommended is not None
            else None
        ),
        "continuous_pose_optimization": {
            "requested": optimize_poses,
            "required_by_independent_seed_library": (
                materialized.independent_frames
            ),
            "enabled": pose_optimization_applied,
            "reason": (
                "multi_seed_joint_pose_search"
                if pose_optimization_applied
                else "single_joint_component_has_no_relative_seed_pose"
                if optimize_poses
                else "disabled_by_user"
            ),
            "shortlist_size": (
                pose_optimize_top if pose_optimization_applied else 0
            ),
            "levels": (
                pose_optimization_levels if pose_optimization_applied else 0
            ),
            "maximum_translation": (
                pose_maximum_translation
                if pose_optimization_applied
                else None
            ),
            "maximum_rotation_deg": (
                pose_maximum_rotation_deg
                if pose_optimization_applied
                else None
            ),
            "global_initialization": materialized.independent_frames,
            "global_pose_samples": effective_pose_samples,
            "input_file_frames_ignored": materialized.independent_frames,
            "hard_constraints": [
                "zero_inter_group_hard_clashes",
                "all_continuous_links_within_maximum_contour",
                "all_linker_chords_clear_fixed_seed_atoms",
                "all_required_input_interfaces_satisfied",
                "all_required_static_objectives_satisfied",
                "strict_frozen_replay",
            ],
        },
        "selection_required": ranked["selected_count"] > 1,
        "supported_contract": (
            "several independent binary preserve_exact seed files; "
            "canonical local frames; deterministic global pose starts; "
            "bounded joint SE(3) refinement; path-cover; full-orbit Cn "
            "winding"
            if multi_seed and materialized.independent_frames
            else "several disjoint pre-positioned binary preserve_exact "
            "seeds; bounded path-cover; full-orbit Cn winding"
            if multi_seed
            else "one binary preserve_exact seed; full-orbit Cn ring"
        ),
        **ranked,
    }
    manifest_path = output / "resolution_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = [
    "SimpleDesignCandidate",
    "enumerate_simple_design_candidates",
    "resolve_simple_intent",
]
