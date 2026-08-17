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
from itertools import combinations, permutations
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import yaml

from rfd3_mosaic.design_compiler import (
    lower_user_design,
    parse_public_selector,
)
from rfd3_mosaic.graph_search import rank_design_candidates
from rfd3_mosaic.geometry import build_transform_registry
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
    DirectedPolymerLink,
    InterfaceHyperedgeSeed,
    PolymerUnitPathCoverHypothesis,
    enumerate_directed_polymer_path_covers,
    enumerate_polymer_hyperedge_covers,
    enumerate_polymer_unit_path_covers,
)
from rfd3_mosaic.topology.interface_seed_graph import (
    analyze_interleaved_interface_seed_topology,
)
from rfd3_mosaic.topology.component_incidence import (
    enumerate_binary_interface_incidence_plans,
)
from rfd3_mosaic.topology.symmetry_connectivity import (
    finite_symmetry_spec,
    generated_transform_ids,
    minimal_group_relations,
)


_CYCLIC = re.compile(r"^C(?P<order>[2-9][0-9]*)$")


def _ordinary_component_pose(intent: SimpleCageIntentSpec) -> dict[str, Any]:
    """Lower one high-level motion choice without exposing SE(3) tuning."""

    motion = intent.preferences.component_motion
    if motion is None or motion.value == "locked":
        return {"mode": "fixed"}
    return {
        "mode": "bounded_mobile",
        "subspace": (
            "bounded_se3"
            if motion.value == "free"
            else "radial_axial_rotation"
        ),
        "proposal": "scaffold_objectives",
        "max_translation": 4.0,
        "max_rotation_deg": 10.0,
        "start_fraction": 0.05,
        "end_fraction": 0.75,
        "response": 0.2,
        "max_step_translation": 0.25,
        "max_step_rotation_deg": 1.0,
    }


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
    physical_interface_count: int | None = None
    component_orbit_multiplicities: tuple[tuple[str, int], ...] = ()
    expanded_topology_status: str | None = None
    unresolved_variables: tuple[str, ...] = ()
    preflight_failures: tuple[str, ...] = ()
    pose_sample_index: int = 0
    global_pose_initialization: bool = False
    interface_hyperedges: tuple[tuple[str, tuple[str, ...]], ...] = ()
    stabilizer_evidence: dict[str, Any] | None = None
    topology_search_complete: bool = True
    supplied_interface_ids: tuple[str, ...] = ()

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
            "physical_interface_count": self.physical_interface_count,
            "component_orbit_multiplicities": dict(
                self.component_orbit_multiplicities
            ),
            "expanded_topology_status": self.expanded_topology_status,
            "unresolved_variables": list(self.unresolved_variables),
            "preflight_failures": list(self.preflight_failures),
            "pose_sample_index": self.pose_sample_index,
            "global_pose_initialization": self.global_pose_initialization,
            "global_pose_initializer": (
                "polyhedral_spherical_low_discrepancy_v1"
                if self.global_pose_initialization
                and self.symmetry in {"T", "O", "I"}
                else "cyclic_dihedral_low_discrepancy_v1"
                if self.global_pose_initialization
                else None
            ),
            "interface_hyperedges": {
                interface_id: list(member_ids)
                for interface_id, member_ids in self.interface_hyperedges
            },
            "topology_search_complete": self.topology_search_complete,
            "supplied_interface_ids": list(self.supplied_interface_ids),
            "invented_interface_count": 0,
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
    # Multi-range selectors are handled by the mixed-component incidence
    # frontend when one supplied interface joins oligomeric building blocks.
    # The historical single-fragment path-cover frontends retain their
    # stricter one-range-per-side contract below.


def _requires_mixed_component_incidence(seed) -> bool:
    return len(seed.participants) == 2 and any(
        len(parse_public_selector(seed.selectors[participant])) > 1
        for participant in seed.participants
    )


def _component_protomer_paths(
    *,
    interface_id: str,
    participant: str,
    selector: str,
) -> dict[str, tuple[Any, ...]]:
    """Resolve explicit source-chain contigs for one oligomer participant."""

    by_chain: dict[str, list[Any]] = {}
    for segment in parse_public_selector(selector):
        by_chain.setdefault(segment.chain_id, []).append(segment)
    if len(by_chain) < 2:
        raise ValueError(
            f"Mixed-component interface {interface_id!r} participant "
            f"{participant!r} must select at least two source protomer "
            "chains; use the ordinary single-component resolver otherwise"
        )
    ordered: dict[str, tuple[Any, ...]] = {}
    for chain_id, segments in by_chain.items():
        chain_segments = tuple(
            sorted(segments, key=lambda item: (item.residue_start, item.residue_end))
        )
        if len(chain_segments) < 2:
            raise NotImplementedError(
                f"Mixed-component interface {interface_id!r} participant "
                f"{participant!r} chain {chain_id!r} has only one fixed "
                "fragment. Mosaic cannot invent whether to generate an N "
                "terminus, C terminus, or cross-chain fusion; provide at "
                "least two ordered fixed fragments on that chain or use "
                "expert connections"
            )
        for left, right in zip(chain_segments, chain_segments[1:]):
            if left.residue_end >= right.residue_start:
                raise ValueError(
                    f"Mixed-component participant {participant!r} has "
                    f"overlapping selectors on chain {chain_id!r}"
                )
        ordered[chain_id] = chain_segments
    return ordered


def _action_payload(action) -> dict[str, Any]:
    return {
        "coset_representative_ids": list(action.coset_representative_ids),
        "stabilizer_transform_ids": list(action.stabilizer_transform_ids),
        "transform_to_coset_representative": dict(
            action.transform_to_coset_representative
        ),
    }


def _enumerate_mixed_component_interface_candidates(
    intent: SimpleCageIntentSpec,
    *,
    symmetry_ids: Iterable[str] | None,
    seed_start: int,
    timesteps: int,
    max_candidates: int,
) -> tuple[SimpleDesignCandidate, ...]:
    """Lower one supplied oligomer--oligomer interface without inventing it.

    Participant valencies come only from the source-chain membership declared
    by the user.  Finite-group incidence chooses compatible component orbits;
    the structure-aware compiler then proves the selected Cn/Dn/T/O/I
    stabilizers against the supplied coordinates before a candidate survives.
    """

    interface_id, seed = next(iter(intent.interface_seeds.items()))
    left_id, right_id = seed.participants
    paths = {
        participant: _component_protomer_paths(
            interface_id=interface_id,
            participant=participant,
            selector=seed.selectors[participant],
        )
        for participant in seed.participants
    }
    # The user supplied this interface; verify that the selected oligomeric
    # sides actually contact in the input before any symmetry is considered.
    _validated_input_evidence(intent, interface_id, seed)
    valencies = {
        participant: len(paths[participant])
        for participant in seed.participants
    }
    requested = (
        tuple(symmetry_ids)
        if symmetry_ids is not None
        else candidate_symmetries(intent)
    )
    if not requested:
        raise ValueError("At least one mixed-component symmetry is required")
    explicit = symmetry_ids is not None or intent.goal.symmetry != "auto"
    rejections: dict[str, list[str]] = {}
    candidates: list[SimpleDesignCandidate] = []
    for symmetry_id in requested:
        reasons: list[str] = []
        try:
            group_order = symmetry_group_action_count(symmetry_id)
        except ValueError as error:
            reasons.append(str(error))
            rejections[symmetry_id] = reasons
            continue
        if not seed.use.accepts(group_order):
            reasons.append(
                f"interface {interface_id} requests {seed.use.description}, "
                f"but one free {symmetry_id} interface orbit has "
                f"{group_order} physical instances"
            )
            rejections[symmetry_id] = reasons
            continue
        try:
            plans = tuple(
                plan
                for plan in enumerate_binary_interface_incidence_plans(
                    symmetry=symmetry_id,
                    interface_id=interface_id,
                    left_participant=left_id,
                    right_participant=right_id,
                    physical_interface_count=group_order,
                    minimum_valency=min(valencies.values()),
                    maximum_valency=max(valencies.values()),
                    max_candidates=max_candidates,
                )
                if plan.left.valency == valencies[left_id]
                and plan.right.valency == valencies[right_id]
            )
        except (NotImplementedError, ValueError) as error:
            reasons.append(str(error))
            plans = ()
        if not plans:
            reasons.append(
                f"no {symmetry_id} incidence action has user-supplied "
                f"participant valencies {valencies[left_id]}/"
                f"{valencies[right_id]}"
            )
            rejections[symmetry_id] = reasons
            continue

        for plan_index, plan in enumerate(plans):
            if len(candidates) >= max_candidates:
                raise ValueError(
                    "Mixed-component resolution exceeds "
                    f"max_candidates={max_candidates}; narrow symmetry"
                )
            components: dict[str, dict[str, Any]] = {}
            ports: dict[str, dict[str, Any]] = {}
            connections: list[dict[str, Any]] = []
            for participant, participant_plan in (
                (left_id, plan.left),
                (right_id, plan.right),
            ):
                selectors = [
                    segment.public_expression
                    for chain_segments in paths[participant].values()
                    for segment in chain_segments
                ]
                component_id = f"component__{participant}"
                port_id = f"port__{participant}"
                components[component_id] = {
                    "selectors": selectors,
                    "geometry": "joint_rigid",
                    "finite_orbit_action": _action_payload(
                        participant_plan.action
                    ),
                    "pose": _ordinary_component_pose(intent),
                }
                ports[port_id] = {
                    "component": component_id,
                    "selectors": selectors,
                }
                for chain_id, chain_segments in paths[participant].items():
                    for gap_index, (left, right) in enumerate(
                        zip(chain_segments, chain_segments[1:]),
                        start=1,
                    ):
                        connections.append({
                            "id": (
                                f"path__{participant}__{chain_id}__"
                                f"{gap_index:02d}"
                            ),
                            "from": {
                                "component": component_id,
                                "selector": left.public_expression,
                                "terminus": "c",
                            },
                            "to": {
                                "component": component_id,
                                "selector": right.public_expression,
                                "terminus": "n",
                            },
                            "length": intent.generation.length,
                            "copy_relation": {"orbit_offset": 0},
                        })
            payload: dict[str, Any] = {
                "schema_version": 1,
                "task": "preserve_supplied_geometry",
                "name": (
                    f"{intent.name}-{symmetry_id.lower()}-mixed-"
                    f"{plan_index:04d}"
                ),
                "input": str(intent.input),
                "symmetry": symmetry_id,
                "components": components,
                "ports": ports,
                "interfaces": [{
                    "id": interface_id,
                    "between": [f"port__{left_id}", f"port__{right_id}"],
                    "copy_relation": {"orbit_offset": 0},
                    "relation": {
                        "mode": "preserve_input",
                        "cutoff": intent.inspection.contact_cutoff,
                        "minimum_heavy_atom_contacts": (
                            intent.inspection.minimum_atom_contacts
                        ),
                    },
                    "use": seed.use.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "required": True,
                }],
                "connections": connections,
                "sampling": {
                    "timesteps": timesteps,
                    "seed": seed_start + len(candidates),
                    "preset": "exact_mosaic",
                    "low_memory_mode": True,
                    "execution_backend": "explicit_all_copy",
                },
                "resources": intent.resources.model_dump(
                    mode="json", exclude_none=True
                ),
                "preferences": intent.preferences.model_dump(
                    mode="json", exclude_none=True
                ),
                "assembly_shape": _assembly_shape_payload(intent),
            }
            if intent.output is not None:
                payload["output"] = intent.output.model_dump(
                    mode="json", exclude_none=True
                )
            try:
                design = UserDesignSpec.model_validate(payload)
                lowered = lower_user_design(design)
            except ValueError as error:
                reasons.append(
                    f"incidence plan {plan_index} failed supplied-geometry "
                    f"validation: {error}"
                )
                continue
            from rfd3_mosaic.compile import expand_symmetry_instances

            instances = expand_symmetry_instances(lowered.specification)
            if len(instances.interfaces) != group_order:
                raise RuntimeError(
                    f"Mixed-component lowering emitted "
                    f"{len(instances.interfaces)} physical interfaces; "
                    f"expected {group_order}"
                )
            candidate_index = len(candidates)
            candidates.append(SimpleDesignCandidate(
                candidate_id=f"candidate_{candidate_index:06d}",
                symmetry=symmetry_id,
                topology_id=(
                    f"{symmetry_id.lower()}__mixed_component__"
                    f"valency_{valencies[left_id]}_{valencies[right_id]}__"
                    f"plan_{plan_index:04d}"
                ),
                design=design,
                connection_order=(left_id, right_id),
                connection_orbit_offset=0,
                resolution_frontend=(
                    "supplied_oligomer_interface_incidence_v1"
                ),
                polymer_units_per_copy=(
                    len(paths[left_id]) + len(paths[right_id])
                ),
                physical_polymer_unit_count=len(
                    instances.generated_segments
                ),
                physical_interface_count=len(instances.interfaces),
                component_orbit_multiplicities=(
                    (
                        f"component__{left_id}",
                        plan.left.physical_component_count,
                    ),
                    (
                        f"component__{right_id}",
                        plan.right.physical_component_count,
                    ),
                ),
                expanded_topology_status="connected_component_incidence",
                stabilizer_evidence={
                    "left_action": _action_payload(plan.left.action),
                    "right_action": _action_payload(plan.right.action),
                    "physical_edges": [list(edge) for edge in plan.physical_edges],
                },
                supplied_interface_ids=(interface_id,),
            ))
        if reasons:
            rejections[symmetry_id] = reasons
    if not candidates:
        boundary = "Explicit" if explicit else "Automatic"
        raise ValueError(
            f"{boundary} mixed-component resolution produced no executable "
            f"candidate: {rejections}"
        )
    return tuple(candidates)


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
    mixed_component_seed = (
        len(seeds) == 1
        and _requires_mixed_component_incidence(seeds[0][1])
    )
    for interface_id, seed in seeds:
        _validate_seed_contract(interface_id, seed)
        for participant in seed.participants:
            selector = seed.selectors[participant]
            segments = tuple(parse_public_selector(selector))
            chains = {segment.chain_id for segment in segments}
            if len(chains) != 1 and not mixed_component_seed:
                raise NotImplementedError(
                    f"Multi-seed interface {interface_id!r} participant "
                    f"{participant!r} selects several source chains. "
                    "Ordinary mode accepts any number of ordered fixed "
                    "fragments on one source polymer, but cross-chain "
                    "covalent topology requires expert component paths"
                )
            for chain_id in chains:
                ordered = tuple(
                    sorted(
                        (
                            segment
                            for segment in segments
                            if segment.chain_id == chain_id
                        ),
                        key=lambda item: (
                            item.residue_start,
                            item.residue_end,
                        ),
                    )
                )
                for left, right in zip(ordered, ordered[1:]):
                    if left.residue_end >= right.residue_start:
                        raise ValueError(
                            f"Multi-seed interface {interface_id!r} "
                            f"participant {participant!r} has overlapping "
                            f"fixed ranges on chain {chain_id!r}"
                        )

    selected_ranges: list[tuple[str, str, int, int]] = []
    for interface_id, seed in seeds:
        for participant in seed.participants:
            for segment in parse_public_selector(
                seed.selectors[participant]
            ):
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
                            "Executable supplied-interface resolution "
                            "requires disjoint selected fragments; "
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
        if len(intent.interface_seeds) < 2 or not intent.polymer_connections:
            raise NotImplementedError(
                "composition=heteromer requires at least two supplied "
                "interface seeds and user-declared polymer_connections so "
                "component ownership is authoritative rather than invented"
            )
    # Diameter and cavity ranges lower into required standalone objectives.
    # A pre-positioned design is measured and accepted/rejected as supplied;
    # an unknown-pose design additionally uses the same objective report in
    # continuous pose restoration.  There is no separate shape filter.


def _assembly_shape_payload(
    intent: SimpleCageIntentSpec,
) -> dict[str, Any] | None:
    diameter = intent.goal.diameter_angstrom
    cavity = intent.goal.cavity_diameter_angstrom
    if diameter is None and cavity is None:
        return None
    payload: dict[str, Any] = {}
    if diameter is not None:
        payload["diameter_angstrom"] = diameter.model_dump(mode="json")
    if cavity is not None:
        payload["cavity_diameter_angstrom"] = cavity.model_dump(mode="json")
    return payload


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


def _enumerate_single_supplied_hyperedge_candidates(
    intent: SimpleCageIntentSpec,
    *,
    interface_id: str,
    seed,
    symmetry_ids: Iterable[str] | None,
    seed_start: int,
    timesteps: int,
    max_candidates: int,
) -> tuple[SimpleDesignCandidate, ...]:
    """Rebuild explicit participant-chain paths around one rigid hyperedge.

    A cooperative supplied interface does not, by itself, authorize Mosaic to
    invent covalent links between its participants.  This executable slice is
    therefore intentionally strict: every participant must provide at least
    two ordered fragments on one source chain.  Only the missing intervals on
    that same chain become generated connections.  The complete interface is
    one joint-rigid component throughout lowering and inference.
    """

    participant_segments: dict[str, tuple[Any, ...]] = {}
    for participant in seed.participants:
        segments = tuple(
            sorted(
                parse_public_selector(seed.selectors[participant]),
                key=lambda item: (
                    item.chain_id,
                    item.residue_start,
                    item.residue_end,
                ),
            )
        )
        chain_ids = {segment.chain_id for segment in segments}
        if len(chain_ids) != 1 or len(segments) < 2:
            raise NotImplementedError(
                f"Single multi-participant interface {interface_id!r} does "
                "not determine polymer connectivity for participant "
                f"{participant!r}. Provide at least two ordered fragments "
                "on one source chain, add another supplied interface seed, "
                "or use expert connections; Mosaic will not invent "
                "covalent links between interface participants"
            )
        for left, right in zip(segments, segments[1:]):
            if left.residue_end >= right.residue_start:
                raise ValueError(
                    f"Participant {participant!r} interface fragments "
                    "must be disjoint and ordered from N to C"
                )
        participant_segments[participant] = segments

    evidence = _validated_input_evidence(intent, interface_id, seed)
    symmetry_order = _resolver_symmetry_order(intent, symmetry_ids)
    contact_tree = _contact_spanning_tree(
        seed.participants,
        evidence["active_contact_pairs"],
    )
    candidates: list[SimpleDesignCandidate] = []
    for symmetry_id in symmetry_order:
        if len(candidates) >= max_candidates:
            raise ValueError(
                "Ordinary single-hyperedge resolution exceeds "
                f"max_candidates={max_candidates}"
            )
        finite_action = _single_seed_finite_action(
            intent,
            symmetry_id=symmetry_id,
            interface_id=interface_id,
            seed=seed,
        )
        order = symmetry_group_action_count(symmetry_id)
        physical_copy_count = (
            len(finite_action.coset_representative_ids)
            if finite_action is not None
            else order
        )
        component_id = f"seed__{interface_id}"
        component_selectors = [
            segment.assembly_expression
            for participant in seed.participants
            for segment in participant_segments[participant]
        ]
        port_ids = {
            participant: f"port__{interface_id}__{participant}"
            for participant in seed.participants
        }
        ports = {
            port_ids[participant]: {
                "component": component_id,
                "selectors": [
                    segment.assembly_expression
                    for segment in participant_segments[participant]
                ],
            }
            for participant in seed.participants
        }
        connections: list[dict[str, Any]] = []
        recorded_links: list[tuple[str, str, int | str]] = []
        connection_ids_by_participant: dict[str, list[str]] = {
            participant: [] for participant in seed.participants
        }
        for participant in seed.participants:
            segments = participant_segments[participant]
            for link_index, (left, right) in enumerate(
                zip(segments, segments[1:]),
                start=1,
            ):
                left_selector = left.assembly_expression
                right_selector = right.assembly_expression
                connection_id = (
                    f"participant__{participant}__link_{link_index:02d}"
                )
                connections.append(
                    {
                        "id": connection_id,
                        "from": {
                            "component": component_id,
                            "selector": left_selector,
                            "terminus": "c",
                        },
                        "to": {
                            "component": component_id,
                            "selector": right_selector,
                            "terminus": "n",
                        },
                        "length": intent.generation.length,
                        "copy_relation": {"orbit_offset": 0},
                    }
                )
                connection_ids_by_participant[participant].append(
                    connection_id
                )
                recorded_links.append(
                    (left_selector, right_selector, 0)
                )

        member_ids = tuple(
            interface_id
            if len(contact_tree) == 1
            else f"{interface_id}__member_{index:02d}"
            for index in range(1, len(contact_tree) + 1)
        )
        finite_action_payload = (
            dict(finite_action.finite_action_payload)
            if finite_action is not None
            else None
        )
        if finite_action_payload is not None and physical_copy_count == 1:
            transform_by_participant = {
                participant: transform_id
                for transform_id, participant
                in finite_action.canonical_to_participant
            }
            if set(transform_by_participant) != set(seed.participants):
                raise ValueError(
                    "Stabilizer evidence does not assign every supplied "
                    "interface participant to one canonical transform"
                )
            finite_action_payload["stabilizer_path_transform_ids"] = {
                connection_id: transform_by_participant[participant]
                for participant in seed.participants
                for connection_id in connection_ids_by_participant[
                    participant
                ]
            }

        payload: dict[str, Any] = {
            "schema_version": 1,
            "task": "preserve_supplied_geometry",
            "name": (
                f"{intent.name}-{symmetry_id.lower()}-"
                f"{len(candidates):04d}"
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
                finite_action_payload
            ),
            "components": {
                component_id: {
                    "selectors": component_selectors,
                    "geometry": "joint_rigid",
                    "pose": _ordinary_component_pose(intent),
                }
            },
            "ports": ports,
            "interfaces": [
                {
                    "id": interface_id,
                    "hyperedge_id": interface_id,
                    "between": [
                        port_ids[participant]
                        for participant in seed.participants
                    ],
                    "contact_pairs": [
                        [port_ids[left], port_ids[right]]
                        for left, right in contact_tree
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
                }
            ],
            "connections": connections,
            "sampling": {
                "timesteps": timesteps,
                "seed": seed_start + len(candidates),
                "preset": "exact_mosaic",
                "low_memory_mode": True,
                "execution_backend": "explicit_all_copy",
            },
            "resources": intent.resources.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "preferences": intent.preferences.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "assembly_shape": _assembly_shape_payload(intent),
        }
        if intent.output is not None:
            payload["output"] = intent.output.model_dump(
                mode="json",
                exclude_none=True,
            )
        design = UserDesignSpec.model_validate(payload)
        # Compile now, not after ranking, so READY always means this exact
        # participant path and atomic hyperedge are executable together.
        lower_user_design(design)
        first_link = recorded_links[0]
        candidates.append(
            SimpleDesignCandidate(
                candidate_id=f"candidate_{len(candidates):06d}",
                symmetry=symmetry_id,
                topology_id=(
                    f"{symmetry_id.lower()}__single_atomic_hyperedge__"
                    "explicit_participant_paths"
                ),
                design=design,
                connection_order=(first_link[0], first_link[1]),
                connection_orbit_offset=0,
                resolution_frontend=(
                    "single_supplied_hyperedge_explicit_paths_v1"
                ),
                polymer_links=tuple(recorded_links),
                polymer_units_per_copy=len(seed.participants),
                physical_polymer_unit_count=(
                    len(seed.participants) * physical_copy_count
                ),
                expanded_topology_status=(
                    "explicit_intra_participant_paths"
                ),
                interface_hyperedges=((interface_id, member_ids),),
                stabilizer_evidence=(
                    finite_action.to_dict()
                    if finite_action is not None
                    else None
                ),
                supplied_interface_ids=(interface_id,),
            )
        )
    if not candidates:
        raise NotImplementedError(
            "No executable Cn symmetry remains for the supplied hyperedge"
        )
    return tuple(candidates)


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


def _ordered_fragment_selectors(selector: str) -> tuple[str, ...]:
    """Expand one multi-helix participant into its polymer-ordered ranges."""

    segments = tuple(
        sorted(
            parse_public_selector(selector),
            key=lambda item: (
                item.chain_id,
                item.residue_start,
                item.residue_end,
            ),
        )
    )
    chains = {segment.chain_id for segment in segments}
    if len(chains) != 1:
        raise NotImplementedError(
            "Ordinary multi-fragment participants must come from one source "
            "polymer chain"
        )
    return tuple(segment.assembly_expression for segment in segments)


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
            f"multi-seed interface/unit contract: {rejected}"
        )
    if not accepted:
        raise ValueError(
            "No requested symmetry satisfies the multi-seed "
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
    *,
    fixed_relations: tuple[int | str | None, ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Assign only the missing relations needed to connect every copy.

    An ordinary user may declare just the polymer chemistry, in which case
    the historical canonical relation family is retained.  A knowledgeable
    user may additionally freeze any subset of copy relations.  Mosaic then
    proves whether those relations already generate the requested finite
    group and, if necessary, enumerates the smallest deterministic completion
    on the still-unassigned links.  It never changes a declared relation.
    """

    if link_count < 1:
        raise ValueError("At least one polymer link is required")
    if fixed_relations is not None and len(fixed_relations) != link_count:
        raise ValueError(
            "fixed polymer copy relations must match the link count"
        )
    if fixed_relations is not None and any(
        relation is not None for relation in fixed_relations
    ):
        return _complete_declared_connection_relations(
            symmetry_id,
            fixed_relations,
        )
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


def _relation_transform_id(
    symmetry_id: str,
    relation: int | str,
) -> str:
    """Resolve either public relation spelling to one registry transform."""

    registry = build_transform_registry(finite_symmetry_spec(symmetry_id))
    if isinstance(relation, int):
        return registry.transform_id_for_offset(relation)
    # Resolve through the registry here so an unknown expert relation fails
    # before candidate materialization rather than much later in compilation.
    registry.transform(relation)
    return relation


def _complete_declared_connection_relations(
    symmetry_id: str,
    fixed_relations: tuple[int | str | None, ...],
) -> tuple[dict[str, Any], ...]:
    """Find every smallest group-generating completion of fixed links."""

    registry = build_transform_registry(finite_symmetry_spec(symmetry_id))
    identity = registry.identity_id
    fixed_transform_ids = tuple(
        _relation_transform_id(symmetry_id, relation)
        for relation in fixed_relations
        if relation is not None
    )
    open_indices = tuple(
        index
        for index, relation in enumerate(fixed_relations)
        if relation is None
    )

    def generates(relations: tuple[str, ...]) -> bool:
        active = tuple(item for item in relations if item != identity)
        return len(generated_transform_ids(symmetry_id, active)) == len(
            registry.transform_ids
        )

    completions: list[tuple[str, ...]] = []
    if generates(fixed_transform_ids):
        completions.append(())
    else:
        candidates = tuple(
            transform_id
            for transform_id in registry.transform_ids
            if transform_id != identity
            and transform_id not in fixed_transform_ids
        )
        for size in range(1, len(open_indices) + 1):
            for completion in combinations(candidates, size):
                if generates((*fixed_transform_ids, *completion)):
                    completions.append(completion)
            if completions:
                break
    if not completions:
        rendered = [
            relation if relation is not None else "auto"
            for relation in fixed_relations
        ]
        raise ValueError(
            f"Declared polymer copy relations {rendered} cannot generate "
            f"the complete {symmetry_id} group with "
            f"{len(open_indices)} unassigned links"
        )

    plans: list[dict[str, Any]] = []
    seen: set[tuple[int | str, ...]] = set()
    for completion in completions:
        position_orders = (
            permutations(open_indices, len(completion))
            if completion
            else ((),)
        )
        for positions in position_orders:
            relations: list[int | str] = [
                relation if relation is not None else 0
                for relation in fixed_relations
            ]
            for position, transform_id in zip(
                positions,
                completion,
                strict=True,
            ):
                relations[position] = transform_id
            relation_tuple = tuple(relations)
            if relation_tuple in seen:
                continue
            seen.add(relation_tuple)
            transform_ids = tuple(
                _relation_transform_id(symmetry_id, relation)
                for relation in relation_tuple
            )
            if not generates(transform_ids):
                continue
            nonidentity_indices = tuple(
                index
                for index, transform_id in enumerate(transform_ids)
                if transform_id != identity
            )
            primary_index = nonidentity_indices[0]
            assignment = "__".join(
                f"link_{index:02d}_{transform_id.replace(':', '_')}"
                for index, transform_id in enumerate(transform_ids)
                if transform_id != identity
            )
            plans.append(
                {
                    "label": f"declared_completion__{assignment}",
                    "relations": relation_tuple,
                    "primary_index": primary_index,
                    "primary_relation": relation_tuple[primary_index],
                    "declared_relation_count": sum(
                        relation is not None for relation in fixed_relations
                    ),
                    "completed_relation_count": len(completion),
                }
            )
    if not plans:
        raise RuntimeError(
            "Finite-group relation completion produced no connected plan"
        )
    return tuple(plans)


def _source_chain_ownership_failures(
    directed_links: tuple[tuple[str, str], ...],
    side_records: dict[str, tuple[str, str, str]],
) -> tuple[str, ...]:
    """Reject ordinary candidates that split one known source polymer.

    A source chain reused by several declared interface seeds is evidence
    that those seed faces belong to one physical component.  All of its
    supplied faces must therefore belong to one connected scaffold path.
    Treating those faces as unrelated components can place a generated
    terminus directly into the occupied continuation of the same source
    backbone (the failure observed for the 7mwr two-patch canary).

    Expert assembly graphs remain free to request a different chain
    topology explicitly.  Ordinary resolution must not silently discard the
    component identity already present in the input file.
    """

    sides_by_chain: dict[str, list[str]] = {}
    for side_id, (_, _, selector) in side_records.items():
        chain_ids = {
            segment.chain_id for segment in parse_public_selector(selector)
        }
        if len(chain_ids) != 1:
            raise ValueError(
                "Ordinary source-chain ownership requires each supplied "
                "participant to belong to one polymer chain"
            )
        sides_by_chain.setdefault(next(iter(chain_ids)), []).append(side_id)

    adjacency: dict[str, set[str]] = {
        side_id: set() for side_id in side_records
    }
    for source_side, target_side in directed_links:
        adjacency[source_side].add(target_side)
        adjacency[target_side].add(source_side)

    component_by_side: dict[str, int] = {}
    component_index = 0
    for root in sorted(adjacency):
        if root in component_by_side:
            continue
        pending = [root]
        component_by_side[root] = component_index
        while pending:
            current = pending.pop()
            for neighbour in sorted(adjacency[current]):
                if neighbour in component_by_side:
                    continue
                component_by_side[neighbour] = component_index
                pending.append(neighbour)
        component_index += 1

    failures: list[str] = []
    for chain_id, side_ids in sorted(sides_by_chain.items()):
        if len(side_ids) < 2:
            continue
        path_components = {
            component_by_side[side_id] for side_id in side_ids
        }
        if len(path_components) != 1:
            failures.append(
                f"source chain {chain_id!r} is split across different "
                "polymer units instead of preserving all supplied "
                f"interface faces as one component: {sorted(side_ids)}"
            )
    return tuple(failures)


def _declared_polymer_path_cover(
    intent: SimpleCageIntentSpec,
    side_records: dict[str, tuple[str, str, str]],
) -> PolymerUnitPathCoverHypothesis:
    """Bind the user's authoritative polymer graph without enumerating it."""

    side_by_endpoint = {
        (interface_id, participant): side_id
        for side_id, (interface_id, participant, _) in side_records.items()
    }
    links: list[DirectedPolymerLink] = []
    incoming: dict[str, str] = {}
    outgoing: dict[str, str] = {}
    adjacency = {side_id: set() for side_id in side_records}
    for connection in intent.polymer_connections:
        source_key = (
            connection.from_endpoint.interface,
            connection.from_endpoint.participant,
        )
        target_key = (
            connection.to_endpoint.interface,
            connection.to_endpoint.participant,
        )
        source_side = side_by_endpoint[source_key]
        target_side = side_by_endpoint[target_key]
        outgoing[source_side] = target_side
        incoming[target_side] = source_side
        adjacency[source_side].add(target_side)
        adjacency[target_side].add(source_side)
        links.append(DirectedPolymerLink(source_side, target_side))

    visited: set[str] = set()
    ordered_paths: list[tuple[str, ...]] = []
    for root in sorted(adjacency):
        if root in visited:
            continue
        component: set[str] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        starts = sorted(component - set(incoming))
        ends = sorted(component - set(outgoing))
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                "user-declared polymer connections must form directed "
                "linear protein units; cyclic or branched covalent paths "
                f"are unsupported: {sorted(component)}"
            )
        path: list[str] = []
        current = starts[0]
        while True:
            if current in path:
                raise ValueError(
                    "user-declared polymer connections contain a cycle"
                )
            path.append(current)
            if current not in outgoing:
                break
            current = outgoing[current]
        if set(path) != component:
            raise ValueError(
                "user-declared polymer connections do not form one "
                f"continuous path for {sorted(component)}"
            )
        interface_ids = [side_records[side_id][0] for side_id in path]
        if len(interface_ids) != len(set(interface_ids)):
            raise ValueError(
                "one polymer unit cannot contain two participants of the "
                "same supplied interface occurrence; declare separate "
                "interface occurrences when reusing an interface type"
            )
        visited.update(component)
        ordered_paths.append(tuple(path))
    if visited != set(side_records):
        raise ValueError(
            "user-declared polymer connections leave supplied interface "
            f"participants unassigned: {sorted(set(side_records) - visited)}"
        )
    canonical_paths = tuple(sorted(ordered_paths))
    return PolymerUnitPathCoverHypothesis(
        canonical_key=canonical_paths,
        ordered_paths=canonical_paths,
        ordered_links=tuple(links),
        unit_count=len(canonical_paths),
        evidence_scope="user_declared_polymer_paths",
        search_complete=True,
    )


def _enumerate_multi_seed_candidates(
    intent: SimpleCageIntentSpec,
    *,
    symmetry_ids: Iterable[str] | None,
    seed_start: int,
    timesteps: int,
    max_candidates: int,
    global_placement: bool = False,
    pose_samples: int | None = None,
) -> tuple[SimpleDesignCandidate, ...]:
    """Freeze supplied interface hyperedges into finite-group hypotheses.

    Each supplied seed is an authoritative rigid interface hyperedge.  The
    relative pose *between* seeds is either preserved from a shared input or
    replaced by deterministic global starts before continuous optimization,
    according to the ordinary intent's ``seed_layout`` policy. The topology
    enumerator chooses which seed sides form polymer units but is forbidden
    from creating any new interface identity. For Cn, one cycle holonomy
    carries the familiar +/-1 winding. For
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
    declared_connections = bool(intent.polymer_connections)
    declared_connection_by_link: dict[
        tuple[str, str], tuple[int | str | None, str | None]
    ] = {}
    if declared_connections:
        side_by_endpoint = {
            (interface_id, participant): side_id
            for side_id, (interface_id, participant, _) in side_records.items()
        }
        for connection in intent.polymer_connections:
            source_side = side_by_endpoint[
                (
                    connection.from_endpoint.interface,
                    connection.from_endpoint.participant,
                )
            ]
            target_side = side_by_endpoint[
                (
                    connection.to_endpoint.interface,
                    connection.to_endpoint.participant,
                )
            ]
            copy_relation = connection.copy_relation
            resolved_relation: int | str | None = None
            if copy_relation is not None:
                resolved_relation = (
                    copy_relation.orbit_offset
                    if copy_relation.orbit_offset is not None
                    else copy_relation.transform
                )
            declared_connection_by_link[(source_side, target_side)] = (
                resolved_relation,
                connection.id,
            )
    # Three or more supplied interface identities can support protein units
    # with more than two faces.  The user supplies every interface geometry
    # and its multiplicity; Mosaic only enumerates how the supplied physical
    # sides can be distributed across polymer units.  It never invents an
    # interface not present in ``interface_seeds``.
    all_binary = all(len(seed.side_ids) == 2 for seed in topology_seeds)
    multi_face_frontend = (
        len(topology_seeds) >= 3
        and intent.goal.symmetry == "auto"
    )
    if declared_connections:
        path_covers = (
            _declared_polymer_path_cover(intent, side_records),
        )
        multi_face_frontend = any(
            len(path) > 2 for path in path_covers[0].ordered_paths
        )
        hyperedge_frontend = not all_binary
    elif multi_face_frontend and all_binary:
        multi_face_covers = enumerate_polymer_unit_path_covers(
            topology_seeds,
            minimum_faces_per_unit=3,
            maximum_faces_per_unit=min(8, len(topology_seeds)),
            require_equal_unit_sizes=False,
            max_candidates=max_candidates,
        )
        if len(topology_seeds) <= 4:
            # Retain the complete historical two-face cycle family where it
            # remains small, then add the higher-valency protein units.
            binary_seeds = tuple(
                BinaryInterfaceSeed(
                    seed_id=seed.seed_id,
                    left_side_id=seed.side_ids[0],
                    right_side_id=seed.side_ids[1],
                )
                for seed in topology_seeds
            )
            path_covers = (
                *enumerate_directed_polymer_path_covers(
                    binary_seeds,
                    max_candidates=max_candidates,
                ),
                *multi_face_covers,
            )
        else:
            # Exhaustive two-face Hamiltonian cycles grow factorially.  For
            # five or more supplied interface identities, spend the bounded
            # ordinary-user search budget on the requested higher-valency
            # cage units instead of pretending to enumerate every cycle.
            path_covers = multi_face_covers
        hyperedge_frontend = False
    elif multi_face_frontend:
        path_covers = enumerate_polymer_unit_path_covers(
            topology_seeds,
            minimum_faces_per_unit=2,
            maximum_faces_per_unit=min(8, len(topology_seeds)),
            require_equal_unit_sizes=False,
            max_candidates=max_candidates,
        )
        hyperedge_frontend = True
    elif all_binary:
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

    total_side_count = sum(len(seed.side_ids) for seed in topology_seeds)
    cover_unit_counts = {
        id(hypothesis): (
            getattr(hypothesis, "unit_count", total_side_count // 2)
        )
        for hypothesis in path_covers
    }
    symmetry_by_unit_count: dict[int, tuple[str, ...]] = {}
    for polymer_unit_count in sorted(set(cover_unit_counts.values())):
        topological_cycle_rank = (
            total_side_count
            - len(topology_seeds)
            - polymer_unit_count
            + 1
        )
        symmetry_by_unit_count[polymer_unit_count] = (
            _multi_seed_symmetry_order(
                intent,
                symmetry_ids,
                polymer_units_per_copy=polymer_unit_count,
                topological_cycle_rank=topological_cycle_rank,
            )
        )
    symmetry_order = tuple(
        dict.fromkeys(
            symmetry_id
            for polymer_unit_count in sorted(symmetry_by_unit_count)
            for symmetry_id in symmetry_by_unit_count[polymer_unit_count]
        )
    )
    total_candidate_count = 0
    direction_count = 1 if declared_connections else 2
    for hypothesis in path_covers:
        polymer_unit_count = cover_unit_counts[id(hypothesis)]
        link_count = len(hypothesis.ordered_links)
        for symmetry_id in symmetry_by_unit_count[polymer_unit_count]:
            total_candidate_count += (
                direction_count
                * len(
                    _connection_relation_plans(
                        symmetry_id,
                        link_count,
                    )
                )
            )
    topology_budget_truncated = False
    if total_candidate_count > max_candidates and multi_face_frontend:
        selected_covers = []
        selected_candidate_count = 0
        for hypothesis in path_covers:
            polymer_unit_count = cover_unit_counts[id(hypothesis)]
            link_count = len(hypothesis.ordered_links)
            hypothesis_cost = sum(
                direction_count * len(
                    _connection_relation_plans(symmetry_id, link_count)
                )
                for symmetry_id in symmetry_by_unit_count[
                    polymer_unit_count
                ]
            )
            if selected_candidate_count + hypothesis_cost > max_candidates:
                continue
            selected_covers.append(hypothesis)
            selected_candidate_count += hypothesis_cost
        if not selected_covers:
            raise ValueError(
                "One multi-face topology hypothesis requires more than "
                f"max_candidates={max_candidates}; increase the resolver "
                "candidate budget or narrow the symmetry choices"
            )
        path_covers = tuple(selected_covers)
        total_candidate_count = selected_candidate_count
        topology_budget_truncated = True
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
    participant_fragments: dict[
        tuple[str, str], tuple[str, ...]
    ] = {}
    internal_connection_payloads: list[dict[str, Any]] = []
    for interface_id, seed in seeds:
        component_id = f"seed__{interface_id}"
        component_selectors: list[str] = []
        for participant in seed.participants:
            fragments = _ordered_fragment_selectors(
                seed.selectors[participant]
            )
            participant_fragments[(interface_id, participant)] = fragments
            component_selectors.extend(fragments)
        components[component_id] = {
            "selectors": component_selectors,
            "geometry": "joint_rigid",
            "pose": _ordinary_component_pose(intent),
        }
        interface_ports: list[str] = []
        for side_index, participant in enumerate(seed.participants):
            port_id = f"port__{interface_id}__{side_index + 1:02d}"
            interface_ports.append(port_id)
            fragments = participant_fragments[(interface_id, participant)]
            ports[port_id] = {
                "component": component_id,
                "selectors": list(fragments),
            }
            for fragment_index, (left, right) in enumerate(
                zip(fragments, fragments[1:]),
                start=1,
            ):
                internal_connection_payloads.append(
                    {
                        "id": (
                            f"internal__{interface_id}__side_"
                            f"{side_index + 1:02d}__link_"
                            f"{fragment_index:02d}"
                        ),
                        "from": {
                            "component": component_id,
                            "selector": left,
                            "terminus": "c",
                        },
                        "to": {
                            "component": component_id,
                            "selector": right,
                            "terminus": "n",
                        },
                        "length": intent.generation.length,
                        "copy_relation": {"orbit_offset": 0},
                    }
                )
        participant_ports = dict(
            zip(seed.participants, interface_ports, strict=True)
        )
        evidence = _validated_input_evidence(intent, interface_id, seed)
        tree = _contact_spanning_tree(
            seed.participants,
            evidence["active_contact_pairs"],
        )
        member_ids = [
            (
                interface_id
                if len(tree) == 1
                else f"{interface_id}__member_{pair_index:02d}"
            )
            for pair_index in range(1, len(tree) + 1)
        ]
        interface_payload: dict[str, Any] = {
            "id": interface_id,
            "hyperedge_id": interface_id,
            "between": interface_ports,
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
        }
        if len(tree) > 1:
            interface_payload["contact_pairs"] = [
                [participant_ports[left], participant_ports[right]]
                for left, right in tree
            ]
        interfaces.append(interface_payload)
        hyperedge_members[interface_id] = member_ids

    supplied_interface_ids = tuple(sorted(hyperedge_members))
    emitted_hyperedge_ids = {
        str(interface["hyperedge_id"])
        for interface in interfaces
    }
    if emitted_hyperedge_ids != set(supplied_interface_ids):
        raise RuntimeError(
            "Ordinary resolver interface-identity invariant failed: emitted "
            f"{sorted(emitted_hyperedge_ids)}, supplied "
            f"{list(supplied_interface_ids)}. Mosaic will not invent, omit "
            "or merge user-supplied interface seeds"
        )

    candidates: list[SimpleDesignCandidate] = []
    for symmetry_id in symmetry_order:
        order = symmetry_group_action_count(symmetry_id)
        for path_index, hypothesis in enumerate(path_covers):
            polymer_units_per_copy = cover_unit_counts[id(hypothesis)]
            if (
                symmetry_id
                not in symmetry_by_unit_count[polymer_units_per_copy]
            ):
                continue
            canonical_links = tuple(
                (link.source_side_id, link.target_side_id)
                for link in hypothesis.ordered_links
            )
            relation_plans = _connection_relation_plans(
                symmetry_id,
                len(canonical_links),
                fixed_relations=(
                    tuple(
                        declared_connection_by_link[link][0]
                        for link in canonical_links
                    )
                    if declared_connections
                    else None
                ),
            )
            directions = (
                (("declared", canonical_links),)
                if declared_connections
                else (
                    ("forward", canonical_links),
                    ("reverse", _reverse_polymer_links(canonical_links)),
                )
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
                        connection_payloads: list[dict[str, Any]] = [
                            dict(connection)
                            for connection in internal_connection_payloads
                        ]
                        recorded_links: list[
                            tuple[str, str, int | str]
                        ] = []
                        for link_index, (
                            source_side,
                            target_side,
                        ) in enumerate(directed_links):
                            source_interface, source_participant, _ = side_records[
                                source_side
                            ]
                            target_interface, target_participant, _ = side_records[
                                target_side
                            ]
                            source_selector = participant_fragments[
                                (source_interface, source_participant)
                            ][-1]
                            target_selector = participant_fragments[
                                (target_interface, target_participant)
                            ][0]
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
                            declared_connection_id = (
                                declared_connection_by_link[
                                    (source_side, target_side)
                                ][1]
                                if declared_connections
                                else None
                            )
                            connection_payloads.append(
                                {
                                    "id": (
                                        declared_connection_id
                                        or f"polymer_link_"
                                        f"{link_index + 1:03d}"
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
                            "task": "preserve_supplied_geometry",
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
                            "preferences": intent.preferences.model_dump(
                                mode="json",
                                exclude_none=True,
                            ),
                            "assembly_shape": _assembly_shape_payload(intent),
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
                            if declared_connections
                            or hyperedge_frontend
                            or multi_face_frontend
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
                                    "user_declared_polymer_paths_v1"
                                    if declared_connections
                                    else "multi_face_polymer_unit_v1"
                                    if multi_face_frontend
                                    else (
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
                                topology_search_complete=(
                                    not topology_budget_truncated
                                    and getattr(
                                        hypothesis,
                                        "search_complete",
                                        True,
                                    )
                                ),
                                supplied_interface_ids=(
                                    supplied_interface_ids
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
    if len(intent.interface_seeds) == 1:
        _, only_seed = next(iter(intent.interface_seeds.items()))
        if _requires_mixed_component_incidence(only_seed):
            if global_placement:
                raise NotImplementedError(
                    "Mixed oligomer-component incidence currently requires "
                    "one supplied coordinate frame; independent component "
                    "pose optimization must first satisfy both stabilizers "
                    "and the supplied natural interface"
                )
            return _enumerate_mixed_component_interface_candidates(
                intent,
                symmetry_ids=symmetry_ids,
                seed_start=seed_start,
                timesteps=timesteps,
                max_candidates=max_candidates,
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
        return _enumerate_single_supplied_hyperedge_candidates(
            intent,
            interface_id=interface_id,
            seed=seed,
            symmetry_ids=symmetry_ids,
            seed_start=seed_start,
            timesteps=timesteps,
            max_candidates=max_candidates,
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
                    "task": "preserve_supplied_geometry",
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
                            "pose": _ordinary_component_pose(intent),
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
                    "preferences": intent.preferences.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "assembly_shape": _assembly_shape_payload(intent),
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
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resolve, rank and freeze ordinary candidates through the expert path."""

    if pose_samples is not None and pose_samples < 1:
        raise ValueError("pose_samples must be positive")
    emit = progress if progress is not None else (lambda _message: None)
    output = Path(output_directory).expanduser().resolve()
    emit("materializing supplied interface seeds")
    materialized = materialize_seed_library(
        intent,
        output / "_inputs",
    )
    working_intent = materialized.intent
    if (
        pose_samples not in (None, 1)
        and not materialized.independent_frames
    ):
        raise NotImplementedError(
            "The first ordinary resolver keeps the supplied seed pose. "
            "Multiple global pose starts apply only to independently "
            "supplied seed files"
        )
    diversity_pose_samples = {
        "low": 4,
        "medium": 8,
        "high": 16,
    }[working_intent.preferences.diversity.value]
    effective_pose_samples = (
        pose_samples
        if materialized.independent_frames and pose_samples is not None
        else diversity_pose_samples
        if materialized.independent_frames
        else 1
    )
    seed_records = _supported_binary_seeds(working_intent)
    emit(
        "validating "
        f"{len(seed_records)} supplied interface geometries and termini"
    )
    evidence_records = [
        _validated_input_evidence(working_intent, interface_id, seed)
        for interface_id, seed in seed_records
    ]
    if len(seed_records) > 1:
        _validate_multi_seed_backbone_anchors(working_intent)
    emit(
        "enumerating finite-group topology and pose starts "
        f"({effective_pose_samples} pose samples)"
    )
    candidates = enumerate_simple_design_candidates(
        working_intent,
        symmetry_ids=symmetry_ids,
        seed_start=seed_start,
        timesteps=timesteps,
        max_candidates=max_candidates,
        global_placement=materialized.independent_frames,
        pose_samples=effective_pose_samples,
    )
    emit(f"enumerated {len(candidates)} candidate states")
    candidate_payloads = tuple(
        (candidate.candidate_id, candidate.design, candidate.metadata())
        for candidate in candidates
    )
    pose_optimization_applied = (
        (optimize_poses or materialized.independent_frames)
        and len(seed_records) > 1
    )
    if pose_optimization_applied:
        emit(
            "optimizing complete-seed SE(3) poses for the best "
            f"{min(pose_optimize_top, len(candidate_payloads))} candidates"
        )
        candidate_payloads = optimize_candidate_subset(
            candidate_payloads,
            top_count=pose_optimize_top,
            levels=pose_optimization_levels,
            maximum_translation=pose_maximum_translation,
            maximum_rotation_deg=pose_maximum_rotation_deg,
        )
        emit("continuous pose optimization finished")
    emit(
        "compiling, ranking and strictly replaying "
        f"{len(candidate_payloads)} candidates"
    )
    ranked = rank_design_candidates(
        candidate_payloads,
        output_directory,
        top_count=top_count,
    )
    emit(
        "strict replay finished: "
        f"accepted={ranked['accepted_count']} "
        f"selected={ranked['selected_count']}"
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
    mixed_component_incidence = (
        len(seed_records) == 1
        and _requires_mixed_component_incidence(seed_records[0][1])
    )
    hyperedge_seed = any(
        len(seed.participants) > 2 for _, seed in seed_records
    )
    manifest = {
        "schema_version": 1,
        "resolver": (
            "rfd3_mosaic.supplied_oligomer_interface_incidence_v1"
            if mixed_component_incidence
            else "rfd3_mosaic.user_declared_polymer_global_pose_v1"
            if multi_seed
            and materialized.independent_frames
            and working_intent.polymer_connections
            else "rfd3_mosaic.user_declared_polymer_replay_v1"
            if multi_seed and working_intent.polymer_connections
            else "rfd3_mosaic.independent_multi_interface_finite_group_v1"
            if multi_seed
            and materialized.independent_frames
            and hyperedge_seed
            else "rfd3_mosaic.independent_multi_seed_global_cn_v1"
            if multi_seed and materialized.independent_frames
            else "rfd3_mosaic.prepositioned_multi_interface_finite_group_v1"
            if multi_seed and hyperedge_seed
            else "rfd3_mosaic.prepositioned_multi_binary_cn_v1"
            if multi_seed
            else "rfd3_mosaic.single_supplied_hyperedge_explicit_paths_v1"
            if hyperedge_seed
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
        "seed_layout": {
            "requested": working_intent.seed_layout,
            "relative_pose": (
                "solve"
                if materialized.independent_frames
                else "preserve_input"
            ),
            "input_file_frames_ignored": (
                materialized.independent_frames
            ),
        },
        "input_evidence": (
            evidence_records
            if multi_seed
            else evidence_records[0]
        ),
        "authoring_mode": "ordinary",
        "execution_path": (
            "UserDesignSpec -> AssemblySpecification -> Mosaic-RFD3"
        ),
        # A CPU score is a recommendation for inspection, never authority to
        # execute a topology or protein-unit arrangement on the user's
        # behalf.  This remains false even when only one replayable YAML is
        # available.
        "automatic_selection": False,
        "cpu_recommendation_available": recommended is not None,
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
        "resolved_component_incidence": (
            {
                "physical_interface_count": recommended.get(
                    "physical_interface_count"
                ),
                "component_orbit_multiplicities": recommended.get(
                    "component_orbit_multiplicities", {}
                ),
                "stabilizer_evidence": recommended.get(
                    "stabilizer_evidence"
                ),
            }
            if mixed_component_incidence and recommended is not None
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
                "all_required_input_interfaces_satisfied",
                "all_required_static_objectives_satisfied",
                "strict_frozen_replay",
            ],
            "soft_ranking_preferences": [
                "clear_straight_chord_linker_corridors",
                "favourable_terminal_tangent_geometry",
                "shorter_linker_endpoint_distances",
                "larger_inter_group_clearance",
            ],
        },
        "selection_required": ranked["selected_count"] > 0,
        "supported_contract": (
            "one user-supplied preserve_exact oligomer--oligomer interface; "
            "participant valencies inferred from source protomer chains; "
            "finite-group stabilizer/coset component orbits; exact physical "
            "interface multiplicity; no invented interface identity"
            if mixed_component_incidence
            else "several user-supplied preserve_exact interface seeds; canonical "
            "local frames; deterministic global finite-group pose starts; "
            "bounded joint SE(3) refinement; polymer path-cover; no invented "
            "interface identities"
            if multi_seed and materialized.independent_frames
            else "several disjoint pre-positioned binary preserve_exact "
            "seeds; bounded path-cover; full-orbit Cn winding"
            if multi_seed
            else "one cooperative preserve_exact hyperedge; explicit "
            "same-chain participant paths; no invented cross-participant "
            "covalent links"
            if hyperedge_seed
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
