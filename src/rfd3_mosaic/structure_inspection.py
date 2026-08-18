"""Input-driven structure inspection for the ordinary-user cage workflow."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from rfd3_mosaic.structure import AtomRecord, read_structure_atoms

_ASSEMBLY_SELECTOR = re.compile(
    r"^(?P<chain>[^/,]+)/(?P<start>-?[0-9]+)"
    r"(?:-(?P<end>-?[0-9]+))?/(?P<atoms>\*|all|backbone|CA)$",
    re.IGNORECASE,
)
_COMPACT_SELECTOR = re.compile(
    r"^(?P<chain>[^0-9,/\s]+)(?P<start>-?[0-9]+)"
    r"(?:-(?P<end>-?[0-9]+))?$"
)
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True)
class ChainInspection:
    chain_id: str
    residue_count: int
    atom_count: int
    residue_ranges: tuple[tuple[int, int], ...]
    selector: str


@dataclass(frozen=True)
class InterfaceCandidate:
    interface_id: str
    left_chain: str
    right_chain: str
    left_selector: str
    right_selector: str
    heavy_atom_contact_count: int
    left_contact_residue_count: int
    right_contact_residue_count: int
    minimum_heavy_atom_distance: float
    cutoff: float
    contact_patch_index: int
    contact_patch_count: int


@dataclass(frozen=True)
class ComponentInterfaceSummary:
    """Detected non-covalent faces carried by one input chain.

    A chain is only an observed input component at this stage.  The ordinary
    architecture resolver may later join several input fragments into one
    polymer unit, but inspection must still expose the measured incidence
    graph without guessing that topology.
    """

    chain_id: str
    interface_ids: tuple[str, ...]

    @property
    def detected_port_count(self) -> int:
        return len(self.interface_ids)


@dataclass(frozen=True)
class StructureInspection:
    source: str
    chains: tuple[ChainInspection, ...]
    interface_candidates: tuple[InterfaceCandidate, ...]
    component_interface_sets: tuple[ComponentInterfaceSummary, ...]
    contact_cutoff: float
    minimum_atom_contacts: int
    minimum_contact_residues_per_side: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": self.source,
            "chain_count": len(self.chains),
            "chains": [asdict(item) for item in self.chains],
            "interface_candidate_count": len(self.interface_candidates),
            "interface_candidates": [
                asdict(item) for item in self.interface_candidates
            ],
            "component_interface_sets": [
                {
                    **asdict(item),
                    "detected_port_count": item.detected_port_count,
                }
                for item in self.component_interface_sets
            ],
            "parameters": {
                "contact_cutoff": self.contact_cutoff,
                "minimum_atom_contacts": self.minimum_atom_contacts,
                "minimum_contact_residues_per_side": (
                    self.minimum_contact_residues_per_side
                ),
            },
        }


@dataclass(frozen=True)
class DeclaredInterfaceEvidence:
    """Contact-graph evidence for a pairwise or multi-participant seed."""

    interface_id: str
    participant_chains: tuple[str, ...]
    pair_evidence: tuple[InterfaceCandidate, ...]
    active_contact_pairs: tuple[tuple[str, str], ...]
    contact_graph_connected: bool


def _is_heavy(atom: AtomRecord) -> bool:
    element = atom.element.strip().upper()
    if element:
        return element != "H"
    name = atom.atom_name.lstrip("0123456789").upper()
    return not name.startswith("H")


def _ranges(values: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(values))
    if not ordered:
        return ()
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous))
        start = previous = value
    result.append((start, previous))
    return tuple(result)


def _selector(chain_id: str, residue_numbers: Iterable[int]) -> str:
    return ",".join(
        f"{chain_id}/{start}-{end}/*"
        for start, end in _ranges(residue_numbers)
    )


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "chain"


def _resolve_selector_atoms(
    atoms: tuple[AtomRecord, ...],
    expression: str,
    *,
    expected_chain: str,
    allow_multiple_chains: bool = False,
) -> tuple[AtomRecord, ...]:
    selected: dict[
        tuple[str, int, str, str, str],
        AtomRecord,
    ] = {}
    for term in expression.split(","):
        match = _ASSEMBLY_SELECTOR.fullmatch(term.strip())
        compact = False
        if match is None:
            match = _COMPACT_SELECTOR.fullmatch(term.strip())
            compact = match is not None
        if match is None:
            raise ValueError(
                f"Invalid ordinary interface selector {term!r}; expected "
                "CHAIN/start-end/* or CHAINstart-end"
            )
        chain_id = match.group("chain")
        if not allow_multiple_chains and chain_id != expected_chain:
            raise ValueError(
                f"Selector {term!r} belongs to chain {chain_id!r}, not "
                f"declared chain {expected_chain!r}"
            )
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if start > end:
            raise ValueError(f"Selector {term!r} has a reversed range")
        atom_scope = "all" if compact else match.group("atoms").lower()
        for atom in atoms:
            if atom.chain_id != chain_id:
                continue
            if not start <= atom.residue_number <= end:
                continue
            if atom_scope == "ca" and atom.atom_name.upper() != "CA":
                continue
            if (
                atom_scope == "backbone"
                and atom.atom_name.upper() not in _BACKBONE_ATOMS
            ):
                continue
            key = (
                atom.chain_id,
                atom.residue_number,
                atom.insertion_code,
                atom.residue_name,
                atom.atom_name,
            )
            selected[key] = atom
    if not selected:
        raise ValueError(
            f"Selector {expression!r} matched no atoms in the input"
        )
    return tuple(selected.values())


def _pair_contacts(
    left: tuple[AtomRecord, ...],
    right: tuple[AtomRecord, ...],
    *,
    cutoff: float,
    chunk_size: int = 512,
) -> tuple[int, set[int], set[int], float]:
    left_coordinates = np.asarray(
        [atom.coordinate for atom in left],
        dtype=np.float64,
    )
    right_coordinates = np.asarray(
        [atom.coordinate for atom in right],
        dtype=np.float64,
    )
    cutoff_squared = cutoff * cutoff
    contact_count = 0
    left_residues: set[int] = set()
    right_residues: set[int] = set()
    minimum_squared = float("inf")
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        differences = (
            left_coordinates[start:stop, None, :]
            - right_coordinates[None, :, :]
        )
        squared = np.einsum("ijk,ijk->ij", differences, differences)
        minimum_squared = min(minimum_squared, float(np.min(squared)))
        rows, columns = np.nonzero(squared <= cutoff_squared)
        contact_count += int(len(rows))
        left_residues.update(
            left[start + int(row)].residue_number for row in rows
        )
        right_residues.update(
            right[int(column)].residue_number for column in columns
        )
    return (
        contact_count,
        left_residues,
        right_residues,
        float(np.sqrt(minimum_squared)),
    )


def _pair_contact_patches(
    left: tuple[AtomRecord, ...],
    right: tuple[AtomRecord, ...],
    *,
    cutoff: float,
    chunk_size: int = 512,
) -> tuple[tuple[set[int], set[int]], ...]:
    """Split one chain-pair contact map into residue-connected patches.

    Two contacts belong to the same patch when they share a contacting
    residue, transitively, on either side of the interface.  This prevents
    spatially separate faces on the same two chains from being merged into a
    fictitious single seed while remaining invariant to atom ordering.
    """

    left_coordinates = np.asarray(
        [atom.coordinate for atom in left],
        dtype=np.float64,
    )
    right_coordinates = np.asarray(
        [atom.coordinate for atom in right],
        dtype=np.float64,
    )
    cutoff_squared = cutoff * cutoff
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        differences = (
            left_coordinates[start:stop, None, :]
            - right_coordinates[None, :, :]
        )
        squared = np.einsum("ijk,ijk->ij", differences, differences)
        rows, columns = np.nonzero(squared <= cutoff_squared)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            left_node = ("left", left[start + row].residue_number)
            right_node = ("right", right[column].residue_number)
            adjacency.setdefault(left_node, set()).add(right_node)
            adjacency.setdefault(right_node, set()).add(left_node)

    components: list[tuple[set[int], set[int]]] = []
    unseen = set(adjacency)
    while unseen:
        start_node = min(unseen)
        stack = [start_node]
        nodes: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            stack.extend(adjacency.get(node, set()) - nodes)
        unseen -= nodes
        components.append(
            (
                {value for side, value in nodes if side == "left"},
                {value for side, value in nodes if side == "right"},
            )
        )
    return tuple(
        sorted(
            components,
            key=lambda item: (
                min(item[0]),
                min(item[1]),
                tuple(sorted(item[0])),
                tuple(sorted(item[1])),
            ),
        )
    )


def inspect_structure_interfaces(
    path: str | Path,
    *,
    contact_cutoff: float = 4.5,
    minimum_atom_contacts: int = 4,
    minimum_contact_residues_per_side: int = 2,
) -> StructureInspection:
    """Detect chain-level interface candidates without assigning topology."""

    if contact_cutoff <= 0.0:
        raise ValueError("contact_cutoff must be positive")
    if minimum_atom_contacts < 1:
        raise ValueError("minimum_atom_contacts must be positive")
    if minimum_contact_residues_per_side < 1:
        raise ValueError(
            "minimum_contact_residues_per_side must be positive"
        )
    source = Path(path).expanduser().resolve()
    atoms = read_structure_atoms(source)
    by_chain: dict[str, list[AtomRecord]] = {}
    for atom in atoms:
        by_chain.setdefault(atom.chain_id, []).append(atom)
    if len(by_chain) < 2:
        raise ValueError(
            "Interface inspection requires at least two input chains"
        )

    chains: list[ChainInspection] = []
    heavy_by_chain: dict[str, tuple[AtomRecord, ...]] = {}
    for chain_id, chain_atoms in sorted(by_chain.items()):
        residue_numbers = sorted(
            {atom.residue_number for atom in chain_atoms}
        )
        chains.append(
            ChainInspection(
                chain_id=chain_id,
                residue_count=len(residue_numbers),
                atom_count=len(chain_atoms),
                residue_ranges=_ranges(residue_numbers),
                selector=_selector(chain_id, residue_numbers),
            )
        )
        heavy_by_chain[chain_id] = tuple(
            atom for atom in chain_atoms if _is_heavy(atom)
        )

    candidates: list[InterfaceCandidate] = []
    chain_ids = sorted(heavy_by_chain)
    for left_index, left_chain in enumerate(chain_ids):
        for right_chain in chain_ids[left_index + 1 :]:
            left_atoms = heavy_by_chain[left_chain]
            right_atoms = heavy_by_chain[right_chain]
            if not left_atoms or not right_atoms:
                continue
            accepted_patches: list[
                tuple[int, set[int], set[int], float]
            ] = []
            for left_patch, right_patch in _pair_contact_patches(
                left_atoms,
                right_atoms,
                cutoff=contact_cutoff,
            ):
                patch_left_atoms = tuple(
                    atom
                    for atom in left_atoms
                    if atom.residue_number in left_patch
                )
                patch_right_atoms = tuple(
                    atom
                    for atom in right_atoms
                    if atom.residue_number in right_patch
                )
                contacts, left_residues, right_residues, minimum = (
                    _pair_contacts(
                        patch_left_atoms,
                        patch_right_atoms,
                        cutoff=contact_cutoff,
                    )
                )
                if contacts < minimum_atom_contacts:
                    continue
                if (
                    len(left_residues)
                    < minimum_contact_residues_per_side
                    or len(right_residues)
                    < minimum_contact_residues_per_side
                ):
                    continue
                accepted_patches.append(
                    (contacts, left_residues, right_residues, minimum)
                )
            patch_count = len(accepted_patches)
            base_id = (
                "interface_"
                f"{_safe_identifier(left_chain)}_"
                f"{_safe_identifier(right_chain)}"
            )
            for patch_index, (
                contacts,
                left_residues,
                right_residues,
                minimum,
            ) in enumerate(accepted_patches, start=1):
                interface_id = (
                    base_id
                    if patch_count == 1
                    else f"{base_id}_patch_{patch_index:03d}"
                )
                candidates.append(
                    InterfaceCandidate(
                        interface_id=interface_id,
                        left_chain=left_chain,
                        right_chain=right_chain,
                        left_selector=_selector(
                            left_chain,
                            left_residues,
                        ),
                        right_selector=_selector(
                            right_chain,
                            right_residues,
                        ),
                        heavy_atom_contact_count=contacts,
                        left_contact_residue_count=len(left_residues),
                        right_contact_residue_count=len(right_residues),
                        minimum_heavy_atom_distance=minimum,
                        cutoff=contact_cutoff,
                        contact_patch_index=patch_index,
                        contact_patch_count=patch_count,
                    )
                )
    candidates.sort(
        key=lambda item: (
            -item.heavy_atom_contact_count,
            -min(
                item.left_contact_residue_count,
                item.right_contact_residue_count,
            ),
            item.interface_id,
        )
    )
    incidence: dict[str, list[str]] = {
        chain_id: [] for chain_id in sorted(by_chain)
    }
    for candidate in candidates:
        incidence[candidate.left_chain].append(candidate.interface_id)
        incidence[candidate.right_chain].append(candidate.interface_id)
    return StructureInspection(
        source=str(source),
        chains=tuple(chains),
        interface_candidates=tuple(candidates),
        component_interface_sets=tuple(
            ComponentInterfaceSummary(
                chain_id=chain_id,
                interface_ids=tuple(sorted(interface_ids)),
            )
            for chain_id, interface_ids in incidence.items()
        ),
        contact_cutoff=contact_cutoff,
        minimum_atom_contacts=minimum_atom_contacts,
        minimum_contact_residues_per_side=(
            minimum_contact_residues_per_side
        ),
    )


def inspect_declared_interface_seed(
    path: str | Path,
    *,
    interface_id: str,
    left_chain: str,
    right_chain: str,
    left_selector: str,
    right_selector: str,
    contact_cutoff: float = 4.5,
    allow_multiple_chains: bool = False,
) -> InterfaceCandidate:
    """Bind and measure one user-declared interface seed selection."""

    source = Path(path).expanduser().resolve()
    atoms = read_structure_atoms(source)
    left_atoms = tuple(
        atom
        for atom in _resolve_selector_atoms(
            atoms,
            left_selector,
            expected_chain=left_chain,
            allow_multiple_chains=allow_multiple_chains,
        )
        if _is_heavy(atom)
    )
    right_atoms = tuple(
        atom
        for atom in _resolve_selector_atoms(
            atoms,
            right_selector,
            expected_chain=right_chain,
            allow_multiple_chains=allow_multiple_chains,
        )
        if _is_heavy(atom)
    )
    if not left_atoms or not right_atoms:
        raise ValueError(
            f"Interface {interface_id!r} has no selected heavy atoms"
        )
    contacts, left_residues, right_residues, minimum = _pair_contacts(
        left_atoms,
        right_atoms,
        cutoff=contact_cutoff,
    )
    return InterfaceCandidate(
        interface_id=interface_id,
        left_chain=left_chain,
        right_chain=right_chain,
        left_selector=left_selector,
        right_selector=right_selector,
        heavy_atom_contact_count=contacts,
        left_contact_residue_count=len(left_residues),
        right_contact_residue_count=len(right_residues),
        minimum_heavy_atom_distance=minimum,
        cutoff=contact_cutoff,
        contact_patch_index=1,
        contact_patch_count=1,
    )


def inspect_declared_interface_relation(
    path: str | Path,
    *,
    interface_id: str,
    participants: tuple[str, ...],
    selectors: dict[str, str],
    contact_cutoff: float = 4.5,
    minimum_atom_contacts: int = 4,
    minimum_contact_residues_per_side: int = 2,
) -> DeclaredInterfaceEvidence:
    """Validate a multi-participant interface through a connected contact graph."""

    if len(participants) < 2:
        raise ValueError("Declared interface requires at least two participants")
    if len(participants) != len(set(participants)):
        raise ValueError("Declared interface participants must be unique")
    if set(selectors) != set(participants):
        raise ValueError(
            "Declared interface selectors must exactly match participants"
        )
    pair_evidence: list[InterfaceCandidate] = []
    active_pairs: list[tuple[str, str]] = []
    adjacency = {participant: set() for participant in participants}
    for left_index, left_chain in enumerate(participants):
        for right_chain in participants[left_index + 1 :]:
            evidence = inspect_declared_interface_seed(
                path,
                interface_id=f"{interface_id}:{left_chain}:{right_chain}",
                left_chain=left_chain,
                right_chain=right_chain,
                left_selector=selectors[left_chain],
                right_selector=selectors[right_chain],
                contact_cutoff=contact_cutoff,
                allow_multiple_chains=True,
            )
            pair_evidence.append(evidence)
            if (
                evidence.heavy_atom_contact_count >= minimum_atom_contacts
                and evidence.left_contact_residue_count
                >= minimum_contact_residues_per_side
                and evidence.right_contact_residue_count
                >= minimum_contact_residues_per_side
            ):
                active_pairs.append((left_chain, right_chain))
                adjacency[left_chain].add(right_chain)
                adjacency[right_chain].add(left_chain)
    visited: set[str] = set()
    stack = [participants[0]]
    while stack:
        participant = stack.pop()
        if participant in visited:
            continue
        visited.add(participant)
        stack.extend(adjacency[participant] - visited)
    return DeclaredInterfaceEvidence(
        interface_id=interface_id,
        participant_chains=participants,
        pair_evidence=tuple(pair_evidence),
        active_contact_pairs=tuple(active_pairs),
        contact_graph_connected=len(visited) == len(participants),
    )


def simple_intent_payload(
    inspection: StructureInspection,
    *,
    name: str,
    architecture: str = "auto",
    composition: str = "auto",
    symmetries: tuple[str, ...] = (),
    generated_length_minimum: int = 40,
    generated_length_maximum: int = 100,
    subunit_minimum: int | None = None,
    subunit_maximum: int | None = None,
    diameter_minimum: float | None = None,
    diameter_maximum: float | None = None,
    cavity_diameter_minimum: float | None = None,
    cavity_diameter_maximum: float | None = None,
    profile: str = "p100",
) -> dict[str, Any]:
    if not inspection.interface_candidates:
        raise ValueError(
            "Cannot create a cage intent without an interface candidate"
        )
    if generated_length_minimum < 1:
        raise ValueError("generated length minimum must be positive")
    if generated_length_minimum > generated_length_maximum:
        raise ValueError(
            "generated length minimum cannot exceed maximum"
        )
    if (subunit_minimum is None) != (subunit_maximum is None):
        raise ValueError("subunit minimum and maximum must be set together")
    if (diameter_minimum is None) != (diameter_maximum is None):
        raise ValueError("diameter minimum and maximum must be set together")
    if (cavity_diameter_minimum is None) != (
        cavity_diameter_maximum is None
    ):
        raise ValueError(
            "cavity diameter minimum and maximum must be set together"
        )
    goal: dict[str, Any] = {
        "architecture": architecture,
        "composition": composition,
        "symmetry": list(symmetries) if symmetries else "auto",
    }
    if subunit_minimum is not None and subunit_maximum is not None:
        goal["subunits"] = {
            "minimum": subunit_minimum,
            "maximum": subunit_maximum,
        }
    if diameter_minimum is not None and diameter_maximum is not None:
        goal["diameter_angstrom"] = {
            "minimum": diameter_minimum,
            "maximum": diameter_maximum,
        }
    if (
        cavity_diameter_minimum is not None
        and cavity_diameter_maximum is not None
    ):
        goal["cavity_diameter_angstrom"] = {
            "minimum": cavity_diameter_minimum,
            "maximum": cavity_diameter_maximum,
        }
    payload = {
        "schema_version": 1,
        "kind": "simple_cage_intent",
        "name": name,
        "input": inspection.source,
        "goal": goal,
        "interface_seeds": {
            candidate.interface_id: {
                "participants": [
                    candidate.left_chain,
                    candidate.right_chain,
                ],
                "selectors": {
                    candidate.left_chain: candidate.left_selector,
                    candidate.right_chain: candidate.right_selector,
                },
                "use": "auto",
                "geometry": "preserve_exact",
            }
            for candidate in inspection.interface_candidates
        },
        "generation": {
            "length": {
                "minimum": generated_length_minimum,
                "maximum": generated_length_maximum,
            }
        },
        "inspection": {
            "contact_cutoff": inspection.contact_cutoff,
            "minimum_atom_contacts": inspection.minimum_atom_contacts,
            "minimum_contact_residues_per_side": (
                inspection.minimum_contact_residues_per_side
            ),
        },
        "resources": {"profile": profile},
    }
    # Validate the emitted public document while preserving its intentionally
    # compact spelling (for example ``use: auto`` rather than expanded fields).
    from rfd3_mosaic.schema.simple_intent import SimpleCageIntentSpec

    SimpleCageIntentSpec.model_validate(payload)
    return payload


def write_structure_inspection(
    inspection: StructureInspection,
    output_directory: str | Path,
    *,
    intent_name: str,
    architecture: str = "auto",
    composition: str = "auto",
    symmetries: tuple[str, ...] = (),
    generated_length_minimum: int = 40,
    generated_length_maximum: int = 100,
    subunit_minimum: int | None = None,
    subunit_maximum: int | None = None,
    diameter_minimum: float | None = None,
    diameter_maximum: float | None = None,
    cavity_diameter_minimum: float | None = None,
    cavity_diameter_maximum: float | None = None,
    profile: str = "p100",
) -> tuple[Path, Path]:
    intent_payload = simple_intent_payload(
        inspection,
        name=intent_name,
        architecture=architecture,
        composition=composition,
        symmetries=symmetries,
        generated_length_minimum=generated_length_minimum,
        generated_length_maximum=generated_length_maximum,
        subunit_minimum=subunit_minimum,
        subunit_maximum=subunit_maximum,
        diameter_minimum=diameter_minimum,
        diameter_maximum=diameter_maximum,
        cavity_diameter_minimum=cavity_diameter_minimum,
        cavity_diameter_maximum=cavity_diameter_maximum,
        profile=profile,
    )
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "structure_inspection.json"
    intent_path = root / "simple_design.yaml"
    report_path.write_text(
        json.dumps(inspection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    intent_path.write_text(
        yaml.safe_dump(
            intent_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return report_path, intent_path


__all__ = [
    "ChainInspection",
    "ComponentInterfaceSummary",
    "DeclaredInterfaceEvidence",
    "InterfaceCandidate",
    "StructureInspection",
    "inspect_structure_interfaces",
    "inspect_declared_interface_seed",
    "inspect_declared_interface_relation",
    "simple_intent_payload",
    "write_structure_inspection",
]
