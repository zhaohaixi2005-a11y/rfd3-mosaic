"""Parser and resolver for Interface-Seed fragment atom selections."""

from dataclasses import dataclass
from pathlib import Path
import re

from rfd3_mosaic.schema import FragmentSpec
from rfd3_mosaic.structure.pdb import AtomRecord, read_pdb_atoms


_SELECTION_PATTERN = re.compile(
    r"^(?P<chain>[^/]+)/(?P<start>-?\d+)"
    r"(?:-(?P<end>-?\d+))?/(?P<atoms>[^/]+)$"
)
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True)
class AtomSelection:
    chain_id: str
    residue_start: int
    residue_end: int
    atom_names: frozenset[str] | None


def parse_atom_selection(expression: str) -> AtomSelection:
    """Parse ``CHAIN/START-END/ATOMS`` used by Interface-Seed configs."""

    match = _SELECTION_PATTERN.fullmatch(expression.strip())
    if match is None:
        raise ValueError(
            "Selection must use CHAIN/START-END/ATOMS syntax, "
            f"got {expression!r}"
        )

    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if start > end:
        raise ValueError("Selection residue start cannot exceed residue end")

    atom_expression = match.group("atoms").strip()
    if atom_expression == "*":
        atom_names = None
    elif atom_expression.lower() == "backbone":
        atom_names = _BACKBONE_ATOMS
    else:
        names = frozenset(
            name.strip().upper()
            for name in atom_expression.split(",")
            if name.strip()
        )
        if not names:
            raise ValueError("Selection atom-name list cannot be empty")
        atom_names = names

    return AtomSelection(
        chain_id=match.group("chain"),
        residue_start=start,
        residue_end=end,
        atom_names=atom_names,
    )


def select_atoms(
    atoms: tuple[AtomRecord, ...],
    selection: AtomSelection | str,
) -> tuple[AtomRecord, ...]:
    """Resolve a parsed selection and fail explicitly when it is empty."""

    resolved = (
        parse_atom_selection(selection)
        if isinstance(selection, str)
        else selection
    )
    matches = tuple(
        atom
        for atom in atoms
        if atom.chain_id == resolved.chain_id
        and resolved.residue_start
        <= atom.residue_number
        <= resolved.residue_end
        and (
            resolved.atom_names is None
            or atom.atom_name.upper() in resolved.atom_names
        )
    )
    if not matches:
        raise ValueError(
            "Selection resolved to zero atoms: "
            f"chain={resolved.chain_id!r}, "
            f"residues={resolved.residue_start}-{resolved.residue_end}"
        )
    return matches


def select_atom_subset(
    atoms: tuple[AtomRecord, ...],
    expression: str,
) -> tuple[AtomRecord, ...]:
    """Apply a port-level atom-name selector to resolved fragment atoms."""

    normalized = expression.strip()
    if normalized in {"*", "all"}:
        selected = atoms
    elif normalized == "heavy":
        selected = tuple(
            atom
            for atom in atoms
            if not (
                atom.element.upper().startswith("H")
                or atom.atom_name.lstrip("0123456789").upper().startswith("H")
            )
        )
    elif normalized == "backbone":
        selected = tuple(
            atom for atom in atoms if atom.atom_name.upper() in _BACKBONE_ATOMS
        )
    else:
        names = {
            name.strip().upper()
            for name in normalized.split(",")
            if name.strip()
        }
        if not names:
            raise ValueError("Port atom selector cannot be empty")
        selected = tuple(
            atom for atom in atoms if atom.atom_name.upper() in names
        )

    if not selected:
        raise ValueError(
            f"Port atom selector {expression!r} resolved to zero atoms"
        )
    return selected


def load_selected_atoms(
    fragment: FragmentSpec,
    *,
    base_directory: str | Path = ".",
) -> tuple[AtomRecord, ...]:
    """Load a fragment source and resolve its configured atom selection."""

    source_path = fragment.source
    if not source_path.is_absolute():
        source_path = Path(base_directory) / source_path
    return select_atoms(read_pdb_atoms(source_path), fragment.selection)
