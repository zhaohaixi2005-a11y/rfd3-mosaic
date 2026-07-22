"""Small deterministic PDB reader for standalone Interface-Seed compilation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AtomRecord:
    record_type: str
    serial: int
    atom_name: str
    alternate_location: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    coordinate: tuple[float, float, float]
    element: str

    @property
    def residue_id(self) -> tuple[str, int, str]:
        return (self.chain_id, self.residue_number, self.insertion_code)


def _parse_atom_line(line: str, line_number: int) -> AtomRecord:
    try:
        return AtomRecord(
            record_type=line[0:6].strip(),
            serial=int(line[6:11]),
            atom_name=line[12:16].strip(),
            alternate_location=line[16:17].strip(),
            residue_name=line[17:20].strip(),
            chain_id=line[21:22].strip(),
            residue_number=int(line[22:26]),
            insertion_code=line[26:27].strip(),
            coordinate=(
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ),
            element=line[76:78].strip(),
        )
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"Invalid PDB atom record at line {line_number}: {line.rstrip()}"
        ) from error


def read_pdb_atoms(path: str | Path) -> tuple[AtomRecord, ...]:
    """Read ATOM/HETATM records, resolving alternate locations predictably."""

    pdb_path = Path(path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")

    selected: dict[tuple[str, int, str, str, str], AtomRecord] = {}
    with pdb_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line[0:6].strip() not in {"ATOM", "HETATM"}:
                continue
            atom = _parse_atom_line(line, line_number)
            if atom.alternate_location not in {"", "A"}:
                continue
            key = (
                atom.chain_id,
                atom.residue_number,
                atom.insertion_code,
                atom.residue_name,
                atom.atom_name,
            )
            existing = selected.get(key)
            if existing is None or (
                existing.alternate_location == "A"
                and atom.alternate_location == ""
            ):
                selected[key] = atom

    if not selected:
        raise ValueError(f"PDB file contains no usable atom records: {pdb_path}")
    return tuple(selected.values())
