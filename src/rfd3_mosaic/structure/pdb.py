"""Small deterministic PDB reader for standalone Interface-Seed compilation."""

from dataclasses import dataclass
import gzip
from pathlib import Path
import shlex


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


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _cif_value(values: list[str], fields: dict[str, int], *names: str) -> str:
    for name in names:
        index = fields.get(name)
        if index is not None and index < len(values):
            value = values[index]
            if value not in {".", "?"}:
                return value
    return ""


def read_mmcif_atoms(
    path: str | Path,
    *,
    identifier_namespace: str = "author",
) -> tuple[AtomRecord, ...]:
    """Read the first ``_atom_site`` model from an mmCIF or mmCIF.gz file.

    RFD3 output files use one atom-site row per physical line.  Keeping this
    small reader local makes post-generation seed audits independent of the
    optional AtomWorks mirror configuration.
    """

    if identifier_namespace not in {"author", "label"}:
        raise ValueError(
            "mmCIF identifier_namespace must be 'author' or 'label'"
        )
    cif_path = Path(path)
    if not cif_path.is_file():
        raise FileNotFoundError(f"mmCIF file does not exist: {cif_path}")

    with _open_text(cif_path) as handle:
        lines = handle.readlines()

    field_names: list[str] = []
    row_start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        cursor = index + 1
        candidate: list[str] = []
        while cursor < len(lines) and lines[cursor].startswith("_atom_site."):
            candidate.append(lines[cursor].strip().split(".", 1)[1])
            cursor += 1
        if candidate:
            field_names = candidate
            row_start = cursor
            break

    if row_start is None:
        raise ValueError(f"mmCIF contains no _atom_site loop: {cif_path}")
    fields = {name: index for index, name in enumerate(field_names)}
    required = {"Cartn_x", "Cartn_y", "Cartn_z"}
    if not required.issubset(fields):
        raise ValueError(
            f"mmCIF atom-site loop lacks coordinates: {cif_path}"
        )

    atoms: list[AtomRecord] = []
    for line_number, line in enumerate(lines[row_start:], start=row_start + 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
            break
        values = shlex.split(line, comments=False, posix=True)
        if len(values) != len(field_names):
            raise ValueError(
                "Unsupported wrapped mmCIF atom row at line "
                f"{line_number}: expected {len(field_names)} values, "
                f"found {len(values)}"
            )
        model = _cif_value(values, fields, "pdbx_PDB_model_num")
        if model and model != "1":
            continue
        record_type = _cif_value(values, fields, "group_PDB") or "ATOM"
        if identifier_namespace == "label":
            atom_fields = ("label_atom_id", "auth_atom_id")
            residue_fields = ("label_comp_id", "auth_comp_id")
            chain_fields = ("label_asym_id", "auth_asym_id")
            sequence_fields = ("label_seq_id", "auth_seq_id")
        else:
            atom_fields = ("auth_atom_id", "label_atom_id")
            residue_fields = ("auth_comp_id", "label_comp_id")
            chain_fields = ("auth_asym_id", "label_asym_id")
            sequence_fields = ("auth_seq_id", "label_seq_id")
        atom_name = _cif_value(values, fields, *atom_fields)
        residue_name = _cif_value(values, fields, *residue_fields)
        chain_id = _cif_value(values, fields, *chain_fields)
        residue_number = _cif_value(values, fields, *sequence_fields)
        if not atom_name or not chain_id or not residue_number:
            raise ValueError(
                f"Incomplete mmCIF atom identity at line {line_number}"
            )
        atoms.append(
            AtomRecord(
                record_type=record_type,
                serial=int(_cif_value(values, fields, "id") or len(atoms) + 1),
                atom_name=atom_name,
                alternate_location=_cif_value(
                    values, fields, "label_alt_id"
                ),
                residue_name=residue_name,
                chain_id=chain_id,
                residue_number=int(residue_number),
                insertion_code=_cif_value(
                    values, fields, "pdbx_PDB_ins_code"
                ),
                coordinate=(
                    float(values[fields["Cartn_x"]]),
                    float(values[fields["Cartn_y"]]),
                    float(values[fields["Cartn_z"]]),
                ),
                element=_cif_value(values, fields, "type_symbol"),
            )
        )

    if not atoms:
        raise ValueError(f"mmCIF contains no usable atom records: {cif_path}")
    return tuple(atoms)


def read_structure_atoms(
    path: str | Path,
    *,
    mmcif_identifier_namespace: str = "author",
) -> tuple[AtomRecord, ...]:
    """Read PDB or mmCIF atoms from plain or gzip-compressed files."""

    structure_path = Path(path)
    lowered = structure_path.name.lower()
    if lowered.endswith((".cif", ".cif.gz")):
        return read_mmcif_atoms(
            structure_path,
            identifier_namespace=mmcif_identifier_namespace,
        )
    if lowered.endswith((".pdb", ".pdb.gz")):
        if lowered.endswith(".gz"):
            raise ValueError("Compressed PDB input is not currently supported")
        return read_pdb_atoms(structure_path)
    raise ValueError(f"Unsupported structure format: {structure_path}")
