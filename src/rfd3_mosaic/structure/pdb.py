"""Small deterministic PDB reader for standalone Interface-Seed compilation."""

import gzip
import math
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
    """Read the first model, resolving residue conformers predictably."""

    pdb_path = Path(path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")

    atoms = []
    model_started = False
    with _open_text(pdb_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            record = line[0:6].strip()
            if record == "MODEL":
                if model_started or atoms:
                    break
                model_started = True
                continue
            if record in {"ENDMDL", "END"}:
                break
            if record not in {"ATOM", "HETATM"}:
                continue
            atom = _parse_atom_line(line, line_number)
            _validate_coordinates(atom, location=f"PDB line {line_number}")
            atoms.append(atom)

    if not atoms:
        raise ValueError(f"PDB file contains no usable atom records: {pdb_path}")
    return _resolve_alternate_locations(atoms)


def _validate_coordinates(atom: AtomRecord, *, location: str) -> None:
    if not all(math.isfinite(value) for value in atom.coordinate):
        raise ValueError(f"Nonfinite atom coordinates at {location}")


def _resolve_alternate_locations(atoms: list[AtomRecord]) -> tuple[AtomRecord, ...]:
    """Use shared atoms and one conformer per residue, preferring A.

    Falling back to the first available conformer retains residues whose only
    alternate label is B. Selecting one label for the whole residue avoids
    inventing a structure from atoms belonging to incompatible conformers.
    """
    labels: dict[tuple[str, int, str], str] = {}
    for atom in atoms:
        if atom.alternate_location:
            previous = labels.setdefault(atom.residue_id, atom.alternate_location)
            if previous != "A" and atom.alternate_location == "A":
                labels[atom.residue_id] = "A"
    selected: dict[tuple[str, int, str, str, str], AtomRecord] = {}
    for atom in atoms:
        if atom.alternate_location not in {"", labels.get(atom.residue_id, "")}:
            continue
        key = (*atom.residue_id, atom.residue_name, atom.atom_name)
        previous = selected.get(key)
        if (
            previous is not None
            and previous.alternate_location == atom.alternate_location
        ):
            raise ValueError(f"Duplicate atom identity in one conformer: {key}")
        if previous is None or (
            previous.alternate_location and not atom.alternate_location
        ):
            selected[key] = atom
    residue_names = {}
    for atom in selected.values():
        name = residue_names.setdefault(atom.residue_id, atom.residue_name)
        if name != atom.residue_name:
            raise ValueError(
                f"Ambiguous residue identity in one conformer: {atom.residue_id}"
            )
    return tuple(selected.values())


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
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


def _cif_tokens(lines: list[str]):
    """Tokenize CIF values without treating apostrophes in atom names as quotes."""
    index = 0
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        index += 1
        if line.startswith(";"):
            value = [line[1:].rstrip("\r\n")]
            while index < len(lines) and not lines[index].startswith(";"):
                value.append(lines[index].rstrip("\r\n"))
                index += 1
            if index == len(lines):
                raise ValueError(f"Unterminated mmCIF text field at line {line_number}")
            index += 1
            yield "\n".join(value), line_number, True
            continue
        offset = 0
        while offset < len(line):
            if line[offset].isspace():
                offset += 1
                continue
            if line[offset] == "#":
                break
            start = offset
            quote = line[offset] if line[offset] in {"'", '"'} else None
            offset += 1
            if quote:
                while offset < len(line):
                    if line[offset] == quote and (
                        offset + 1 == len(line) or line[offset + 1].isspace()
                    ):
                        break
                    offset += 1
                if offset == len(line):
                    raise ValueError(f"Unterminated mmCIF quote at line {line_number}")
                yield line[start + 1 : offset], line_number, True
                offset += 1
            else:
                while offset < len(line) and not line[offset].isspace():
                    offset += 1
                yield line[start:offset], line_number, False


def read_mmcif_atoms(
    path: str | Path,
    *,
    identifier_namespace: str = "author",
) -> tuple[AtomRecord, ...]:
    """Read the first ``_atom_site`` model from an mmCIF or mmCIF.gz file.

    Keeping this reader local makes seed compilation and post-generation
    audits independent of the optional AtomWorks mirror configuration.
    """

    if identifier_namespace not in {"author", "label"}:
        raise ValueError("mmCIF identifier_namespace must be 'author' or 'label'")
    cif_path = Path(path)
    if not cif_path.is_file():
        raise FileNotFoundError(f"mmCIF file does not exist: {cif_path}")

    with _open_text(cif_path) as handle:
        lines = handle.readlines()

    tokens = list(_cif_tokens(lines))
    field_names = []
    row_start = None
    for index, (value, _, quoted) in enumerate(tokens):
        if quoted or value != "loop_":
            continue
        cursor = index + 1
        candidate: list[str] = []
        while (
            cursor < len(tokens)
            and not tokens[cursor][2]
            and tokens[cursor][0].startswith("_atom_site.")
        ):
            candidate.append(tokens[cursor][0].split(".", 1)[1])
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
        raise ValueError(f"mmCIF atom-site loop lacks coordinates: {cif_path}")

    atoms: list[AtomRecord] = []
    first_model = None
    values = []
    for value, line_number, quoted in tokens[row_start:]:
        if not quoted and (
            value in {"loop_", "stop_"} or value.startswith(("_", "data_", "save_"))
        ):
            break
        values.append(value)
        if len(values) < len(field_names):
            continue
        model = _cif_value(values, fields, "pdbx_PDB_model_num")
        if first_model is None:
            first_model = model
        if model != first_model:
            values = []
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
            raise ValueError(f"Incomplete mmCIF atom identity at line {line_number}")
        atom = AtomRecord(
            record_type=record_type,
            serial=int(_cif_value(values, fields, "id") or len(atoms) + 1),
            atom_name=atom_name,
            alternate_location=_cif_value(values, fields, "label_alt_id"),
            residue_name=residue_name,
            chain_id=chain_id,
            residue_number=int(residue_number),
            insertion_code=_cif_value(values, fields, "pdbx_PDB_ins_code"),
            coordinate=(
                float(values[fields["Cartn_x"]]),
                float(values[fields["Cartn_y"]]),
                float(values[fields["Cartn_z"]]),
            ),
            element=_cif_value(values, fields, "type_symbol"),
        )
        _validate_coordinates(atom, location=f"mmCIF line {line_number}")
        atoms.append(atom)
        values = []

    if values:
        raise ValueError(
            "Incomplete mmCIF atom row: "
            f"expected {len(field_names)} values, found {len(values)}"
        )

    if not atoms:
        raise ValueError(f"mmCIF contains no usable atom records: {cif_path}")
    return _resolve_alternate_locations(atoms)


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
        return read_pdb_atoms(structure_path)
    raise ValueError(f"Unsupported structure format: {structure_path}")
