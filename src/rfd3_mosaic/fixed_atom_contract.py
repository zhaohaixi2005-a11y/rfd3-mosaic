"""Resolve coordinate/sequence intent without weakening physical tolerances."""
import re

_BACKBONE = frozenset(("N", "CA", "C", "O"))
_SELECTOR = re.compile(r"^([^0-9,+-]+)([0-9]+)(?:-([0-9]+))?$")


def resolve_fixed_atom_scope(atoms, *, explicit, sequence_conditioned, redesign):
    scope = getattr(atoms, "value", atoms)
    # A sequence clause alone is not permission to discard sidechain coordinates.
    # Only the explicit sidechain-redesign opt-in resolves an omitted default.
    if scope == "all" and redesign:
        if explicit:
            raise ValueError(
                "Explicit atoms=all conflicts with motif sequence/sidechain "
                "redesign; preserve source identity or declare atoms=backbone"
            )
        return "backbone"
    return scope


def atom_is_selected(atom_name, selection):
    name = atom_name.upper()
    if selection is True or selection is None:
        return True
    if selection is False:
        return False
    if isinstance(selection, (list, tuple)):
        values = {str(x).upper() for x in selection}
    else:
        values = {x.strip().upper() for x in str(selection).split(",")}
    if values & {"ALL", "*"}:
        return True
    if values & {"BKBN", "BACKBONE"}:
        return name in _BACKBONE
    return name in values


def source_atom_is_fixed(example, chain, residue, atom_name):
    selections = example.get("select_fixed_atoms")
    if selections is None or isinstance(selections, bool):
        return atom_is_selected(atom_name, selections)
    if not isinstance(selections, dict):
        raise ValueError("Fixed-atom audit requires explicit compiled selections")
    for selectors, selection in selections.items():
        for part in selectors.split(","):
            match = _SELECTOR.fullmatch(part.strip())
            if match is None:
                raise ValueError("Unsupported compiled fixed selector: " + part)
            selected_chain, first, last = match.groups()
            if selected_chain == chain and int(first) <= residue <= int(last or first):
                if atom_is_selected(atom_name, selection):
                    return True
    return False
