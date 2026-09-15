"""Shared finite-symmetry name grammar for schemas and resolver frontends."""

import re

# No leading zero; include all integers >= 2, including 10, 12 and 100.
NONTRIVIAL_ORDER_PATTERN = r"(?:[2-9]|[1-9][0-9]+)"
SYMMETRY_NAME_PATTERN = (
    rf"^(?:C[1-9][0-9]*|D{NONTRIVIAL_ORDER_PATTERN}|T|O|I)$"
)
CYCLIC_NAME = re.compile(r"^C(?P<order>[1-9][0-9]*)$")
NONTRIVIAL_CYCLIC_NAME = re.compile(
    rf"^C(?P<order>{NONTRIVIAL_ORDER_PATTERN})$"
)
DIHEDRAL_NAME = re.compile(rf"^D(?P<order>{NONTRIVIAL_ORDER_PATTERN})$")
