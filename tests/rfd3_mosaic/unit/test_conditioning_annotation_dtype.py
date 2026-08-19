from __future__ import annotations

import unittest

import numpy as np
from biotite.structure import AtomArray, BondList, BondType

from rfd3.transforms.conditioning_base import (
    UnindexFlaggedTokens,
    convert_existing_annotations_to_bool,
)


class ConditioningAnnotationDtypeTestCase(unittest.TestCase):
    def test_integer_cif_flags_become_true_boolean_masks(self) -> None:
        atoms = AtomArray(6)
        atoms.chain_id[:] = "A"
        atoms.res_id[:] = [1, 1, 2, 2, 3, 3]
        atoms.ins_code[:] = ""
        atoms.res_name[:] = "ALA"
        atoms.atom_name[:] = ["N", "CA"] * 3
        atoms.element[:] = ["N", "C"] * 3
        atoms.coord[:] = np.arange(18, dtype=float).reshape(6, 3)
        atoms.bonds = BondList(
            6,
            np.asarray(
                [
                    [0, 1, int(BondType.SINGLE)],
                    [2, 3, int(BondType.SINGLE)],
                    [4, 5, int(BondType.SINGLE)],
                ]
            ),
        )
        atoms.set_annotation(
            "is_motif_atom_unindexed",
            np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64),
        )
        atoms.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint",
            np.zeros(6, dtype=np.int64),
        )

        convert_existing_annotations_to_bool(
            atoms,
            annotations=(
                "is_motif_atom_unindexed",
                "is_motif_atom_unindexed_motif_breakpoint",
            ),
        )
        mask_ll, mask_i = UnindexFlaggedTokens(
            central_atom="CA"
        ).create_unindexed_masks(atoms, is_inference=True)

        self.assertEqual(atoms.is_motif_atom_unindexed.dtype, np.dtype(bool))
        self.assertEqual(mask_i.dtype, np.dtype(bool))
        self.assertEqual(mask_i.tolist(), [False, False, True])
        self.assertEqual(mask_ll.dtype, np.dtype(bool))
        self.assertEqual(mask_ll.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
