from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def test_unindexed_masks_do_not_slice_the_token_level_bond_graph(self) -> None:
        class AnnotationOnlyArray:
            is_motif_atom_unindexed = np.asarray(
                [True, False, True], dtype=bool
            )
            is_motif_atom_unindexed_motif_breakpoint = np.asarray(
                [False, False, True], dtype=bool
            )

            def __getitem__(self, index):
                raise AssertionError(
                    "mask construction must not slice an AtomArray BondList"
                )

        # Duplicate token representatives reproduce the condition that made
        # Biotite's second BondList slice fail for explicit T/O/I expansions.
        with patch(
            "rfd3.transforms.conditioning_base.get_token_starts",
            return_value=np.asarray([0, 0, 2], dtype=np.int64),
        ):
            mask_ll, mask_i = UnindexFlaggedTokens(
                central_atom="CA"
            ).create_unindexed_masks(
                AnnotationOnlyArray(),
                is_inference=True,
            )

        self.assertEqual(mask_i.tolist(), [True, True, True])
        self.assertEqual(mask_ll.shape, (3, 3))
        np.testing.assert_array_equal(
            mask_ll,
            np.asarray(
                [
                    [False, False, True],
                    [False, False, True],
                    [True, True, False],
                ],
                dtype=bool,
            ),
        )


if __name__ == "__main__":
    unittest.main()
