"""Native metric definitions, independent of model weights or CCD resources."""

import numpy as np
import pytest
from biotite.structure import AtomArray
from rfd3.metrics.design_metrics import get_clash_metrics


def _atoms(coordinates, *, residue_ids, chain_ids, atom_names=None):
    array = AtomArray(len(coordinates))
    array.coord = np.asarray(coordinates, dtype=np.float32)
    array.res_id = np.asarray(residue_ids)
    # Deliberately duplicate the display chain label: chain_iid is the
    # authoritative physical identity used by native RFD3 metrics.
    array.chain_id = np.full(len(array), "A")
    array.res_name = np.full(len(array), "ALA")
    array.atom_name = np.asarray(atom_names or ["CA"] * len(array))
    array.element = np.asarray([name[0] for name in array.atom_name])
    array.set_annotation("chain_iid", np.asarray(chain_ids))
    array.set_annotation("is_protein", np.ones(len(array), dtype=bool))
    array.set_annotation("is_ligand", np.zeros(len(array), dtype=bool))
    array.set_annotation("is_motif_atom_unindexed", np.zeros(len(array), dtype=bool))
    return array


@pytest.mark.parametrize("residue_ids", [(10, 10), (10, 11), (10, 12), (101, -9)])
def test_interchain_clash_is_invariant_to_residue_numbering(residue_ids):
    array = _atoms([[0, 0, 0], [1, 0, 0]], residue_ids=residue_ids, chain_ids=[0, 1])
    measured = get_clash_metrics(array)
    assert measured["n_clashing.interresidue_clashes_w_backbone"] == 1
    assert measured["n_clashing.interresidue_clashes_w_sidechain"] == 1
    assert measured["n_chainbreaks"] == 0


@pytest.mark.parametrize("residue_ids", [(10, 10), (10, 11), (11, 10)])
def test_same_chain_local_pairs_remain_excluded(residue_ids):
    array = _atoms([[0, 0, 0], [1, 0, 0]], residue_ids=residue_ids, chain_ids=[0, 0])
    measured = get_clash_metrics(array)
    assert measured["n_clashing.interresidue_clashes_w_backbone"] == 0
    assert measured["n_clashing.interresidue_clashes_w_sidechain"] == 0


def test_same_chain_nonlocal_clash_is_preserved():
    array = _atoms([[0, 0, 0], [1, 0, 0]], residue_ids=[10, 12], chain_ids=[0, 0])
    measured = get_clash_metrics(array)
    assert measured["n_clashing.interresidue_clashes_w_backbone"] == 1
    assert measured["n_clashing.interresidue_clashes_w_sidechain"] == 1


@pytest.mark.parametrize("distance, expected", [(1.49, 1), (1.5, 0), (1.51, 0)])
def test_native_clash_cutoff_is_unchanged(distance, expected):
    array = _atoms([[0, 0, 0], [distance, 0, 0]], residue_ids=[10, 10], chain_ids=[0, 1])
    measured = get_clash_metrics(array)
    assert measured["n_clashing.interresidue_clashes_w_backbone"] == expected
    assert measured["n_clashing.interresidue_clashes_w_sidechain"] == expected


def test_legacy_backbone_scope_excludes_oxygen_but_all_protein_scope_includes_it():
    array = _atoms(
        [[0, 0, 0], [30, 0, 0], [1, 0, 0], [50, 0, 0]],
        residue_ids=[10, 10, 10, 10], chain_ids=[0, 0, 1, 1],
        atom_names=["N", "CA", "O", "CA"],
    )
    measured = get_clash_metrics(array)
    assert measured["n_clashing.interresidue_clashes_w_backbone"] == 0
    assert measured["n_clashing.interresidue_clashes_w_sidechain"] == 1
