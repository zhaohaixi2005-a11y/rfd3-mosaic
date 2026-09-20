"""Sparse donor/acceptor requests must reach the native metric calculation."""
import biotite.structure as struc
import numpy as np
import pytest
from rfd3.metrics import hbonds_metrics as module


@pytest.mark.parametrize("donor_only", [False, True])
def test_sparse_hbond_requests_are_measured(monkeypatch, donor_only):
    atoms = struc.AtomArray(2)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 2]
    atoms.atom_name[:] = ["N", "O"]
    atoms.res_name[:] = "ALA"
    atoms.element[:] = ["N", "O"]
    atoms.coord[:] = [[0, 0, 0], [3, 0, 0]]
    for name, value in {
        "active_donor": [True, False], "chain_type": [1, 1], "gt_atom_name": ["N", "O"],
        "is_motif_atom_with_fixed_coord": [True, True], "is_motif_atom_with_fixed_seq": [True, True],
        "is_motif_atom_unindexed": [False, False], "atomize": [False, False],
    }.items():
        atoms.set_annotation(name, np.array(value))
    if not donor_only:
        atoms.set_annotation("active_acceptor", np.array([False, True]))
    calls = []
    def measure(array, left, right, **kwargs):
        calls.append(True)
        return [object()], np.array([[True, False], [False, True]]), array
    monkeypatch.setattr(module, "simplified_processing_atom_array", lambda x: [a.copy() for a in x])
    monkeypatch.setattr(module, "add_hydrogen_atom_positions", lambda x: x)
    monkeypatch.setattr(module, "remove_hydrogens", lambda x: x)
    monkeypatch.setattr(module, "calculate_hbonds", measure)
    result = module.calculate_hbond_stats([atoms], [atoms], [1], [1], "donor", 3.5, 120., ["N"], ["O"], False)
    assert calls == [True]
    assert result == (1., 1., 1.)
    if donor_only:
        assert "active_acceptor" not in atoms.get_annotation_categories()
