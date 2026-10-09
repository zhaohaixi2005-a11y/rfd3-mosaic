"""CPU regression for legacy metrics; no hydrogenation/CCD or model weights."""

import biotite.structure as struc
import numpy as np
import pytest
from rfd3na.metrics import hbonds_metrics


def test_metric_module_import_and_construction():
    # Import previously failed on the retired foundry.metrics.base module.
    assert isinstance(hbonds_metrics.HbondMetrics(), hbonds_metrics.Metric)


@pytest.mark.parametrize("donor_only", [False, True])
def test_sparse_requests_reach_metric_and_resolve_motif_features(
    monkeypatch, donor_only
):
    atoms = struc.AtomArray(2)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 2]
    atoms.atom_name[:] = ["N", "O"]
    atoms.res_name[:] = "ALA"
    atoms.element[:] = ["N", "O"]
    atoms.coord[:] = [[0, 0, 0], [3, 0, 0]]
    for name, values in {
        "active_donor": [True, False],
        "chain_type": [1, 1],
        "gt_atom_name": ["N", "O"],
        "is_motif_atom_with_fixed_coord": [True, True],
        "is_motif_atom_with_fixed_seq": [True, True],
        "is_motif_atom_unindexed": [False, False],
        "atomize": [False, False],
    }.items():
        atoms.set_annotation(name, np.asarray(values))
    if not donor_only:
        atoms.set_annotation("active_acceptor", np.array([False, True]))

    calls = []

    def measure(array, left, right, **kwargs):
        # get_motif_features is deliberately real: the former missing import
        # must not be hidden by a mocked name in the regression.
        np.testing.assert_array_equal(right, [True, True])
        calls.append(True)
        return [object()], np.array([[True, False], [False, True]]), array

    monkeypatch.setattr(
        hbonds_metrics,
        "simplified_processing_atom_array",
        lambda arrays: [array.copy() for array in arrays],
    )
    monkeypatch.setattr(hbonds_metrics, "add_hydrogen_atom_positions", lambda a: a)
    monkeypatch.setattr(hbonds_metrics, "remove_hydrogens", lambda a: a)
    monkeypatch.setattr(hbonds_metrics, "calculate_hbonds", measure)
    result = hbonds_metrics.calculate_hbond_stats(
        [atoms], [atoms], [1], [2], "donor", 3.5, 120.0, ["N"], ["O"], False
    )
    assert calls == [True]
    assert result == (1.0, 1.0, 1.0)
    if donor_only:
        assert "active_acceptor" not in atoms.get_annotation_categories()
