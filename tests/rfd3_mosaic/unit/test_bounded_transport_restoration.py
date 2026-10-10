"""Strict guards assess actual repaired coordinates against an unchanged prior."""

import pytest
import torch
from rfd3.inference.symmetry.scaffold_core_guidance import build_scaffold_core_topology
from rfd3.inference.symmetry.scaffold_transport import ScaffoldReferenceTransport
from test_reference_transport import transport_fixture


@pytest.mark.parametrize('deformation, accepted', [(1e-6, True), (8e-5, False), (1e-3, False)])
def test_coordinate_restoration_keeps_reference_and_rejects_real_regression(deformation, accepted):
    _, _, _, features, _ = transport_fixture()
    features['motif_pos'] = features['motif_pos'].float()
    fixed = torch.tensor([r['fixed'] for r in features['mosaic_scaffold_contract']['residues']]).repeat_interleave(4)
    runtime = ScaffoldReferenceTransport(features, build_scaffold_core_topology(features, fixed))
    before = features['motif_pos'].clone()
    before[runtime.generated, 2] += 1.5
    candidate = before.clone()
    candidate[runtime.generated, 2] += deformation
    old_reference = runtime.reference.reference_backbone.clone()
    restored, _, _, guard = runtime._restore_geometry(before, candidate, runtime.reference, before, lambda x, t: x)
    assert guard['passed'] is accepted
    assert torch.equal(runtime.reference.reference_backbone, old_reference)
    assert torch.equal(restored[fixed], before[fixed])
    if accepted:
        assert float((restored-candidate).abs().max()) <= 8*torch.finfo(before.dtype).eps*float(candidate.abs().max())
    else:
        assert torch.equal(restored, candidate)
    assert runtime.accepted == []


def test_restoration_cannot_accept_a_projector_that_changes_fixed_atoms():
    _, _, _, features, _ = transport_fixture()
    features['motif_pos'] = features['motif_pos'].float()
    fixed = torch.tensor([r['fixed'] for r in features['mosaic_scaffold_contract']['residues']]).repeat_interleave(4)
    runtime = ScaffoldReferenceTransport(features, build_scaffold_core_topology(features, fixed))
    before = features['motif_pos'].clone()
    before[runtime.generated, 2] += 1.5
    candidate = before.clone()
    candidate[runtime.generated, 2] += 1e-6
    def invalid_projector(x, target):
        x = x.clone()
        x[:, fixed, 2] += .001
        return x
    result, _, _, guard = runtime._restore_geometry(before, candidate, runtime.reference, before, invalid_projector)
    assert not guard['passed']
    assert torch.equal(result, candidate)


@pytest.mark.parametrize('deformation', [1e-7, 5e-7])
def test_distributed_float64_regression_is_not_a_rounding_allowance(deformation):
    from rfd3.inference.symmetry.reference_scaffold import (
        reference_scaffold_guidance_deficits,
    )
    from rfd3.inference.symmetry.scaffold_core_guidance import route_nonregression_check
    _, _, _, features, _ = transport_fixture()
    features['motif_pos'] = features['motif_pos'].double()
    fixed = torch.tensor([r['fixed'] for r in features['mosaic_scaffold_contract']['residues']]).repeat_interleave(4)
    runtime = ScaffoldReferenceTransport(features, build_scaffold_core_topology(features, fixed))
    before = features['motif_pos'].clone()
    before[runtime.generated, 2] += 1.5
    candidate = before.clone()
    candidate[runtime.generated, 2] += deformation
    b = reference_scaffold_guidance_deficits(before, runtime.reference)
    a = reference_scaffold_guidance_deficits(candidate, runtime.reference)
    initial = route_nonregression_check(b, a, tolerance=runtime.base['limits']['geometry_tolerance'])
    assert initial['maximum_nonregression']
    assert not initial['squared_sum_nonregression']
    result, _, _, guard = runtime._restore_geometry(before, candidate, runtime.reference, before, lambda x,t:x)
    assert not guard['passed']
    assert torch.equal(result, candidate)
