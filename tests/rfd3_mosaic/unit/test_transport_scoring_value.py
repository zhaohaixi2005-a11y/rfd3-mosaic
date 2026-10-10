"""Score exactly the state later validated; retain the rigid-pose derivative."""
from types import SimpleNamespace

import pytest
import torch
from rfd3.inference.symmetry.scaffold_core_guidance import build_scaffold_core_topology
from rfd3.inference.symmetry.scaffold_transport import ScaffoldReferenceTransport
from test_reference_transport import transport_fixture


@pytest.mark.parametrize("with_gradient", [False, True])
def test_float32_score_value_equals_prepared_candidate(with_gradient):
    _, _, _, features, _ = transport_fixture()
    features["motif_pos"] = features["motif_pos"].float()
    fixed = torch.tensor([r["fixed"] for r in features["mosaic_scaffold_contract"]["residues"]]).repeat_interleave(4)
    topology = build_scaffold_core_topology(features, fixed)
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features["motif_pos"].clone()
    target = original.clone()
    target[:, 2] += .125
    target.requires_grad_(with_gradient)
    scoring = runtime.candidate_coordinates(target, original, projector=lambda x,t:x)
    prepared = runtime.prepare(original[None], SimpleNamespace(target=target.detach()[None], coordinates=None), projector=lambda x,t:x)
    assert torch.equal(scoring.detach(), prepared[0][0])
    if with_gradient:
        scoring.square().sum().backward()
        assert torch.isfinite(target.grad).all()
        assert float(target.grad.abs().sum()) > 0
    assert runtime.accepted == []


@pytest.mark.parametrize('with_gradient', [False, True])
def test_noop_and_packing_score_match_every_prepare_branch(with_gradient):
    _, _, _, features, _ = transport_fixture()
    features['motif_pos'] = features['motif_pos'].float()
    fixed = torch.tensor([r['fixed'] for r in features['mosaic_scaffold_contract']['residues']]).repeat_interleave(4)
    runtime = ScaffoldReferenceTransport(features, build_scaffold_core_topology(features, fixed))
    before = features['motif_pos'].clone()
    for phase in ('initial_noop', 'translated', 'accepted_noop', 'packing'):
        target = before.clone()
        source = before.clone()
        if phase == 'translated':
            target[:, 2] += .125
        if phase == 'packing':
            source[runtime.generated, 0] += .0001
        target.requires_grad_(with_gradient)
        scored = runtime.candidate_coordinates(target, source, projector=lambda x,t:x, before=before)
        prepared = runtime.prepare(before[None], SimpleNamespace(target=target.detach()[None], coordinates=source[None]), projector=lambda x,t:x)
        assert torch.equal(scored.detach(), prepared[0][0]), phase
        if with_gradient:
            gradient, = torch.autograd.grad(scored.square().sum(), target)
            assert torch.isfinite(gradient).all(), phase
        if phase == 'translated':
            runtime.commit(prepared, progress=.1)
            before = prepared[0][0]
