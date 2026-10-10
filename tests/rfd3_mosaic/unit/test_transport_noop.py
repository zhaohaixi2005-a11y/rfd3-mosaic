"""A no-pose trial must preserve its accepted scaffold and stored reference."""
from types import SimpleNamespace

import pytest
import torch
from rfd3.inference.symmetry.scaffold_core_guidance import build_scaffold_core_topology
from rfd3.inference.symmetry.scaffold_transport import ScaffoldReferenceTransport
from test_reference_transport import transport_fixture


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("explicit_coordinates", [False, True])
def test_zero_pose_does_not_rebuild_rounded_accepted_state(dtype, explicit_coordinates):
    _, _, _, features, _ = transport_fixture()
    features["motif_pos"] = features["motif_pos"].to(dtype)
    fixed = torch.tensor([r["fixed"] for r in features["mosaic_scaffold_contract"]["residues"]]).repeat_interleave(4)
    topology = build_scaffold_core_topology(features, fixed)
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features["motif_pos"][None].clone()
    def unexpected_projection(candidate, target):
        raise AssertionError("An exact no-op must not reproject the accepted state")

    proposal = SimpleNamespace(
        target=original.clone(), coordinates=original.clone() if explicit_coordinates else None
    )
    prepared = runtime.prepare(original, proposal, projector=unexpected_projection)
    assert torch.equal(prepared[0], original)
    assert torch.equal(prepared[1].reference_backbone, runtime.reference.reference_backbone)
    assert runtime.accepted == []


def test_unchanged_fixed_pose_does_not_skip_generated_deformation_guard():
    _, _, _, features, topology = transport_fixture()
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features["motif_pos"][None].clone()
    deformed = original.clone()
    generated_index = int(torch.nonzero(topology.generated_atom_mask)[0])
    deformed[0, generated_index, 0] += 10.0
    with pytest.raises(ValueError, match="regresses the complete-scaffold contract"):
        runtime.prepare(original, SimpleNamespace(target=original, coordinates=deformed), projector=lambda x, t: x)
    assert runtime.accepted == []


def test_exact_noop_after_committed_motion_keeps_reference_and_pose_history():
    _, _, _, features, topology = transport_fixture()
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features["motif_pos"][None].clone()
    target = original.clone()
    target[:, :, 2] += 0.1
    prepared = runtime.prepare(original, SimpleNamespace(target=target, coordinates=None), projector=lambda x, t: x)
    runtime.commit(prepared, progress=0.2)
    current = prepared[0]
    again = runtime.prepare(current, SimpleNamespace(target=current.clone(), coordinates=None), projector=lambda x, t: x)
    assert torch.equal(again[0], current)
    assert again[1] is runtime.reference
    assert len(runtime.accepted) == 1
    assert again[3] == runtime.motions
