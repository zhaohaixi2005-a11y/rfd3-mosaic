import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.output import compile_rfd3_input
from rfd3_mosaic.schema import UserDesignSpec
from rfd3_mosaic.rfd3_scaffold_audit import _build_runtime_input
from rfd3.transforms.symmetry import AddSymmetryFeats
from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry
from test_cylindrical_public_runtime import _write_motif


@pytest.fixture
def bound(tmp_path):
    motif = tmp_path / 'motif.pdb'
    _write_motif(motif)
    design = UserDesignSpec.model_validate({
        'name': 'source-reference-test', 'input': str(motif), 'symmetry': 'C3',
        'generation': [{'kind': 'terminal', 'anchor': 'A1-3', 'terminus': 'c', 'length': 8}],
        'constraints': [{'kind': 'cylindrical', 'selector': 'A1-3', 'atoms': 'ca', 'keep': ['radius', 'axial']}],
    })
    lowered = lower_user_design(design)
    config = tmp_path / 'assembly.yaml'
    config.write_text(yaml.safe_dump({'assembly': lowered.specification.model_dump(mode='json')}))
    output = compile_rfd3_input(config, tmp_path / 'compiled', extra_metadata=lowered.runtime_constraint_metadata)
    payload = json.loads(output.input_path.read_text())
    declaration = next(iter(payload.values()))['extra']['cylindrical_constraints']
    array = _build_runtime_input(output.input_path)
    features = AddSymmetryFeats.make_cylindrical_constraint_features(array, declaration)
    # Exercise the actual native sampler constructor method, including reference
    # selection and affine group centre; no network or checkpoint is involved.
    identity = torch.eye(3)
    holder = SimpleNamespace(allow_realignment=False, _symmetry_features=lambda _: {'sym_transform': {0: (identity, torch.zeros(3))}})
    initializer = torch.from_numpy(array.coord.copy()).unsqueeze(0)
    return array, declaration, features, holder, initializer


def test_native_zero_initializer_does_not_replace_original_targets(bound):
    array, declaration, features, holder, initializer = bound
    active = features['cylindrical_keep_mask'].any(dim=1)
    assert active.sum() == 9
    assert torch.count_nonzero(initializer[:, active]) == 0
    original = features['cylindrical_reference'][active]
    assert torch.linalg.vector_norm(original[:, :2], dim=-1).min() > 10
    assert torch.count_nonzero(original[:, 2]) > 0
    projector = SampleDiffusionWithSymmetry._cylindrical_projector(holder, features, initializer)
    rng = torch.random.get_rng_state().clone()
    projected = projector.project(initializer)
    torch.testing.assert_close(projected[0, active], original)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert projector.maximum_error(projected) < 1e-5
    torch.testing.assert_close(torch.linalg.vector_norm(projected[0, active, :2], dim=-1), torch.linalg.vector_norm(original[:, :2], dim=-1))
    assert torch.count_nonzero(projected[:, ~active]) == 0
    assert features['cylindrical_reference'][~active].count_nonzero() == 0


def test_missing_source_reference_fails_before_silent_origin_fallback(bound):
    array, declaration, features, holder, initializer = bound
    missing = dict(features)
    del missing['cylindrical_reference']
    with pytest.raises(ValueError, match='preserved source reference'):
        SampleDiffusionWithSymmetry._cylindrical_projector(holder, missing, initializer)
    for name in ('x', 'y', 'z'):
        array.del_annotation('cylindrical_reference_' + name)
    with pytest.raises(ValueError, match='preserved source reference'):
        AddSymmetryFeats.make_cylindrical_constraint_features(array, declaration)


@pytest.mark.parametrize('kind', ['shape', 'nonfinite'])
def test_invalid_source_reference_rejected(bound, kind):
    _, _, features, holder, initializer = bound
    changed = dict(features)
    changed['cylindrical_reference'] = features['cylindrical_reference'].clone()
    if kind == 'shape':
        changed['cylindrical_reference'] = changed['cylindrical_reference'][:-1]
    else:
        changed['cylindrical_reference'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='source reference'):
        SampleDiffusionWithSymmetry._cylindrical_projector(holder, changed, initializer)


def test_ca_projection_transports_real_residue_geometry_and_sidechain():
    from rfd3.inference.symmetry.cylindrical_projector import CylindricalCoordinateProjector
    # Actual two-atom counterexample extended to a complete residue and CB.
    xyz = torch.tensor([[[0.542, 0., 0.], [2., 0., 0.], [2.55, 1.42, 0.], [2.3, 2.6, 0.], [2.55, -0.8, 1.2]]])
    ref = xyz.clone(); ref[:, 1, 0] = 2.542
    keep = torch.zeros(5, 3, dtype=torch.bool); keep[1, 0] = True
    projector = CylindricalCoordinateProjector(ref, keep, torch.tensor([0., 0., 1.]), torch.zeros(3), torch.zeros(5, dtype=torch.long), torch.tensor([False, True, False, False, False]))
    result = projector.project(xyz)
    torch.testing.assert_close(torch.cdist(result, result), torch.cdist(xyz, xyz))
    torch.testing.assert_close(result[:, 1], ref[:, 1])
    assert projector.maximum_error(result) < 1e-6
    assert abs(torch.linalg.vector_norm(result[0, 0]-result[0, 1]).item()-1.458) < 1e-6
    # The legacy scalar projector remains available when no biological atom
    # binding was supplied; it reproduces the original failing counterexample.
    scalar = CylindricalCoordinateProjector(ref, keep, torch.tensor([0., 0., 1.]), torch.zeros(3)).project(xyz)
    assert abs(torch.linalg.vector_norm(scalar[0, 0]-scalar[0, 1]).item()-2.) < 1e-6


def test_ca_transport_rejects_fixed_xyz_conflict():
    from rfd3.inference.symmetry.cylindrical_projector import CylindricalCoordinateProjector
    xyz = torch.ones(1, 2, 3)
    keep = torch.zeros(2, 3, dtype=torch.bool);keep[1, 0] = True
    with pytest.raises(ValueError, match='fixed-XYZ'):
        CylindricalCoordinateProjector(xyz, keep, torch.tensor([0., 0., 1.]), torch.zeros(3), torch.zeros(2, dtype=torch.long), torch.tensor([False, True]), torch.tensor([True, False]))
