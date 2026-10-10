"""Physical acceptance must not change under a numerically exact rigid move."""

import pytest
import torch
from rfd3.inference.symmetry.constraint_runtime import (
    ConstraintProposalResult,
    MosaicConstraintRuntime,
)
from rfd3.inference.symmetry.joint_projector import UnifiedJointProjector
from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig,
    build_scaffold_core_topology,
    scaffold_geometry_guard,
)
from rfd3.inference.symmetry.scaffold_guidance import propose_bounded_se3_step


def geometry_fixture(chains=3, residues=151, dtype=torch.float32, device='cpu'):
    count = chains * residues
    features = {
        'atom_to_token_map': torch.arange(count, device=device),
        'asym_id': torch.arange(chains, device=device).repeat_interleave(residues),
        'residue_index': torch.arange(residues, device=device).repeat(chains),
        'is_ca': torch.ones(count, dtype=torch.bool, device=device),
        'is_protein': torch.ones(count, dtype=torch.bool, device=device),
    }
    fixed = torch.zeros(count, dtype=torch.bool, device=device)
    fixed[::residues] = True
    topology = build_scaffold_core_topology(features, fixed)
    generator = torch.Generator().manual_seed(6)
    # One binade ensures the binary translation is exact in the stored dtype.
    # Existing overlaps are intentional: preserve each deficit, do not only
    # check a clean structure whose hinge penalties are identically zero.
    coordinates = (torch.randn((1, count, 3), generator=generator) * 2 + 100).to(
        dtype=dtype, device=device
    )
    return coordinates, topology, fixed


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('chains,residues', [(1, 30), (3, 151)])
def test_guard_accepts_exact_global_translation_at_native_chain_size(dtype, chains, residues):
    before, topology, _ = geometry_fixture(chains, residues, dtype)
    original = before.clone()
    after = before + before.new_tensor([0.125, 0., 0.])
    candidate_copy = after.clone()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        report = scaffold_geometry_guard(before, topology, ScaffoldCoreGuidanceConfig())(after)
    assert report['accepted'], report
    assert all(check['numerical_tolerance_angstrom'] == 1e-6 for check in report['checks'])
    assert all(
        check['maximum_increase_angstrom'] is None
        or check['maximum_increase_angstrom'] < 1e-10
        for check in report['checks']
    )
    assert torch.equal(before, original)
    assert torch.equal(after, candidate_copy)
    assert before.dtype == after.dtype == dtype
    assert before.device == after.device


@pytest.mark.parametrize('distance', [3.2, 3.1])
def test_guard_still_rejects_real_overlap_increase_above_unchanged_tolerance(distance):
    # Mark both sites generated for this pairwise hard-guard test.
    features = {
        'atom_to_token_map': torch.arange(4),
        'asym_id': torch.tensor([0, 0, 1, 1]),
        'residue_index': torch.tensor([0, 1, 0, 1]),
        'is_ca': torch.ones(4, dtype=torch.bool),
        'is_protein': torch.ones(4, dtype=torch.bool),
    }
    topology = build_scaffold_core_topology(features, torch.zeros(4, dtype=torch.bool))
    coordinates = torch.tensor(
        [[[0., 0., 0.], [0., 3.8, 0.], [distance, 0., 0.], [distance, 3.8, 0.]]],
        dtype=torch.float64,
    )
    candidate = coordinates.clone()
    candidate[0, 2:, 0] -= 5e-6
    report = scaffold_geometry_guard(coordinates, topology, ScaffoldCoreGuidanceConfig())(candidate)
    assert not report['accepted']
    check = next(c for c in report['checks'] if c['rule'] == 'ca_overlap_regression')
    assert check['maximum_increase_angstrom'] == pytest.approx(5e-6, abs=1e-12)
    assert check['numerical_tolerance_angstrom'] == 1e-6
    assert not check['passed']


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_guard_nonfinite_coordinates_fail_closed(invalid):
    before, topology, _ = geometry_fixture(chains=1, residues=30)
    damaged = before.clone()
    damaged[0, 0, 0] = invalid
    guard = scaffold_geometry_guard(before, topology, ScaffoldCoreGuidanceConfig())
    assert guard(damaged) == {'accepted': False, 'reason': 'invalid_coordinates'}
    with pytest.raises(ValueError, match='finite coordinates'):
        scaffold_geometry_guard(damaged, topology, ScaffoldCoreGuidanceConfig())


def run_rigid_transaction(*, guard_factory=scaffold_geometry_guard, device="cpu"):
    before, topology, fixed = geometry_fixture(device=device)
    guard = guard_factory(before, topology, ScaffoldCoreGuidanceConfig())
    proposals = []
    synchronized = []

    def restore(value, target, mask):
        result = value.clone()
        result[:, mask] = target[:, mask]
        return result

    projector = UnifiedJointProjector(
        project_symmetry=lambda value: value,
        restore_constraints=restore,
        validate_closure=lambda value, label: None,
    )

    def proposal_hook(coordinates, progress):
        proposal = propose_bounded_se3_step(
            torch.eye(3, device=device), torch.zeros(3, device=device),
            lambda rotation, translation: (translation[0] - 0.125).square(),
            maximum_step_translation=0.125,
            maximum_step_rotation_degrees=0.,
            maximum_total_translation=1.,
            maximum_total_rotation_degrees=0.,
            line_search_scales=(1.,),
            candidate_validator=lambda rotation, translation: guard(coordinates + translation),
        )
        proposals.append(proposal)
        translated = coordinates + proposal.translation
        return ConstraintProposalResult(
            target=translated, coordinates=translated, applied=proposal.accepted
        )

    runtime = MosaicConstraintRuntime(
        projector=projector, fixed_target=before, fixed_mask=fixed,
        proposal_source='scaffold_boundary', proposal_hook=proposal_hook,
        synchronize_conditioning=lambda target: synchronized.append(target.clone()),
    )
    runtime.initialize_state(before)
    after = runtime.process_model_prediction(before, step_num=0, total_steps=1)
    return runtime, before, after, synchronized, proposals[0]


def test_native_runtime_commits_whole_rigid_proposal_and_preserves_coordinate_dtype():
    runtime, before, after, synchronized, proposal = run_rigid_transaction()
    assert proposal.accepted
    assert runtime.diagnostics()['phase_counts']['proposal_applied'] == 1
    assert runtime.conditioning_refresh_count == 1
    assert len(synchronized) == 1
    expected = before + before.new_tensor([0.125, 0., 0.])
    assert torch.equal(after, expected)
    assert torch.equal(runtime.fixed_target, expected)
    assert torch.equal(synchronized[0], expected)
    assert after.dtype == before.dtype and after.device == before.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_guard_retains_device_under_autocast():
    before, topology, _ = geometry_fixture(device='cuda')
    after = before + before.new_tensor([0.125, 0., 0.])
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        report = scaffold_geometry_guard(before, topology, ScaffoldCoreGuidanceConfig())(after)
    assert report['accepted'], report
    assert before.device.type == after.device.type == 'cuda'
    assert before.dtype == after.dtype == torch.float32
