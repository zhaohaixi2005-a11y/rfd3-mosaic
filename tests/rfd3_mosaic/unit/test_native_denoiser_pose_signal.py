"""Native EDM fixed atoms contain no learned coordinate signal for pose fitting."""

from types import SimpleNamespace

import pytest
import torch
from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry
from rfd3.model.RFD3_diffusion_module import RFD3DiffusionModule


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_edm_fixed_atom_output_has_zero_network_pose_signal(dtype):
    module = SimpleNamespace(f_pred="edm", sigma_data=16)
    coordinates = torch.arange(90, dtype=dtype).reshape(1, 30, 3) / 4 - 10
    fixed = torch.arange(30) % 3 == 0
    time = torch.tensor([20.0], dtype=dtype)
    # The native forward method zeros each fixed atom's diffusion time.
    atom_time = time.unsqueeze(-1).expand(-1, 30) * (~fixed).float().unsqueeze(0)
    network_update = torch.arange(90, dtype=dtype).reshape(1, 30, 3).requires_grad_()
    first = RFD3DiffusionModule.scale_positions_out(
        module, network_update, coordinates, atom_time
    )
    second = RFD3DiffusionModule.scale_positions_out(
        module, -123 * network_update, coordinates, atom_time
    )
    sensitivity = torch.autograd.grad(first[:, fixed].sum(), network_update)[0]
    assert torch.equal(first[:, fixed], coordinates[:, fixed])
    assert torch.equal(first[:, fixed], second[:, fixed])
    assert torch.count_nonzero(sensitivity) == 0
    assert torch.any(first[:, ~fixed] != second[:, ~fixed])


def test_native_sampler_rejects_mobile_denoiser_fit_with_migration_message():
    with pytest.raises(ValueError, match="no learned pose signal.*scaffold_objectives"):
        SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
            preserve_fixed_motif_during_symmetry=True,
            enable_orbit_rigid_motif_mobility=True,
            motif_mobility_proposal_source="denoiser",
        )


def test_native_sampler_keeps_disabled_mobility_and_objective_mobility_valid():
    locked = SampleDiffusionWithSymmetry(gamma_0=0.6)
    assert not locked.enable_orbit_rigid_motif_mobility
    assert locked.motif_mobility_proposal_source == "denoiser"
    mobile = SampleDiffusionWithSymmetry(
        gamma_0=0.6,
        symmetry_state_mode="orbit_average",
        symmetry_noise_mode="coupled",
        preserve_fixed_motif_during_symmetry=True,
        enable_orbit_rigid_motif_mobility=True,
        motif_mobility_proposal_source="scaffold_boundary",
    )
    assert mobile.enable_orbit_rigid_motif_mobility
