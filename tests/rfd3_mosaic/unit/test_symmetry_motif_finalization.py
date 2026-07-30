import math
import unittest

import torch

from rfd3.inference.symmetry.symmetry_utils import (
    apply_symmetry_to_xyz_atomwise,
    build_symmetry_orbit_layout,
    project_symmetry_orbit_average,
    symmetry_orbit_residual,
)
from rfd3.model.inference_sampler import (
    ConditionalDiffusionSampler,
    SampleDiffusionWithSymmetry,
)


class _FragmentBreakingSymmetrySampler(SampleDiffusionWithSymmetry):
    """Stand-in for symmetry projection that separates two motif fragments."""

    def apply_symmetry_to_X_L(self, X_L, f):
        self.symmetry_projection_calls = (
            getattr(self, "symmetry_projection_calls", 0) + 1
        )
        projected = X_L.clone()
        projected[:, :3] += torch.tensor([20.0, 0.0, 0.0])
        projected[:, 3:6] += torch.tensor([0.0, 20.0, 0.0])
        projected[:, 6:] += torch.tensor([0.0, 0.0, 5.0])
        return projected


class _AsymmetricFakeDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, X_noisy_L, **_):
        self.calls += 1
        prediction = X_noisy_L * 0.5
        prediction[:, 2, :] += torch.tensor(
            [3.0, -1.0, 0.5],
            dtype=prediction.dtype,
            device=prediction.device,
        )
        return {"X_L": prediction}


class SymmetryMotifFinalizationTestCase(unittest.TestCase):
    @staticmethod
    def _c3_features() -> dict:
        transforms = {}
        for transform_id, angle in enumerate(
            (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
        ):
            cosine = math.cos(angle)
            sine = math.sin(angle)
            transforms[str(transform_id)] = (
                torch.tensor(
                    [
                        [cosine, -sine, 0.0],
                        [sine, cosine, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                torch.zeros(3),
            )
        return {
            "sym_entity_id": torch.zeros(6, dtype=torch.long),
            "sym_transform_id": torch.tensor([0, 0, 1, 1, 2, 2]),
            "is_sym_asu": torch.tensor(
                [True, True, False, False, False, False]
            ),
            "sym_orbit_slot": torch.tensor([0, 1, 0, 1, 0, 1]),
            "sym_orbit_slot_verified": torch.tensor(True),
            "sym_transform": transforms,
            "motif_constraint_group_membership": torch.ones(
                (1, 6),
                dtype=torch.bool,
            ),
        }

    def test_exact_orbit_mode_requires_coupled_noise(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires both",
        ):
            SampleDiffusionWithSymmetry(
                gamma_0=0.6,
                symmetry_state_mode="orbit_average",
            )

    def test_default_sampler_rejects_silently_ignored_exact_flags(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "kind=symmetry"):
            ConditionalDiffusionSampler(
                kind="default",
                symmetry_state_mode="orbit_average",
                symmetry_noise_mode="coupled",
            )

    def test_diffusion_requires_at_least_two_timesteps(self) -> None:
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            num_timesteps=1,
        )

        with self.assertRaisesRegex(ValueError, "at least two"):
            sampler._construct_inference_noise_schedule(
                device=torch.device("cpu"),
            )

    def test_partial_diffusion_cannot_reduce_schedule_to_one_step(
        self,
    ) -> None:
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            num_timesteps=10,
        )
        complete = sampler._construct_inference_noise_schedule(
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(ValueError, "after partial_t"):
            sampler._construct_inference_noise_schedule(
                device=torch.device("cpu"),
                partial_t=complete[-1:],
            )

    def test_exact_orbit_mode_rejects_arbitrary_realignment(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "allow_realignment=True",
        ):
            SampleDiffusionWithSymmetry(
                gamma_0=0.6,
                allow_realignment=True,
                symmetry_state_mode="orbit_average",
                symmetry_noise_mode="coupled",
            )

    def test_orbit_rigid_mobility_requires_exact_state_mode(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires exact",
        ):
            SampleDiffusionWithSymmetry(
                gamma_0=0.6,
                enable_orbit_rigid_motif_mobility=True,
            )

    def test_static_fixed_target_is_projected_once_into_exact_c3(
        self,
    ) -> None:
        features = self._c3_features()
        canonical = torch.tensor(
            [[[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]]
        )
        target = apply_symmetry_to_xyz_atomwise(
            canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        )
        target[:, 2, 0] += 0.003
        fixed = torch.ones(6, dtype=torch.bool)
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )

        prepared = sampler._prepare_static_symmetry_target(
            target,
            fixed,
            features,
            diffusion_batch_size=1,
        )

        rms, maximum = symmetry_orbit_residual(prepared, features)
        self.assertTrue(torch.all(rms < 1e-6))
        self.assertTrue(torch.all(maximum < 1e-6))

    def test_orbit_closure_uses_only_a_high_noise_roundoff_floor(
        self,
    ) -> None:
        features = self._c3_features()
        canonical = torch.tensor(
            [[[2560.0, -1800.0, 900.0], [1700.0, 2400.0, -700.0]]]
        )
        exact_high_noise = apply_symmetry_to_xyz_atomwise(
            canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        )
        high_noise = exact_high_noise.clone()
        high_noise[:, 2, 0] += 0.002
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )
        sampler._exact_symmetry_orbit_layout = (
            build_symmetry_orbit_layout(
                features,
                like=high_noise,
            )
        )

        # At the initial EDM scale this error is below the unavoidable
        # float32 relative roundoff floor.
        sampler._assert_symmetry_orbit_closed(
            high_noise,
            features,
            label="high-noise test state",
        )

        low_noise = project_symmetry_orbit_average(
            exact_high_noise / 100.0,
            features,
            layout=sampler._exact_symmetry_orbit_layout,
        )
        low_noise[:, 2, 0] += 0.002
        # The same absolute error must not be hidden at molecular scale.
        with self.assertRaisesRegex(ValueError, "left the runtime"):
            sampler._assert_symmetry_orbit_closed(
                low_noise,
                features,
                label="low-noise test state",
            )

    def test_incompatible_static_fixed_target_fails_before_sampling(
        self,
    ) -> None:
        features = self._c3_features()
        canonical = torch.tensor(
            [[[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]]
        )
        target = apply_symmetry_to_xyz_atomwise(
            canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        )
        target[:, 2, 0] += 0.2
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )

        with self.assertRaisesRegex(
            ValueError,
            "incompatible",
        ):
            sampler._prepare_static_symmetry_target(
                target,
                torch.ones(6, dtype=torch.bool),
                features,
                diffusion_batch_size=1,
            )

    def test_nonfinite_static_fixed_target_fails_before_sampling(
        self,
    ) -> None:
        features = self._c3_features()
        target = torch.zeros((1, 6, 3))
        target[0, 0, 0] = float("nan")
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            sampler._prepare_static_symmetry_target(
                target,
                torch.ones(6, dtype=torch.bool),
                features,
                diffusion_batch_size=1,
            )

    def test_exact_sampler_keeps_every_runtime_state_c3_closed(
        self,
    ) -> None:
        features = self._c3_features()
        canonical = torch.tensor(
            [[[5.0, 0.0, 0.0], [7.0, 1.0, 0.5]]]
        )
        coordinates = apply_symmetry_to_xyz_atomwise(
            canonical.repeat(1, 3, 1),
            features,
            partial_diffusion=True,
        )
        fixed = torch.tensor(
            [True, False, True, False, True, False]
        )
        features.update(
            {
                "is_motif_atom_with_fixed_coord": fixed,
                "motif_constraint_group_membership": fixed[None, :],
                "ref_element": torch.zeros(6, dtype=torch.long),
            }
        )
        module = _AsymmetricFakeDiffusion()
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            num_timesteps=3,
            preserve_fixed_motif_during_symmetry=True,
            require_motif_constraint_groups=True,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )

        with torch.no_grad():
            result = sampler.sample_diffusion_like_af3(
                f=features,
                diffusion_module=module,
                diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=coordinates,
                initializer_outputs={},
                ref_initializer_outputs=None,
                f_ref=None,
            )

        self.assertEqual(module.calls, 2)
        sampler._assert_symmetry_orbit_closed(
            result["X_L"],
            features,
            label="Final test diffusion state",
        )
        self.assertTrue(
            torch.allclose(
                result["X_L"][:, fixed, :],
                coordinates[:, fixed, :],
                atol=1e-5,
            )
        )
        for index, state in enumerate(result["X_denoised_L_traj"]):
            # Intermediate EDM predictions can remain at a coordinate scale
            # where float32 rotation round-trips cannot satisfy the final
            # molecular-scale 1e-5 A check. Exercise the same scale-aware
            # orbit-closure gate used by the production sampler at each step.
            sampler._assert_symmetry_orbit_closed(
                state,
                features,
                label=f"Denoised trajectory state {index}",
            )

    def test_compactness_moves_generated_tokens_but_not_fixed_motif(
        self,
    ) -> None:
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [14.0, 0.0, 0.0],
                    [15.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [21.0, 0.0, 0.0],
                ]
            ]
        )
        fixed_mask = torch.tensor(
            [True, True, False, False, True, True]
        )
        features = {
            "atom_to_token_map": torch.tensor([0, 0, 1, 1, 2, 2]),
            "asym_id": torch.tensor([0, 0, 0]),
        }
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            interface_seed_compactness_weight=0.1,
            interface_seed_compactness_end_frac=0.75,
            interface_seed_compactness_max_step=0.5,
        )

        guided = sampler._apply_interface_seed_compactness(
            coordinates,
            features,
            fixed_mask,
            step_num=0,
            num_steps=10,
        )

        self.assertTrue(
            torch.equal(guided[:, fixed_mask], coordinates[:, fixed_mask])
        )
        anchor_center = coordinates[:, fixed_mask].mean(dim=1)
        before = torch.linalg.vector_norm(
            coordinates[:, 2:4].mean(dim=1) - anchor_center,
            dim=-1,
        )
        after = torch.linalg.vector_norm(
            guided[:, 2:4].mean(dim=1) - anchor_center,
            dim=-1,
        )
        self.assertTrue(torch.all(after < before))
        # All atoms in one residue receive the same translation.
        self.assertTrue(
            torch.allclose(
                guided[:, 3] - guided[:, 2],
                coordinates[:, 3] - coordinates[:, 2],
            )
        )

    def test_compactness_is_disabled_after_end_fraction(self) -> None:
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]
        )
        fixed_mask = torch.tensor([True, False])
        features = {
            "atom_to_token_map": torch.tensor([0, 1]),
            "asym_id": torch.tensor([0, 0]),
        }
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            interface_seed_compactness_weight=0.1,
            interface_seed_compactness_end_frac=0.5,
        )

        guided = sampler._apply_interface_seed_compactness(
            coordinates,
            features,
            fixed_mask,
            step_num=5,
            num_steps=10,
        )

        self.assertTrue(torch.equal(guided, coordinates))

    def test_complete_cross_chain_motif_wins_over_symmetry_projection(
        self,
    ) -> None:
        """Finalization must restore the pair, not only each fragment."""

        reference = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [4.0, 0.0, 1.0],
                    [8.0, 8.0, 8.0],
                ]
            ]
        )
        prediction = reference.clone()
        fixed_mask = torch.tensor(
            [True, True, True, True, True, True, False]
        )
        sampler = _FragmentBreakingSymmetrySampler(
            gamma_0=0.6,
            allow_realignment=True,
            insert_motif_at_end=True,
        )

        torch.manual_seed(7)
        finalized = sampler._finalize_with_fixed_motif(
            prediction,
            reference,
            fixed_mask,
            {},
        )

        self.assertTrue(
            torch.allclose(
                finalized[:, fixed_mask],
                reference[:, fixed_mask],
                atol=1e-4,
            )
        )
        # In particular, the relative distance between atoms on opposite
        # fragments must remain the original 4 Å.
        distance = torch.linalg.norm(finalized[:, 0] - finalized[:, 3])
        self.assertAlmostEqual(float(distance), 4.0, places=4)

    def test_stepwise_projection_restores_complete_fixed_motif(
        self,
    ) -> None:
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [4.0, 0.0, 1.0],
                    [8.0, 8.0, 8.0],
                ]
            ]
        )
        fixed_mask = torch.tensor(
            [True, True, True, True, True, True, False]
        )
        sampler = _FragmentBreakingSymmetrySampler(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
        )

        projected = sampler._apply_symmetry_preserving_fixed_motif(
            coordinates,
            {},
            fixed_mask,
        )

        self.assertTrue(
            torch.equal(
                projected[:, fixed_mask],
                coordinates[:, fixed_mask],
            )
        )
        self.assertTrue(
            torch.equal(
                projected[:, ~fixed_mask],
                coordinates[:, ~fixed_mask]
                + torch.tensor([0.0, 0.0, 5.0]),
            )
        )

    def test_updated_state_is_projected_when_compactness_is_disabled(
        self,
    ) -> None:
        updated = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [8.0, 8.0, 8.0],
                ]
            ]
        )
        fixed_coordinates = updated.clone()
        fixed_mask = torch.tensor([True, True, False])
        sampler = _FragmentBreakingSymmetrySampler(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
            interface_seed_compactness_weight=0.0,
        )

        projected = sampler._project_stepwise_updated_coordinates(
            updated,
            {},
            fixed_mask,
            fixed_coordinates,
        )

        self.assertEqual(sampler.symmetry_projection_calls, 1)
        self.assertTrue(
            torch.equal(
                projected[:, fixed_mask],
                fixed_coordinates[:, fixed_mask],
            )
        )
        self.assertTrue(
            torch.equal(
                projected[:, ~fixed_mask],
                updated[:, ~fixed_mask]
                + torch.tensor([20.0, 0.0, 0.0]),
            )
        )

    def test_stepwise_mode_avoids_a_second_final_symmetry_projection(
        self,
    ) -> None:
        reference = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [4.0, 0.0, 1.0],
                    [8.0, 8.0, 8.0],
                ]
            ]
        )
        fixed_mask = torch.tensor(
            [True, True, True, True, True, True, False]
        )
        sampler = _FragmentBreakingSymmetrySampler(
            gamma_0=0.6,
            allow_realignment=True,
            insert_motif_at_end=True,
            preserve_fixed_motif_during_symmetry=True,
        )

        torch.manual_seed(7)
        finalized = sampler._finalize_with_fixed_motif(
            reference.clone(),
            reference,
            fixed_mask,
            {},
        )

        self.assertEqual(
            getattr(sampler, "symmetry_projection_calls", 0),
            0,
        )
        self.assertTrue(
            torch.allclose(
                finalized[:, fixed_mask],
                reference[:, fixed_mask],
                atol=1e-4,
            )
        )

    def test_overlapping_constraint_groups_merge_order_independently(
        self,
    ) -> None:
        coordinates = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0],
                ]
            ]
        )
        projected = coordinates + 10.0
        fixed_mask = torch.tensor([True, True, True, False])
        membership = torch.tensor(
            [
                [True, True, False, False],
                [False, True, True, False],
            ]
        )
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
        )

        restored = sampler._restore_motif_constraint_groups(
            projected,
            coordinates,
            fixed_mask,
            {"motif_constraint_group_membership": membership},
        )
        reversed_groups = sampler._restore_motif_constraint_groups(
            projected,
            coordinates,
            fixed_mask,
            {
                "motif_constraint_group_membership": torch.flip(
                    membership,
                    dims=(0,),
                )
            },
        )

        self.assertTrue(torch.equal(restored, reversed_groups))
        self.assertTrue(
            torch.equal(
                restored[:, fixed_mask],
                coordinates[:, fixed_mask],
            )
        )
        self.assertTrue(
            torch.equal(
                restored[:, ~fixed_mask],
                projected[:, ~fixed_mask],
            )
        )

    def test_conflicting_overlapping_constraint_groups_are_rejected(
        self,
    ) -> None:
        coordinates = torch.zeros((1, 3, 3))
        fixed_mask = torch.tensor([True, True, True])
        membership = torch.tensor(
            [
                [True, True, False],
                [False, True, True],
            ]
        )
        targets = coordinates[:, None, :, :].repeat(1, 2, 1, 1)
        targets[:, 1, 1, 0] = 1.0
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Overlapping motif constraint groups disagree",
        ):
            sampler._restore_motif_constraint_groups(
                coordinates,
                coordinates,
                fixed_mask,
                {
                    "motif_constraint_group_membership": membership,
                    "motif_constraint_target_coordinates": targets,
                },
            )

    def test_group_mode_requires_complete_fixed_atom_coverage(
        self,
    ) -> None:
        coordinates = torch.zeros((1, 3, 3))
        fixed_mask = torch.tensor([True, True, True])
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            preserve_fixed_motif_during_symmetry=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Every fixed motif atom",
        ):
            sampler._restore_motif_constraint_groups(
                coordinates,
                coordinates,
                fixed_mask,
                {
                    "motif_constraint_group_membership": torch.tensor(
                        [[True, True, False]]
                    )
                },
            )


if __name__ == "__main__":
    unittest.main()
