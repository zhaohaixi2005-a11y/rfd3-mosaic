import math
import unittest

import torch
from rfd3.inference.symmetry.local_neighbourhood import (
    build_local_symmetry_neighbourhood,
    crop_features_to_local_neighbourhood,
    expand_local_prediction_to_full_orbit,
    expand_local_token_prediction_to_full_orbit,
    select_local_transform_ids,
)
from rfd3.inference.symmetry.symmetry_utils import (
    build_symmetry_orbit_layout,
    symmetry_orbit_residual,
    symmetry_orbit_tolerance,
)
from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry


class _ArgmaxSequenceHead:
    @staticmethod
    def decode(logits):
        return logits.argmax(dim=-1)


class _LocalRecordingDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.atom_counts = []
        self.sequence_head = _ArgmaxSequenceHead()

    def forward(self, X_noisy_L, f, **_):
        self.calls += 1
        self.atom_counts.append(X_noisy_L.shape[-2])
        token_count = int(f["atom_to_token_map"].max().item()) + 1
        logits = torch.zeros(
            (X_noisy_L.shape[0], token_count, 4),
            dtype=X_noisy_L.dtype,
            device=X_noisy_L.device,
        )
        logits[..., 2] = 1.0
        return {
            "X_L": X_noisy_L * 0.5,
            "sequence_logits_I": logits,
        }


class LocalSymmetryNeighbourhoodTestCase(unittest.TestCase):
    @staticmethod
    def _cyclic_features(order: int, atoms_per_copy: int = 2) -> dict:
        transforms = {}
        for transform_id in range(order):
            angle = 2.0 * math.pi * transform_id / order
            transforms[str(transform_id)] = (
                torch.tensor(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float64,
                ),
                torch.zeros(3, dtype=torch.float64),
            )
        return {
            "sym_entity_id": torch.zeros(
                order * atoms_per_copy,
                dtype=torch.long,
            ),
            "sym_transform_id": torch.repeat_interleave(
                torch.arange(order),
                atoms_per_copy,
            ),
            "is_sym_asu": torch.tensor(
                [True] * atoms_per_copy
                + [False] * ((order - 1) * atoms_per_copy)
            ),
            "sym_orbit_slot": torch.arange(
                atoms_per_copy,
                dtype=torch.long,
            ).repeat(order),
            "sym_orbit_slot_verified": torch.tensor(True),
            "sym_transform": transforms,
        }

    def test_c200_view_contains_only_master_and_two_neighbours(self) -> None:
        self.assertEqual(
            select_local_transform_ids("C200", neighbour_radius=1),
            (0, 199, 1),
        )

    def test_cyclic_view_size_does_not_grow_with_order(self) -> None:
        for symmetry_id in ("C12", "C20", "C200"):
            with self.subTest(symmetry_id=symmetry_id):
                self.assertEqual(
                    len(
                        select_local_transform_ids(
                            symmetry_id,
                            neighbour_radius=2,
                        )
                    ),
                    5,
                )

    def test_dihedral_view_includes_both_local_cosets(self) -> None:
        self.assertEqual(
            select_local_transform_ids(
                "D100",
                neighbour_radius=1,
            ),
            (0, 99, 1, 100, 199, 101),
        )

    def test_negative_radius_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            select_local_transform_ids("C12", neighbour_radius=-1)

    def test_local_prediction_rebuilds_the_complete_c12_orbit(self) -> None:
        features = self._cyclic_features(12)
        full = torch.zeros((1, 24, 3), dtype=torch.float64)
        layout = build_symmetry_orbit_layout(features, like=full)
        neighbourhood = build_local_symmetry_neighbourhood(
            features,
            "C12",
            like=full,
            neighbour_radius=1,
            layout=layout,
        )
        self.assertEqual(neighbourhood.copy_count, 3)
        self.assertEqual(len(neighbourhood.atom_indices), 6)

        canonical = torch.tensor(
            [[[4.0, 1.0, -2.0], [5.0, -1.0, 3.0]]],
            dtype=torch.float64,
        )
        local = torch.empty((1, 6, 3), dtype=torch.float64)
        for transform_id in neighbourhood.selected_transform_ids:
            global_indices = torch.nonzero(
                features["sym_transform_id"] == transform_id,
                as_tuple=False,
            ).flatten()
            local_indices = neighbourhood.global_to_local_atom[
                global_indices
            ]
            rotation, translation = layout.sym_transforms[transform_id]
            local[:, local_indices, :] = (
                torch.matmul(canonical, rotation.transpose(-1, -2))
                + translation
            )

        # Deliberately poison every omitted copy.  These values must not
        # influence the local canonical average.
        full.fill_(1000.0)
        rebuilt = expand_local_prediction_to_full_orbit(
            local,
            full,
            neighbourhood,
            layout=layout,
        )
        rms, maximum = symmetry_orbit_residual(
            rebuilt,
            features,
            layout=layout,
        )
        self.assertTrue(torch.all(rms < 1e-10))
        self.assertTrue(torch.all(maximum < 1e-10))
        torch.testing.assert_close(
            rebuilt[:, :2, :],
            canonical,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_atom_token_and_pair_features_are_cropped_consistently(self) -> None:
        features = self._cyclic_features(12, atoms_per_copy=4)
        full = torch.zeros((1, 48, 3), dtype=torch.float64)
        neighbourhood = build_local_symmetry_neighbourhood(
            features,
            "C12",
            like=full,
            neighbour_radius=1,
        )
        features.update(
            {
                "atom_to_token_map": torch.repeat_interleave(
                    torch.arange(24),
                    2,
                ),
                "atom_feature": torch.arange(48),
                "token_feature": torch.arange(24),
                "token_bonds": torch.arange(24 * 24).reshape(24, 24),
                "motif_constraint_group_membership": torch.ones(
                    (1, 48),
                    dtype=torch.bool,
                ),
                "motif_constraint_target_coordinates": torch.zeros(
                    (1, 1, 48, 3),
                ),
            }
        )

        view = crop_features_to_local_neighbourhood(
            features,
            neighbourhood,
        )

        self.assertEqual(
            view.token_indices.tolist(),
            [0, 1, 22, 23, 2, 3],
        )
        self.assertEqual(
            view.features["atom_to_token_map"].tolist(),
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )
        self.assertEqual(view.features["atom_feature"].shape, (12,))
        self.assertEqual(view.features["token_feature"].shape, (6,))
        self.assertEqual(view.features["token_bonds"].shape, (6, 6))
        self.assertEqual(
            view.features["motif_constraint_group_membership"].shape,
            (1, 12),
        )
        self.assertEqual(
            view.features["motif_constraint_target_coordinates"].shape,
            (1, 1, 12, 3),
        )

    def test_feature_crop_rejects_a_split_atomized_token(self) -> None:
        features = self._cyclic_features(3, atoms_per_copy=2)
        full = torch.zeros((1, 6, 3), dtype=torch.float64)
        neighbourhood = build_local_symmetry_neighbourhood(
            features,
            "C3",
            like=full,
            neighbour_radius=0,
        )
        # Token zero deliberately spans the selected master and omitted copy.
        features["atom_to_token_map"] = torch.tensor([0, 1, 0, 2, 3, 4])

        with self.assertRaisesRegex(ValueError, "splits"):
            crop_features_to_local_neighbourhood(
                features,
                neighbourhood,
            )

    def test_local_sequence_logits_expand_over_all_c12_tokens(self) -> None:
        features = self._cyclic_features(12, atoms_per_copy=4)
        features["atom_to_token_map"] = torch.repeat_interleave(
            torch.arange(24),
            2,
        )
        full = torch.zeros((1, 48, 3), dtype=torch.float64)
        neighbourhood = build_local_symmetry_neighbourhood(
            features,
            "C12",
            like=full,
            neighbour_radius=1,
        )
        view = crop_features_to_local_neighbourhood(
            features,
            neighbourhood,
        )
        local_logits = torch.empty((1, 6, 3))
        local_logits[:, 0::2, :] = torch.tensor([1.0, 2.0, 3.0])
        local_logits[:, 1::2, :] = torch.tensor([4.0, 5.0, 6.0])

        expanded = expand_local_token_prediction_to_full_orbit(
            local_logits,
            features,
            view,
        )

        self.assertEqual(expanded.shape, (1, 24, 3))
        for copy_index in range(12):
            torch.testing.assert_close(
                expanded[:, 2 * copy_index, :],
                torch.tensor([[1.0, 2.0, 3.0]]),
            )
            torch.testing.assert_close(
                expanded[:, 2 * copy_index + 1, :],
                torch.tensor([[4.0, 5.0, 6.0]]),
            )

    def test_sampler_denoises_local_c12_and_returns_complete_orbit(
        self,
    ) -> None:
        torch.manual_seed(0)
        features = self._cyclic_features(12)
        atom_count = 24
        features.update(
            {
                "symmetry_id": "C12",
                "atom_to_token_map": torch.arange(atom_count),
                "ref_element": torch.zeros(atom_count),
                "is_motif_atom_with_fixed_coord": torch.zeros(
                    atom_count,
                    dtype=torch.bool,
                ),
                "is_motif_token_with_fully_fixed_coord": torch.zeros(
                    atom_count,
                    dtype=torch.bool,
                ),
            }
        )
        coordinates = torch.zeros((1, atom_count, 3))
        sampler = SampleDiffusionWithSymmetry(
            gamma_0=0.6,
            num_timesteps=3,
            symmetry_execution_backend="local_neighbourhood",
            symmetry_neighbour_radius=1,
            preserve_fixed_motif_during_symmetry=True,
            symmetry_state_mode="orbit_average",
            symmetry_noise_mode="coupled",
        )
        context = sampler.prepare_local_network_view(
            features,
            coordinates,
        )
        self.assertIsNotNone(context)
        assert context is not None
        diffusion = _LocalRecordingDiffusion()

        with torch.no_grad():
            result = sampler.sample_diffusion_like_af3(
                f=features,
                network_f=context.feature_view.features,
                local_symmetry_context=context,
                diffusion_module=diffusion,
                diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=coordinates,
                initializer_outputs={
                    "chunked_pairwise_embedder": object(),
                },
                ref_initializer_outputs=None,
                f_ref=None,
            )

        self.assertEqual(diffusion.calls, 2)
        self.assertEqual(diffusion.atom_counts, [6, 6])
        self.assertEqual(result["X_L"].shape, (1, 24, 3))
        self.assertEqual(result["sequence_logits_I"].shape, (1, 24, 4))
        self.assertTrue(torch.all(result["sequence_indices_I"] == 2))
        rms, maximum = symmetry_orbit_residual(
            result["X_L"],
            features,
            layout=context.layout,
        )
        tolerance, _ = symmetry_orbit_tolerance(
            result["X_L"],
            configured_tolerance=sampler.symmetry_orbit_max_error,
        )
        self.assertTrue(torch.all(rms <= tolerance))
        self.assertTrue(torch.all(maximum <= tolerance))


if __name__ == "__main__":
    unittest.main()
