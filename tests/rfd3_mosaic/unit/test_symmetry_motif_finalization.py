import unittest

import torch

from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry


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


class SymmetryMotifFinalizationTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
