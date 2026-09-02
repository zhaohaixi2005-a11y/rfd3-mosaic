import unittest

import torch
from rfd3.model.layers.block_utils import (
    create_attention_indices,
    get_sparse_attention_indices_with_inter_chain,
)


class InterChainAttentionTestCase(unittest.TestCase):
    @staticmethod
    def _five_chain_case(atoms_per_chain: int = 10):
        chain_id = torch.arange(5).repeat_interleave(atoms_per_chain)
        atom_count = len(chain_id)
        coordinates = torch.arange(
            atom_count,
            dtype=torch.float32,
        ).reshape(1, atom_count, 1)
        coordinates = torch.cat(
            [coordinates, torch.zeros((1, atom_count, 2))],
            dim=-1,
        )
        distances = torch.cdist(coordinates, coordinates)
        token_index = torch.arange(atom_count)
        base_mask = torch.ones(
            (atom_count, atom_count),
            dtype=torch.bool,
        )
        return chain_id, distances, token_index, base_mask

    def test_every_query_receives_only_inter_chain_reserved_keys(self):
        chain_id, distances, token_index, base_mask = (
            self._five_chain_case()
        )

        indices = get_sparse_attention_indices_with_inter_chain(
            token_index,
            distances,
            n_seq_neighbours=1,
            k_intra=4,
            k_inter=3,
            chain_id=chain_id,
            base_mask=base_mask,
        )

        reserved = indices[0, :, -3:]
        query_chains = chain_id.unsqueeze(-1)
        self.assertTrue(
            torch.all(chain_id[reserved] != query_chains)
        )

    def test_small_attention_budget_remains_valid_for_many_chains(self):
        chain_id, _, token_index, _ = self._five_chain_case(
            atoms_per_chain=2
        )
        atom_count = len(chain_id)
        features = {
            "atom_to_token_map": token_index,
            "asym_id": chain_id,
            "unindexing_pair_mask": torch.zeros(
                (atom_count, atom_count),
                dtype=torch.bool,
            ),
        }
        coordinates = torch.randn((1, atom_count, 3))

        indices = create_attention_indices(
            features,
            n_attn_keys=8,
            n_attn_seq_neighbours=1,
            X_L=coordinates,
        )

        self.assertEqual(tuple(indices.shape), (1, atom_count, 8))
        reserved = indices[0, :, -7:]
        self.assertTrue(
            torch.all(chain_id[reserved] != chain_id.unsqueeze(-1))
        )


if __name__ == "__main__":
    unittest.main()
