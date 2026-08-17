import unittest

from rfd3_mosaic.topology.polymer_path_solver import (
    BinaryInterfaceSeed,
    InterfaceHyperedgeSeed,
    enumerate_directed_polymer_path_covers,
    enumerate_polymer_hyperedge_covers,
    enumerate_polymer_unit_path_covers,
)


def _seeds(count: int):
    return tuple(
        BinaryInterfaceSeed(
            seed_id=f"seed_{index}",
            left_side_id=f"seed_{index}:left",
            right_side_id=f"seed_{index}:right",
        )
        for index in range(count)
    )


class DirectedPolymerPathCoverTestCase(unittest.TestCase):
    def test_three_binary_interfaces_form_two_three_face_units(self) -> None:
        seeds = tuple(
            InterfaceHyperedgeSeed(
                seed_id=f"interface_{index}",
                side_ids=(
                    f"interface_{index}:left",
                    f"interface_{index}:right",
                ),
            )
            for index in range(3)
        )

        hypotheses = enumerate_polymer_unit_path_covers(
            seeds,
            minimum_faces_per_unit=3,
            maximum_faces_per_unit=3,
        )

        self.assertTrue(hypotheses)
        expected_sides = {
            side_id for seed in seeds for side_id in seed.side_ids
        }
        owner = {
            side_id: seed.seed_id
            for seed in seeds
            for side_id in seed.side_ids
        }
        for hypothesis in hypotheses:
            self.assertEqual(hypothesis.unit_count, 2)
            self.assertEqual(len(hypothesis.ordered_paths), 2)
            self.assertEqual(len(hypothesis.ordered_links), 4)
            flattened = [
                side_id
                for path in hypothesis.ordered_paths
                for side_id in path
            ]
            self.assertEqual(set(flattened), expected_sides)
            self.assertEqual(len(flattened), len(set(flattened)))
            for path in hypothesis.ordered_paths:
                self.assertEqual(len(path), 3)
                self.assertEqual(len({owner[side] for side in path}), 3)

    def test_multi_face_path_cover_is_input_order_independent(self) -> None:
        seeds = tuple(
            InterfaceHyperedgeSeed(
                seed_id=f"interface_{index}",
                side_ids=(f"{index}:a", f"{index}:b"),
            )
            for index in range(4)
        )

        forward = enumerate_polymer_unit_path_covers(
            seeds,
            minimum_faces_per_unit=4,
            maximum_faces_per_unit=4,
        )
        reverse = enumerate_polymer_unit_path_covers(
            reversed(seeds),
            minimum_faces_per_unit=4,
            maximum_faces_per_unit=4,
        )

        self.assertEqual(forward, reverse)

    def test_multi_face_path_cover_rejects_impossible_valency(self) -> None:
        seeds = (
            InterfaceHyperedgeSeed("alpha", ("a", "b", "c")),
            InterfaceHyperedgeSeed("beta", ("d", "e")),
        )
        with self.assertRaisesRegex(ValueError, "No polymer-unit count"):
            enumerate_polymer_unit_path_covers(
                seeds,
                minimum_faces_per_unit=3,
                maximum_faces_per_unit=3,
            )

    def test_two_three_participant_interfaces_have_complete_matchings(
        self,
    ) -> None:
        seeds = (
            InterfaceHyperedgeSeed(
                "alpha",
                ("alpha:a", "alpha:b", "alpha:c"),
            ),
            InterfaceHyperedgeSeed(
                "beta",
                ("beta:d", "beta:e", "beta:f"),
            ),
        )

        hypotheses = enumerate_polymer_hyperedge_covers(seeds)

        # Every alpha side must pair with one beta side: 3! bijections.
        self.assertEqual(len(hypotheses), 6)
        expected_sides = {
            side_id for seed in seeds for side_id in seed.side_ids
        }
        for hypothesis in hypotheses:
            endpoints = [
                side_id
                for link in hypothesis.ordered_links
                for side_id in (
                    link.source_side_id,
                    link.target_side_id,
                )
            ]
            self.assertEqual(set(endpoints), expected_sides)
            self.assertEqual(len(endpoints), len(set(endpoints)))
            self.assertTrue(
                all(
                    link.source_side_id.split(":", 1)[0]
                    != link.target_side_id.split(":", 1)[0]
                    for link in hypothesis.ordered_links
                )
            )

    def test_hyperedge_cover_rejects_odd_total_participant_count(
        self,
    ) -> None:
        seeds = (
            InterfaceHyperedgeSeed("alpha", ("a", "b", "c")),
            InterfaceHyperedgeSeed("beta", ("d", "e")),
        )
        with self.assertRaisesRegex(ValueError, "even total number"):
            enumerate_polymer_hyperedge_covers(seeds)

    def test_two_seeds_have_two_rotation_reversal_unique_cycles(self) -> None:
        hypotheses = enumerate_directed_polymer_path_covers(_seeds(2))

        self.assertEqual(len(hypotheses), 2)
        self.assertEqual(
            len({item.canonical_key for item in hypotheses}),
            2,
        )
        self.assertTrue(all(not item.executable for item in hypotheses))
        self.assertTrue(
            all(
                item.evidence_scope == "disjoint_binary_seed_topology_only"
                for item in hypotheses
            )
        )

    def test_three_seed_cycles_are_complete_cross_seed_covers(self) -> None:
        seeds = _seeds(3)
        hypotheses = enumerate_directed_polymer_path_covers(seeds)
        side_owner = {
            side_id: seed.seed_id
            for seed in seeds
            for side_id in (seed.left_side_id, seed.right_side_id)
        }
        all_sides = set(side_owner)

        self.assertEqual(len(hypotheses), 8)
        for hypothesis in hypotheses:
            self.assertEqual(len(hypothesis.ordered_links), 3)
            endpoints = [
                side_id
                for link in hypothesis.ordered_links
                for side_id in (link.source_side_id, link.target_side_id)
            ]
            self.assertEqual(set(endpoints), all_sides)
            self.assertEqual(len(endpoints), len(set(endpoints)))
            self.assertTrue(
                all(
                    side_owner[link.source_side_id]
                    != side_owner[link.target_side_id]
                    for link in hypothesis.ordered_links
                )
            )

    def test_enumeration_is_deterministic_and_input_order_independent(
        self,
    ) -> None:
        seeds = _seeds(4)
        forward = enumerate_directed_polymer_path_covers(seeds)
        reverse = enumerate_directed_polymer_path_covers(reversed(seeds))

        self.assertEqual(forward, reverse)
        self.assertEqual(
            tuple(item.canonical_key for item in forward),
            tuple(sorted(item.canonical_key for item in forward)),
        )

    def test_rotation_and_reversal_are_collapsed(self) -> None:
        hypotheses = enumerate_directed_polymer_path_covers(_seeds(3))

        for hypothesis in hypotheses:
            key = hypothesis.canonical_key
            rotated = key[1:] + key[:1]
            reversed_key = tuple(
                (seed_id, exit_side, entry_side)
                for seed_id, entry_side, exit_side in reversed(key)
            )
            all_keys = {item.canonical_key for item in hypotheses}
            self.assertNotIn(rotated, all_keys - {key})
            self.assertNotIn(reversed_key, all_keys - {key})

    def test_candidate_limit_fails_instead_of_returning_partial_evidence(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "max_candidates=7"):
            enumerate_directed_polymer_path_covers(
                _seeds(3),
                max_candidates=7,
            )

    def test_fewer_than_two_seeds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            enumerate_directed_polymer_path_covers(_seeds(1))

    def test_overlapping_side_ids_fail_closed(self) -> None:
        seeds = (
            BinaryInterfaceSeed("alpha", "shared", "alpha:right"),
            BinaryInterfaceSeed("beta", "shared", "beta:right"),
        )

        with self.assertRaisesRegex(ValueError, "overlapping side IDs"):
            enumerate_directed_polymer_path_covers(seeds)

    def test_duplicate_seed_ids_fail_closed(self) -> None:
        seeds = (
            BinaryInterfaceSeed("same", "a:left", "a:right"),
            BinaryInterfaceSeed("same", "b:left", "b:right"),
        )

        with self.assertRaisesRegex(ValueError, "seed IDs must be unique"):
            enumerate_directed_polymer_path_covers(seeds)

    def test_max_candidates_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            enumerate_directed_polymer_path_covers(
                _seeds(2),
                max_candidates=0,
            )


if __name__ == "__main__":
    unittest.main()
