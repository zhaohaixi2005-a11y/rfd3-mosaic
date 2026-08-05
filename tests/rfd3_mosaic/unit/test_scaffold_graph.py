import unittest
from pathlib import Path

from rfd3_mosaic.compile import (
    expand_symmetry_instances,
    load_interface_seed_config,
)
from rfd3_mosaic.schema import CompiledInstanceSet, ScaffoldLinkInstance
from rfd3_mosaic.topology import compile_scaffold_graph


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LHD101_CONFIG = (
    REPOSITORY_ROOT
    / "configs/rfd3_mosaic/single_interface/lhd101_c3.yaml"
)


def make_instances() -> CompiledInstanceSet:
    spec = load_interface_seed_config(LHD101_CONFIG)
    return expand_symmetry_instances(spec)


class ScaffoldGraphTestCase(unittest.TestCase):
    def test_lhd101_link_expands_once_per_c3_copy(self) -> None:
        instances = make_instances()

        self.assertEqual(len(instances.scaffold_links), 3)
        self.assertEqual(
            set(instances.scaffold_links),
            {
                "protomer@primary_orbit[0]",
                "protomer@primary_orbit[1]",
                "protomer@primary_orbit[2]",
            },
        )

    def test_expanded_link_preserves_direction_and_copy(self) -> None:
        link = make_instances().scaffold_links[
            "protomer@primary_orbit[2]"
        ]

        self.assertEqual(
            link.from_fragment_instance_id,
            "right@primary_orbit[2]",
        )
        self.assertEqual(
            link.to_fragment_instance_id,
            "left@primary_orbit[0]",
        )
        self.assertEqual(link.copy_index, 2)

    def test_graph_compiles_lhd101_topology(self) -> None:
        graph = compile_scaffold_graph(make_instances())

        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(len(graph.links), 3)
        self.assertEqual(
            graph.tied_length_ranges["protomer_length"].minimum,
            70,
        )
        self.assertEqual(
            graph.tied_length_ranges["protomer_length"].maximum,
            100,
        )

    def test_native_generated_links_use_the_same_graph_compiler(self) -> None:
        instances = make_instances()
        native = instances.model_copy(
            update={
                "scaffold_links": {},
                "generated_segments": dict(instances.scaffold_links),
            }
        )

        graph = compile_scaffold_graph(native)

        self.assertEqual(set(graph.links), set(instances.scaffold_links))
        self.assertEqual(
            graph.tied_length_ranges["protomer_length"].minimum,
            70,
        )

    def test_incoming_and_outgoing_queries_are_directed(self) -> None:
        graph = compile_scaffold_graph(make_instances())

        self.assertEqual(
            graph.outgoing_links("right@primary_orbit[1]"),
            ("protomer@primary_orbit[1]",),
        )
        self.assertEqual(
            graph.incoming_links("left@primary_orbit[2]"),
            ("protomer@primary_orbit[1]",),
        )
        self.assertEqual(
            graph.outgoing_links("left@primary_orbit[1]"),
            (),
        )

    def test_conflicting_tied_lengths_are_rejected(self) -> None:
        instances = make_instances()
        links = dict(instances.scaffold_links)
        original = links["protomer@primary_orbit[1]"]
        links[original.id] = original.model_copy(
            update={"minimum_length": 120, "maximum_length": 140}
        )
        conflicting = instances.model_copy(
            update={"scaffold_links": links}
        )

        with self.assertRaises(ValueError):
            compile_scaffold_graph(conflicting)

    def test_continuous_cycle_is_rejected(self) -> None:
        instances = make_instances()
        links = {
            "forward": ScaffoldLinkInstance(
                id="forward",
                source_id="protomer",
                from_fragment_instance_id="left@primary_orbit[0]",
                from_terminus="C",
                to_fragment_instance_id="right@primary_orbit[0]",
                to_terminus="N",
                minimum_length=1,
                maximum_length=5,
                copy_index=0,
            ),
            "reverse": ScaffoldLinkInstance(
                id="reverse",
                source_id="protomer",
                from_fragment_instance_id="right@primary_orbit[0]",
                from_terminus="C",
                to_fragment_instance_id="left@primary_orbit[0]",
                to_terminus="N",
                minimum_length=1,
                maximum_length=5,
                copy_index=0,
            ),
        }
        cyclic = instances.model_copy(update={"scaffold_links": links})

        with self.assertRaises(ValueError):
            compile_scaffold_graph(cyclic)


if __name__ == "__main__":
    unittest.main()
