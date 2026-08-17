import unittest
from types import SimpleNamespace

from rfd3_mosaic.schema import ScaffoldLinkInstance, Terminus
from rfd3_mosaic.topology import (
    analyze_interleaved_interface_seed_topology,
)


def _edge(edge_id, left_port, right_port, copy_index):
    return SimpleNamespace(
        id=edge_id,
        source_id="supplied_interface",
        left_port_instance_id=left_port,
        right_port_instance_id=right_port,
        required=True,
        satisfaction_stage="input",
        source_copy_index=copy_index,
        target_copy_index=copy_index,
    )


def _link(link_id, source, target, copy_index, target_copy_index):
    return ScaffoldLinkInstance(
        id=link_id,
        source_id="generated_unit",
        from_fragment_instance_id=source,
        from_terminus=Terminus.C,
        to_fragment_instance_id=target,
        to_terminus=Terminus.N,
        minimum_length=20,
        maximum_length=40,
        chain_break=False,
        orbit_id="motif_orbit",
        copy_index=copy_index,
        target_copy_index=target_copy_index,
    )


def _instances(
    *,
    cross_pair=True,
    omit_last_unit=False,
    multi_fragment_sides=False,
    internal_side_links=False,
):
    fragments = {
        fragment_id: SimpleNamespace(id=fragment_id)
        for fragment_id in ("A1", "B1", "A2", "B2", "A3", "B3")
    }
    ports = {}
    interfaces = {}
    for copy_index in range(1, 4):
        left_port = f"pair_{copy_index}_a"
        right_port = f"pair_{copy_index}_b"
        ports[left_port] = SimpleNamespace(
            fragment_instance_ids=(
                (f"A{copy_index}", f"C{copy_index}", f"E{copy_index}")
                if multi_fragment_sides
                else (f"A{copy_index}",)
            )
        )
        ports[right_port] = SimpleNamespace(
            fragment_instance_ids=(
                (
                    f"B{copy_index}",
                    f"D{copy_index}",
                    f"F{copy_index}",
                    f"G{copy_index}",
                )
                if multi_fragment_sides
                else (f"B{copy_index}",)
            )
        )
        edge_id = f"pair_{copy_index}"
        interfaces[edge_id] = _edge(
            edge_id,
            left_port,
            right_port,
            copy_index - 1,
        )

    if cross_pair:
        links = {
            "unit_1": _link("unit_1", "B1", "A2", 0, 1),
            "unit_2": _link("unit_2", "B2", "A3", 1, 2),
            "unit_3": _link("unit_3", "B3", "A1", 2, 0),
        }
    else:
        links = {
            "unit_1": _link("unit_1", "A1", "B1", 0, 0),
            "unit_2": _link("unit_2", "A2", "B2", 1, 1),
            "unit_3": _link("unit_3", "A3", "B3", 2, 2),
        }
    if omit_last_unit:
        links.pop("unit_3")
    if internal_side_links:
        if not multi_fragment_sides:
            raise ValueError(
                "internal_side_links require multi_fragment_sides"
            )
        for copy_index in range(1, 4):
            links[f"left_internal_{copy_index}"] = _link(
                f"left_internal_{copy_index}",
                f"A{copy_index}",
                f"C{copy_index}",
                copy_index - 1,
                copy_index - 1,
            )
            links[f"right_internal_{copy_index}"] = _link(
                f"right_internal_{copy_index}",
                f"B{copy_index}",
                f"D{copy_index}",
                copy_index - 1,
                copy_index - 1,
            )

    return SimpleNamespace(
        fragments=fragments,
        ports=ports,
        interfaces=interfaces,
        scaffold_links={},
        generated_segments=links,
    )


def _multi_interface_units():
    interface_ids = ("A", "C", "D")
    ports = {}
    interfaces = {}
    fragments = {}
    for interface_id in interface_ids:
        left_fragment = f"{interface_id}1"
        right_fragment = f"{interface_id}2"
        fragments[left_fragment] = SimpleNamespace(id=left_fragment)
        fragments[right_fragment] = SimpleNamespace(id=right_fragment)
        left_port = f"{interface_id}_left"
        right_port = f"{interface_id}_right"
        ports[left_port] = SimpleNamespace(
            fragment_instance_ids=(left_fragment,)
        )
        ports[right_port] = SimpleNamespace(
            fragment_instance_ids=(right_fragment,)
        )
        interfaces[interface_id] = _edge(
            interface_id,
            left_port,
            right_port,
            0,
        )

    links = {
        "unit_1_a_c": _link("unit_1_a_c", "A1", "C1", 0, 0),
        "unit_1_c_d": _link("unit_1_c_d", "C1", "D1", 0, 0),
        "unit_2_a_c": _link("unit_2_a_c", "A2", "C2", 0, 0),
        "unit_2_c_d": _link("unit_2_c_d", "C2", "D2", 0, 0),
    }
    return SimpleNamespace(
        fragments=fragments,
        ports=ports,
        interfaces=interfaces,
        scaffold_links={},
        generated_segments=links,
    )


class InterleavedInterfaceSeedTopologyTestCase(unittest.TestCase):
    def test_cross_pair_units_form_closed_alternating_cycle(self) -> None:
        report = analyze_interleaved_interface_seed_topology(_instances())

        self.assertEqual(report.status, "valid_interface_unit_graph")
        self.assertTrue(report.is_valid_interface_unit_graph)
        self.assertTrue(report.is_closed_alternating_cycle)
        self.assertEqual(len(report.interface_pairs), 3)
        self.assertEqual(len(report.polymer_units), 3)
        self.assertEqual(len(report.alternating_components), 1)
        self.assertEqual(report.violations, ())
        self.assertEqual(
            {unit.interface_pair_ids for unit in report.polymer_units},
            {
                ("pair_1", "pair_2"),
                ("pair_2", "pair_3"),
                ("pair_1", "pair_3"),
            },
        )

    def test_interface_sides_may_contain_multiple_fixed_fragments(self) -> None:
        report = analyze_interleaved_interface_seed_topology(
            _instances(multi_fragment_sides=True)
        )

        self.assertEqual(report.status, "valid_interface_unit_graph")
        self.assertEqual(len(report.interface_sides), 6)
        self.assertEqual(
            report.interface_pairs[0].left_fragment_instance_ids,
            ("A1", "C1", "E1"),
        )
        self.assertEqual(
            report.interface_pairs[0].right_fragment_instance_ids,
            ("B1", "D1", "F1", "G1"),
        )
        self.assertEqual(report.violations, ())

    def test_ordered_links_within_one_multi_fragment_side_are_valid(
        self,
    ) -> None:
        report = analyze_interleaved_interface_seed_topology(
            _instances(
                multi_fragment_sides=True,
                internal_side_links=True,
            )
        )

        self.assertEqual(report.status, "valid_interface_unit_graph")
        self.assertEqual(len(report.polymer_units), 3)
        self.assertEqual(report.violations, ())

    def test_same_pair_scaffold_links_are_not_cross_pair_units(self) -> None:
        report = analyze_interleaved_interface_seed_topology(
            _instances(cross_pair=False)
        )

        self.assertEqual(report.status, "invalid_interface_unit_graph")
        self.assertFalse(report.is_closed_alternating_cycle)
        self.assertTrue(
            any(
                "one supplied interface seed" in item
                for item in report.violations
            )
        )

    def test_one_unit_may_combine_three_different_interfaces(self) -> None:
        report = analyze_interleaved_interface_seed_topology(
            _multi_interface_units()
        )

        self.assertEqual(report.status, "valid_interface_unit_graph")
        self.assertFalse(report.is_closed_alternating_cycle)
        self.assertEqual(len(report.interface_pairs), 3)
        self.assertEqual(len(report.polymer_units), 2)
        self.assertEqual(
            {unit.interface_pair_ids for unit in report.polymer_units},
            {("A", "C", "D")},
        )
        self.assertEqual(report.violations, ())


if __name__ == "__main__":
    unittest.main()
