"""Topology analysis for supplied interface seeds and scaffold units."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rfd3_mosaic.topology.scaffold_graph import compiled_scaffold_links


@dataclass(frozen=True)
class InterfaceSeedSideRecord:
    """One physical protomer side of a supplied interface seed."""

    side_id: str
    edge_instance_id: str
    role: Literal["left", "right"]
    fragment_instance_ids: tuple[str, ...]
    copy_index: int


@dataclass(frozen=True)
class InterfaceSeedPairRecord:
    """One supplied non-covalent interface connecting two protomer sides."""

    edge_instance_id: str
    source_interface_id: str
    left_side_id: str
    right_side_id: str
    left_fragment_instance_ids: tuple[str, ...]
    right_fragment_instance_ids: tuple[str, ...]
    source_copy_index: int
    target_copy_index: int
    required: bool


@dataclass(frozen=True)
class PolymerUnitRecord:
    """One protein unit assembled from any number of interface-seed sides."""

    unit_id: str
    interface_side_ids: tuple[str, ...]
    interface_pair_ids: tuple[str, ...]
    fragment_instance_ids: tuple[str, ...]
    link_instance_ids: tuple[str, ...]
    source_link_ids: tuple[str, ...]


@dataclass(frozen=True)
class InterleavedInterfaceSeedTopology:
    """Compiled interface--unit incidence graph and its validation evidence."""

    status: str
    interface_sides: tuple[InterfaceSeedSideRecord, ...]
    interface_pairs: tuple[InterfaceSeedPairRecord, ...]
    polymer_units: tuple[PolymerUnitRecord, ...]
    alternating_components: tuple[tuple[str, ...], ...]
    violations: tuple[str, ...]

    @property
    def is_valid_interface_unit_graph(self) -> bool:
        return self.status == "valid_interface_unit_graph"

    @property
    def is_closed_alternating_cycle(self) -> bool:
        """Whether this valid graph is the old two-interfaces-per-unit case."""

        return self.is_valid_interface_unit_graph and all(
            len(unit.interface_side_ids) == 2 for unit in self.polymer_units
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "is_valid_interface_unit_graph": (
                self.is_valid_interface_unit_graph
            ),
            "is_closed_alternating_cycle": (
                self.is_closed_alternating_cycle
            ),
            "interface_side_count": len(self.interface_sides),
            "interface_pair_count": len(self.interface_pairs),
            "polymer_unit_count": len(self.polymer_units),
            "interface_sides": [
                asdict(record) for record in self.interface_sides
            ],
            "interface_pairs": [
                asdict(record) for record in self.interface_pairs
            ],
            "polymer_units": [
                asdict(record) for record in self.polymer_units
            ],
            "alternating_components": [
                list(component) for component in self.alternating_components
            ],
            "violations": list(self.violations),
        }


def _connected_components(
    nodes: set[str],
    adjacency: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    components: list[tuple[str, ...]] = []
    unvisited = set(nodes)
    while unvisited:
        start = min(unvisited)
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, set()) - component)
        unvisited -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


class _DisjointSet:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def analyze_interleaved_interface_seed_topology(
    instances,
) -> InterleavedInterfaceSeedTopology:
    """Build the general Interface-Seed interface--unit graph.

    ``A``, ``B``, ``C`` and ``D`` denote different supplied interface seeds,
    not necessarily fragments of one interface.  Every interface seed has two
    physical sides.  A protein unit is a scaffold-connected collection of any
    number of those sides, so units such as ``A-C-D`` and ``B-C-D`` are native
    graph cases rather than special scripts.

    A fragment may expose several interface ports.  Such overlapping sides
    are assigned to the same protein unit.  Continuous scaffold links then
    merge further sides into an ordered polymer component.  The resulting
    bipartite incidence graph has interface nodes of degree two and protein-
    unit nodes of arbitrary degree.

    The analyzer reports instead of raising because general assembly graphs
    may intentionally fall outside this Interface-Seed contract.
    """

    violations: list[str] = []
    interface_sides: list[InterfaceSeedSideRecord] = []
    interface_pairs: list[InterfaceSeedPairRecord] = []
    fragment_side_memberships: dict[str, list[str]] = defaultdict(list)
    side_pair: dict[str, str] = {}

    input_interfaces = sorted(
        (
            edge
            for edge in instances.interfaces.values()
            if edge.satisfaction_stage == "input"
        ),
        key=lambda edge: edge.id,
    )
    for edge in input_interfaces:
        left = instances.ports[edge.left_port_instance_id]
        right = instances.ports[edge.right_port_instance_id]
        left_fragments = tuple(dict.fromkeys(left.fragment_instance_ids))
        right_fragments = tuple(dict.fromkeys(right.fragment_instance_ids))
        left_side_id = f"{edge.id}:left"
        right_side_id = f"{edge.id}:right"
        if not left_fragments or not right_fragments:
            violations.append(
                f"Interface seed {edge.id!r} has an empty protomer side"
            )
            continue
        overlap = sorted(set(left_fragments) & set(right_fragments))
        if overlap:
            violations.append(
                f"Interface seed {edge.id!r} places the same physical "
                f"fragments on both sides: {overlap}"
            )
            continue

        side_records = (
            InterfaceSeedSideRecord(
                side_id=left_side_id,
                edge_instance_id=edge.id,
                role="left",
                fragment_instance_ids=left_fragments,
                copy_index=edge.source_copy_index,
            ),
            InterfaceSeedSideRecord(
                side_id=right_side_id,
                edge_instance_id=edge.id,
                role="right",
                fragment_instance_ids=right_fragments,
                copy_index=edge.target_copy_index,
            ),
        )
        interface_sides.extend(side_records)
        interface_pairs.append(
            InterfaceSeedPairRecord(
                edge_instance_id=edge.id,
                source_interface_id=edge.source_id,
                left_side_id=left_side_id,
                right_side_id=right_side_id,
                left_fragment_instance_ids=left_fragments,
                right_fragment_instance_ids=right_fragments,
                source_copy_index=edge.source_copy_index,
                target_copy_index=edge.target_copy_index,
                required=edge.required,
            )
        )
        side_pair[left_side_id] = edge.id
        side_pair[right_side_id] = edge.id
        for side in side_records:
            for fragment_id in side.fragment_instance_ids:
                fragment_side_memberships[fragment_id].append(side.side_id)

    side_ids = {side.side_id for side in interface_sides}
    side_groups = _DisjointSet(side_ids)

    # One physical fragment can expose several distinct interface faces.  All
    # such faces necessarily belong to the same protein unit.
    for memberships in fragment_side_memberships.values():
        for side_id in memberships[1:]:
            side_groups.union(memberships[0], side_id)

    continuous_links = sorted(
        (
            link
            for link in compiled_scaffold_links(instances).values()
            if not link.chain_break
        ),
        key=lambda link: link.id,
    )
    link_side_ids: dict[str, set[str]] = {}
    for link in continuous_links:
        from_sides = set(
            fragment_side_memberships.get(
                link.from_fragment_instance_id,
                (),
            )
        )
        to_sides = set(
            fragment_side_memberships.get(
                link.to_fragment_instance_id,
                (),
            )
        )
        if not from_sides:
            violations.append(
                f"Scaffold link {link.id!r} starts outside all supplied "
                f"interface sides: {link.from_fragment_instance_id!r}"
            )
        if not to_sides:
            violations.append(
                f"Scaffold link {link.id!r} ends outside all supplied "
                f"interface sides: {link.to_fragment_instance_id!r}"
            )
        touched_sides = from_sides | to_sides
        link_side_ids[link.id] = touched_sides
        if touched_sides:
            first = min(touched_sides)
            for side_id in touched_sides - {first}:
                side_groups.union(first, side_id)

        from_pairs = {side_pair[side_id] for side_id in from_sides}
        to_pairs = {side_pair[side_id] for side_id in to_sides}
        if from_pairs and from_pairs == to_pairs and len(from_pairs) == 1:
            violations.append(
                f"Scaffold link {link.id!r} directly joins the two sides "
                "of one supplied interface seed; expected a protein unit "
                "to combine different interface identities"
            )

    sides_by_root: dict[str, set[str]] = defaultdict(set)
    for side_id in side_ids:
        sides_by_root[side_groups.find(side_id)].add(side_id)

    sorted_side_groups = sorted(
        sides_by_root.values(),
        key=lambda values: tuple(sorted(values)),
    )
    unit_by_side: dict[str, str] = {}
    polymer_units: list[PolymerUnitRecord] = []
    for index, unit_sides in enumerate(sorted_side_groups, start=1):
        unit_id = f"polymer_unit_{index:03d}"
        for side_id in unit_sides:
            unit_by_side[side_id] = unit_id
        unit_links = tuple(
            link.id
            for link in continuous_links
            if link_side_ids[link.id] & unit_sides
        )
        source_link_ids = tuple(
            dict.fromkeys(
                link.source_id
                for link in continuous_links
                if link.id in unit_links
            )
        )
        unit_fragments = tuple(
            sorted(
                {
                    fragment_id
                    for side in interface_sides
                    if side.side_id in unit_sides
                    for fragment_id in side.fragment_instance_ids
                }
            )
        )
        unit_pairs = tuple(
            sorted({side_pair[side_id] for side_id in unit_sides})
        )
        polymer_units.append(
            PolymerUnitRecord(
                unit_id=unit_id,
                interface_side_ids=tuple(sorted(unit_sides)),
                interface_pair_ids=unit_pairs,
                fragment_instance_ids=unit_fragments,
                link_instance_ids=unit_links,
                source_link_ids=source_link_ids,
            )
        )

    incidence_adjacency: dict[str, set[str]] = defaultdict(set)
    incidence_nodes: set[str] = set()
    for pair in interface_pairs:
        interface_node = f"interface:{pair.edge_instance_id}"
        incidence_nodes.add(interface_node)
        left_unit = unit_by_side[pair.left_side_id]
        right_unit = unit_by_side[pair.right_side_id]
        if left_unit == right_unit:
            violations.append(
                f"Both sides of interface seed {pair.edge_instance_id!r} "
                f"belong to the same protein unit {left_unit!r}"
            )
        for unit_id in (left_unit, right_unit):
            unit_node = f"unit:{unit_id}"
            incidence_nodes.add(unit_node)
            incidence_adjacency[interface_node].add(unit_node)
            incidence_adjacency[unit_node].add(interface_node)

    components = _connected_components(
        incidence_nodes,
        incidence_adjacency,
    )
    if interface_pairs and len(components) != 1:
        violations.append(
            "Interface/unit incidence graph does not form one connected "
            f"assembly: {len(components)} components"
        )

    if not input_interfaces and not continuous_links:
        status = "not_applicable"
    elif (
        not violations
        and bool(interface_pairs)
        and bool(polymer_units)
        and len(components) == 1
    ):
        status = "valid_interface_unit_graph"
    else:
        status = "invalid_interface_unit_graph"

    return InterleavedInterfaceSeedTopology(
        status=status,
        interface_sides=tuple(interface_sides),
        interface_pairs=tuple(interface_pairs),
        polymer_units=tuple(polymer_units),
        alternating_components=components,
        violations=tuple(violations),
    )
