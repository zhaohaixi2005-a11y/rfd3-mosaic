"""Directed protein-scaffold topology and RFD3-independent validation."""

from collections import defaultdict

from pydantic import Field, model_validator

from rfd3_mosaic.schema import CompiledInstanceSet, ScaffoldLinkInstance
from rfd3_mosaic.schema.specs import StrictModel


class LengthRange(StrictModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "LengthRange":
        if self.minimum > self.maximum:
            raise ValueError("Length-range intersection is empty")
        return self


class ScaffoldGraph(StrictModel):
    """Validated directed graph of compiled fragment instances."""

    nodes: tuple[str, ...]
    links: dict[str, ScaffoldLinkInstance]
    tied_length_ranges: dict[str, LengthRange]

    def outgoing_links(self, fragment_instance_id: str) -> tuple[str, ...]:
        self._require_node(fragment_instance_id)
        return tuple(
            link_id
            for link_id, link in self.links.items()
            if link.from_fragment_instance_id == fragment_instance_id
        )

    def incoming_links(self, fragment_instance_id: str) -> tuple[str, ...]:
        self._require_node(fragment_instance_id)
        return tuple(
            link_id
            for link_id, link in self.links.items()
            if link.to_fragment_instance_id == fragment_instance_id
        )

    def _require_node(self, fragment_instance_id: str) -> None:
        if fragment_instance_id not in self.nodes:
            raise KeyError(
                f"Unknown scaffold node {fragment_instance_id!r}"
            )


def _validate_no_continuous_cycles(
    nodes: tuple[str, ...],
    links: dict[str, ScaffoldLinkInstance],
) -> None:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for link in links.values():
        if not link.chain_break:
            adjacency[link.from_fragment_instance_id].append(
                link.to_fragment_instance_id
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(
                f"Continuous scaffold topology contains a cycle at {node!r}"
            )
        if node in visited:
            return
        visiting.add(node)
        for neighbor in adjacency.get(node, []):
            visit(neighbor)
        visiting.remove(node)
        visited.add(node)

    for node in nodes:
        visit(node)


def compile_scaffold_graph(instances: CompiledInstanceSet) -> ScaffoldGraph:
    """Compile links and reject combinatorially impossible topology."""

    nodes = tuple(instances.fragments)
    links = dict(instances.scaffold_links)

    incoming: dict[str, str] = {}
    outgoing: dict[str, str] = {}
    for link_id, link in links.items():
        if link.chain_break:
            continue
        previous_outgoing = outgoing.get(link.from_fragment_instance_id)
        if previous_outgoing is not None:
            raise ValueError(
                f"Fragment {link.from_fragment_instance_id!r} has multiple "
                f"continuous outgoing links: {previous_outgoing!r}, {link_id!r}"
            )
        outgoing[link.from_fragment_instance_id] = link_id

        previous_incoming = incoming.get(link.to_fragment_instance_id)
        if previous_incoming is not None:
            raise ValueError(
                f"Fragment {link.to_fragment_instance_id!r} has multiple "
                f"continuous incoming links: {previous_incoming!r}, {link_id!r}"
            )
        incoming[link.to_fragment_instance_id] = link_id

    tied_bounds: dict[str, tuple[int, int]] = {}
    for link in links.values():
        if link.tie_group is None:
            continue
        current = tied_bounds.get(
            link.tie_group,
            (link.minimum_length, link.maximum_length),
        )
        intersection = (
            max(current[0], link.minimum_length),
            min(current[1], link.maximum_length),
        )
        if intersection[0] > intersection[1]:
            raise ValueError(
                f"Tie group {link.tie_group!r} has incompatible length ranges"
            )
        tied_bounds[link.tie_group] = intersection

    _validate_no_continuous_cycles(nodes, links)

    return ScaffoldGraph(
        nodes=nodes,
        links=links,
        tied_length_ranges={
            tie_group: LengthRange(minimum=bounds[0], maximum=bounds[1])
            for tie_group, bounds in tied_bounds.items()
        },
    )
