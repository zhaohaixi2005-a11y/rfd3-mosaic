"""Authoritative repair of executable assembly candidates before replay.

Candidate discovery is allowed to expose ranges and imperfect initial poses.
Published candidates are not.  This module converts every repairable choice
that is already authorized by the public design into one explicit, replayable
decision while preserving the user's scientific topology:

* supplied interfaces, participants and component membership are immutable;
* symmetry and copy relations are immutable;
* polymer endpoints are immutable;
* a linker range may be bound to one exact length inside that range.

The standalone compiler is the sole geometry authority.  Restoration consumes
its fully symmetry-expanded manifest, so search, freezing and the native RFD3
adapter cannot silently evaluate different physical link instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rfd3_mosaic.schema import UserDesignSpec


@dataclass(frozen=True)
class LinkerLengthBinding:
    """One range-to-exact decision over all physical symmetry instances."""

    source_link_id: str
    tie_group: str | None
    configured_minimum: int
    configured_maximum: int
    midpoint: int
    required_minimum: int
    selected_length: int
    policy: str
    physical_instance_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_link_id": self.source_link_id,
            "tie_group": self.tie_group,
            "configured_range": [
                self.configured_minimum,
                self.configured_maximum,
            ],
            "midpoint": self.midpoint,
            "required_minimum_over_physical_instances": (
                self.required_minimum
            ),
            "selected_length": self.selected_length,
            "policy": self.policy,
            "physical_instance_ids": list(self.physical_instance_ids),
        }


@dataclass(frozen=True)
class CandidateRestorationResult:
    """A normal public design plus auditable, deterministic repairs."""

    design: UserDesignSpec
    linker_bindings: tuple[LinkerLengthBinding, ...]

    @property
    def changed(self) -> bool:
        return any(
            binding.configured_minimum != binding.selected_length
            or binding.configured_maximum != binding.selected_length
            for binding in self.linker_bindings
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "compiler_manifest_feasibility_restoration_v1",
            "changed": self.changed,
            "linker_length_bindings": [
                binding.to_dict() for binding in self.linker_bindings
            ],
            "preserved_invariants": [
                "supplied_interface_geometry",
                "component_membership",
                "interface_usage",
                "symmetry_and_copy_relations",
                "polymer_connection_endpoints",
            ],
        }


@dataclass(frozen=True)
class _PendingLinkerBinding:
    connection: Any
    minimum: int
    maximum: int
    midpoint: int
    required_minimum: int
    required_by_instance: tuple[tuple[str, int], ...]


def _configured_length_bounds(length: Any) -> tuple[int, int]:
    """Normalize the public exact-int and range spellings."""

    if isinstance(length, int):
        return length, length
    try:
        return int(length.minimum), int(length.maximum)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "Connection length must be an integer or minimum/maximum range"
        ) from error


def _link_reports_by_source(
    standalone_manifest: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    try:
        reports = standalone_manifest["validation"][
            "scaffold_link_geometry"
        ]["links"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Standalone manifest is missing scaffold-link geometry"
        ) from error
    if not isinstance(reports, list):
        raise TypeError("scaffold-link geometry links must be a list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        if not isinstance(report, dict):
            raise TypeError("scaffold-link geometry entry must be a mapping")
        source_id = report.get("source_link_id")
        if source_id is None:
            raise ValueError(
                "Scaffold-link geometry entry is missing source_link_id"
            )
        grouped.setdefault(str(source_id), []).append(report)
    return grouped


def _reports_for_public_connection(
    reports_by_source: dict[str, list[dict[str, Any]]],
    connection_id: str,
) -> list[dict[str, Any]]:
    """Resolve a public connection through the canonical IR name.

    Public ``connections`` lower to generated-segment IDs prefixed with
    ``connection__``.  Older/legacy Assembly IR may retain the public ID
    directly.  Accept exactly those two spellings so restoration works on
    both paths without fuzzy suffix matching or changing the published ID.
    """

    direct = reports_by_source.get(connection_id)
    lowered = reports_by_source.get(f"connection__{connection_id}")
    if direct and lowered:
        raise ValueError(
            "Standalone manifest ambiguously contains both public and "
            f"lowered source IDs for connection {connection_id!r}"
        )
    return direct or lowered or []


def bind_feasible_linker_lengths(
    design: UserDesignSpec,
    standalone_manifest: dict[str, Any],
) -> CandidateRestorationResult:
    """Freeze each generated connection at a symmetry-safe exact length.

    A range is user authorization to choose a length, not an instruction to
    use its midpoint.  The selected exact length is the larger of the range
    midpoint and the worst physical instance's 3.8 A contour requirement.
    If that value exceeds the declared maximum, the candidate is genuinely
    infeasible and is rejected rather than changing the user's range.
    """

    # Designs without generated polymer connections have nothing to repair.
    # Returning before reading the manifest is important for compatibility
    # with fixed-geometry and interface-only paths whose retained historical
    # manifests predate scaffold-link geometry diagnostics entirely.
    if not design.connections:
        return CandidateRestorationResult(
            design=design,
            linker_bindings=(),
        )

    reports_by_source = _link_reports_by_source(standalone_manifest)
    pending: list[_PendingLinkerBinding] = []
    for connection in design.connections:
        reports = _reports_for_public_connection(
            reports_by_source,
            connection.id,
        )
        if not reports:
            raise ValueError(
                "Standalone manifest contains no physical instances for "
                f"declared connection {connection.id!r}"
            )
        required_by_instance: list[tuple[str, int]] = []
        for report in reports:
            if bool(report.get("chain_break", False)):
                continue
            try:
                instance_id = str(report["link_instance_id"])
                required = int(
                    report["minimum_required_residues_at_3_8A"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Scaffold-link geometry is missing a valid physical "
                    f"contour requirement for {connection.id!r}"
                ) from error
            required_by_instance.append((instance_id, required))
        if not required_by_instance:
            raise ValueError(
                f"Connection {connection.id!r} has no continuous physical "
                "scaffold-link instances"
            )
        minimum, maximum = _configured_length_bounds(connection.length)
        midpoint = (minimum + maximum) // 2
        required_minimum = max(
            required for _, required in required_by_instance
        )
        if required_minimum > maximum:
            failing = [
                instance_id
                for instance_id, required in required_by_instance
                if required > maximum
            ]
            raise ValueError(
                "No linker length in the user-authorized range "
                f"[{minimum}, {maximum}] can span every symmetry instance "
                f"of {connection.id!r}; required={required_minimum}, "
                f"failing_instances={failing}"
            )
        pending.append(
            _PendingLinkerBinding(
                connection=connection,
                minimum=minimum,
                maximum=maximum,
                midpoint=midpoint,
                required_minimum=required_minimum,
                required_by_instance=tuple(required_by_instance),
            )
        )

    groups: dict[str, list[_PendingLinkerBinding]] = {}
    for item in pending:
        tie_group = item.connection.tie_group
        key = (
            f"tie:{tie_group}"
            if tie_group is not None
            else f"connection:{item.connection.id}"
        )
        groups.setdefault(key, []).append(item)

    selected_by_connection: dict[str, int] = {}
    binding_by_connection: dict[str, LinkerLengthBinding] = {}
    for group in groups.values():
        common_minimum = max(item.minimum for item in group)
        common_maximum = min(item.maximum for item in group)
        if common_minimum > common_maximum:
            connection_ids = [item.connection.id for item in group]
            raise ValueError(
                "Linker tie group has no common user-authorized length "
                f"for connections {connection_ids}: intersection="
                f"[{common_minimum}, {common_maximum}]"
            )
        group_midpoint = (common_minimum + common_maximum) // 2
        group_required = max(item.required_minimum for item in group)
        selected = max(group_midpoint, group_required)
        if selected > common_maximum:
            connection_ids = [item.connection.id for item in group]
            failing_instances = [
                instance_id
                for item in group
                for instance_id, required in item.required_by_instance
                if required > common_maximum
            ]
            raise ValueError(
                "No common linker length in the user-authorized tie-group "
                f"range [{common_minimum}, {common_maximum}] can span "
                f"connections {connection_ids}; required={group_required}, "
                f"failing_instances={failing_instances}"
            )
        tied = (
            group[0].connection.tie_group is not None
            and len(group) > 1
        )
        for item in group:
            policy = (
                "user_exact"
                if item.minimum == item.maximum == selected
                else "tie_group_common_midpoint"
                if tied and selected == group_midpoint
                else "tie_group_contour_sufficient"
                if tied
                else "configured_range_midpoint"
                if selected == item.midpoint
                else "configured_range_contour_sufficient"
            )
            binding = LinkerLengthBinding(
                source_link_id=item.connection.id,
                tie_group=item.connection.tie_group,
                configured_minimum=item.minimum,
                configured_maximum=item.maximum,
                midpoint=item.midpoint,
                required_minimum=item.required_minimum,
                selected_length=selected,
                policy=policy,
                physical_instance_ids=tuple(
                    instance_id
                    for instance_id, _ in item.required_by_instance
                ),
            )
            selected_by_connection[item.connection.id] = selected
            binding_by_connection[item.connection.id] = binding

    updated_connections = []
    bindings: list[LinkerLengthBinding] = []
    for connection in design.connections:
        selected = selected_by_connection[connection.id]
        bindings.append(binding_by_connection[connection.id])
        restored_length = (
            selected
            if isinstance(connection.length, int)
            else connection.length.model_copy(
                update={"minimum": selected, "maximum": selected}
            )
        )
        updated_connections.append(
            connection.model_copy(update={"length": restored_length})
        )

    restored = design.model_copy(
        update={"connections": tuple(updated_connections)}
    )
    # Guard against accidental future broadening of this repair boundary.
    if restored.components != design.components:
        raise RuntimeError("Feasibility restoration changed components")
    if restored.interfaces != design.interfaces:
        raise RuntimeError("Feasibility restoration changed interfaces")
    if restored.symmetry != design.symmetry:
        raise RuntimeError("Feasibility restoration changed symmetry")
    if tuple(
        (
            item.id,
            item.from_endpoint,
            item.to_endpoint,
            item.copy_relation,
        )
        for item in restored.connections
    ) != tuple(
        (
            item.id,
            item.from_endpoint,
            item.to_endpoint,
            item.copy_relation,
        )
        for item in design.connections
    ):
        raise RuntimeError(
            "Feasibility restoration changed polymer topology"
        )
    return CandidateRestorationResult(
        design=restored,
        linker_bindings=tuple(bindings),
    )


__all__ = [
    "CandidateRestorationResult",
    "LinkerLengthBinding",
    "bind_feasible_linker_lengths",
]
