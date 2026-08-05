"""Bidirectional provenance mapping for compiled Interface-Seed objects."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Mapping

from pydantic import Field, model_validator

from rfd3_mosaic.schema import AssemblySpecification, CompiledInstanceSet
from rfd3_mosaic.schema.specs import Identifier, StrictModel


NonNegativeIndex = Annotated[int, Field(ge=0)]


class InstanceKind(str, Enum):
    MOTION_GROUP = "motion_group"
    FRAGMENT = "fragment"
    PORT = "port"
    SCAFFOLD_LINK = "scaffold_link"
    INTERFACE_EDGE = "interface_edge"


class InstanceProvenance(StrictModel):
    """Trace one compiled instance back to its source specification."""

    instance_id: str
    kind: InstanceKind
    source_id: Identifier
    motion_group_instance_id: str | None = None
    member_instance_ids: tuple[str, ...] = ()
    orbit_id: Identifier | None = None
    transform_set_id: Identifier | None = None
    copy_index: NonNegativeIndex
    transform_id: str
    source_path: Path | None = None
    source_selection: str | None = None
    provenance_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_kind_specific_fields(self) -> "InstanceProvenance":
        if (self.orbit_id is None) != (self.transform_set_id is None):
            raise ValueError(
                "orbit_id and transform_set_id must either both be set "
                "or both be absent"
            )
        if self.kind == InstanceKind.MOTION_GROUP:
            if self.motion_group_instance_id is not None:
                raise ValueError("Motion-group records cannot have an owner")
        elif (
            self.kind in (InstanceKind.FRAGMENT, InstanceKind.PORT)
            and self.motion_group_instance_id is None
        ):
            raise ValueError(
                f"{self.kind.value} records require a motion-group owner"
            )

        if self.kind == InstanceKind.FRAGMENT:
            if self.source_path is None or self.source_selection is None:
                raise ValueError(
                    "Fragment records require source_path and source_selection"
                )
        elif self.source_path is not None or self.source_selection is not None:
            raise ValueError(
                "Only fragment records may define source structure fields"
            )

        return self


class RFD3IndexMapping(StrictModel):
    """Adapter-provided RFD3 indices for one fragment instance."""

    entity_id: str
    chain_id: str
    residue_indices: tuple[NonNegativeIndex, ...] = ()
    atom_indices: tuple[NonNegativeIndex, ...] = ()

    @model_validator(mode="after")
    def validate_indices(self) -> "RFD3IndexMapping":
        if not self.residue_indices and not self.atom_indices:
            raise ValueError(
                "At least one residue or atom index must be provided"
            )
        if len(self.residue_indices) != len(set(self.residue_indices)):
            raise ValueError("RFD3 residue indices must be unique")
        if len(self.atom_indices) != len(set(self.atom_indices)):
            raise ValueError("RFD3 atom indices must be unique")
        return self


class MappingRegistry(StrictModel):
    """Serializable registry joining specs, instances, and adapter indices."""

    records: dict[str, InstanceProvenance]
    rfd3_mappings: dict[str, RFD3IndexMapping] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registry(self) -> "MappingRegistry":
        for instance_id, record in self.records.items():
            if instance_id != record.instance_id:
                raise ValueError(
                    "Provenance mapping key must match its instance ID"
                )
            if record.motion_group_instance_id is not None:
                owner = self.records.get(record.motion_group_instance_id)
                if owner is None or owner.kind != InstanceKind.MOTION_GROUP:
                    raise ValueError(
                        f"Record {instance_id!r} has an invalid motion-group "
                        "owner"
                    )
            for member_id in record.member_instance_ids:
                if member_id not in self.records:
                    raise ValueError(
                        f"Record {instance_id!r} references unknown member "
                        f"{member_id!r}"
                    )

        claimed_atom_indices: dict[int, str] = {}
        for fragment_instance_id, mapping in self.rfd3_mappings.items():
            record = self.records.get(fragment_instance_id)
            if record is None or record.kind != InstanceKind.FRAGMENT:
                raise ValueError(
                    "RFD3 mappings must reference a fragment instance"
                )
            for atom_index in mapping.atom_indices:
                previous_owner = claimed_atom_indices.get(atom_index)
                if previous_owner is not None:
                    raise ValueError(
                        f"RFD3 atom index {atom_index} is assigned to both "
                        f"{previous_owner!r} and {fragment_instance_id!r}"
                    )
                claimed_atom_indices[atom_index] = fragment_instance_id

        return self

    def source_to_instances(
        self,
        source_id: str,
        *,
        kind: InstanceKind | None = None,
    ) -> tuple[str, ...]:
        """Return deterministic compiled instance IDs for a source spec ID."""

        return tuple(
            instance_id
            for instance_id, record in self.records.items()
            if record.source_id == source_id
            and (kind is None or record.kind == kind)
        )

    def instance_to_source(self, instance_id: str) -> InstanceProvenance:
        """Return complete source provenance for one compiled instance."""

        try:
            return self.records[instance_id]
        except KeyError as error:
            raise KeyError(f"Unknown compiled instance {instance_id!r}") from error

    def group_fragments(self, group_instance_id: str) -> tuple[str, ...]:
        record = self.instance_to_source(group_instance_id)
        if record.kind != InstanceKind.MOTION_GROUP:
            raise ValueError(f"{group_instance_id!r} is not a motion group")
        return record.member_instance_ids

    def port_fragments(self, port_instance_id: str) -> tuple[str, ...]:
        record = self.instance_to_source(port_instance_id)
        if record.kind != InstanceKind.PORT:
            raise ValueError(f"{port_instance_id!r} is not a port")
        return record.member_instance_ids

    def link_fragments(self, link_instance_id: str) -> tuple[str, ...]:
        record = self.instance_to_source(link_instance_id)
        if record.kind != InstanceKind.SCAFFOLD_LINK:
            raise ValueError(f"{link_instance_id!r} is not a scaffold link")
        return record.member_instance_ids

    def edge_ports(self, edge_instance_id: str) -> tuple[str, ...]:
        record = self.instance_to_source(edge_instance_id)
        if record.kind != InstanceKind.INTERFACE_EDGE:
            raise ValueError(f"{edge_instance_id!r} is not an interface edge")
        return record.member_instance_ids

    def rfd3_indices(self, fragment_instance_id: str) -> RFD3IndexMapping:
        try:
            return self.rfd3_mappings[fragment_instance_id]
        except KeyError as error:
            raise KeyError(
                f"No RFD3 indices registered for {fragment_instance_id!r}"
            ) from error

    def with_rfd3_mappings(
        self,
        mappings: Mapping[str, RFD3IndexMapping],
    ) -> "MappingRegistry":
        """Return a revalidated registry with adapter mappings attached."""

        payload = self.model_dump(mode="python")
        merged = dict(self.rfd3_mappings)
        merged.update(mappings)
        payload["rfd3_mappings"] = merged
        return MappingRegistry.model_validate(payload)


def build_mapping_registry(
    spec: AssemblySpecification,
    instances: CompiledInstanceSet,
) -> MappingRegistry:
    """Build object-level provenance without importing or depending on RFD3."""

    records: dict[str, InstanceProvenance] = {}

    for instance_id, group in instances.motion_groups.items():
        records[instance_id] = InstanceProvenance(
            instance_id=instance_id,
            kind=InstanceKind.MOTION_GROUP,
            source_id=group.source_id,
            member_instance_ids=group.fragment_instance_ids,
            orbit_id=group.orbit_id,
            transform_set_id=group.transform_set_id,
            copy_index=group.copy_index,
            transform_id=group.transform_id,
        )

    for instance_id, fragment in instances.fragments.items():
        fragment_spec = spec.fragments[fragment.source_id]
        records[instance_id] = InstanceProvenance(
            instance_id=instance_id,
            kind=InstanceKind.FRAGMENT,
            source_id=fragment.source_id,
            motion_group_instance_id=fragment.motion_group_instance_id,
            orbit_id=fragment.orbit_id,
            transform_set_id=fragment.transform_set_id,
            copy_index=fragment.copy_index,
            transform_id=fragment.transform_id,
            source_path=fragment_spec.source,
            source_selection=fragment_spec.selection,
            provenance_tags=fragment_spec.provenance_tags,
        )

    for instance_id, port in instances.ports.items():
        records[instance_id] = InstanceProvenance(
            instance_id=instance_id,
            kind=InstanceKind.PORT,
            source_id=port.source_id,
            motion_group_instance_id=port.motion_group_instance_id,
            member_instance_ids=port.fragment_instance_ids,
            orbit_id=port.orbit_id,
            transform_set_id=port.transform_set_id,
            copy_index=port.copy_index,
            transform_id=port.transform_id,
        )

    for instance_id, link in instances.scaffold_links.items():
        endpoint_record = records[link.from_fragment_instance_id]
        records[instance_id] = InstanceProvenance(
            instance_id=instance_id,
            kind=InstanceKind.SCAFFOLD_LINK,
            source_id=link.source_id,
            member_instance_ids=(
                link.from_fragment_instance_id,
                link.to_fragment_instance_id,
            ),
            orbit_id=link.orbit_id,
            transform_set_id=endpoint_record.transform_set_id,
            copy_index=link.copy_index,
            transform_id=endpoint_record.transform_id,
        )

    for instance_id, edge in instances.interfaces.items():
        left_record = records[edge.left_port_instance_id]
        records[instance_id] = InstanceProvenance(
            instance_id=instance_id,
            kind=InstanceKind.INTERFACE_EDGE,
            source_id=edge.source_id,
            member_instance_ids=(
                edge.left_port_instance_id,
                edge.right_port_instance_id,
            ),
            orbit_id=edge.orbit_id,
            transform_set_id=left_record.transform_set_id,
            copy_index=edge.source_copy_index,
            transform_id=left_record.transform_id,
        )

    return MappingRegistry(records=records)
