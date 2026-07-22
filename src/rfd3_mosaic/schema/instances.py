"""Compiled, immutable instances derived from Interface-Seed specifications."""

from pydantic import Field, model_validator

from rfd3_mosaic.schema.specs import (
    Identifier,
    StrictModel,
    TargetGeometrySpec,
    Terminus,
)


TransformRow = tuple[float, float, float, float]
TransformMatrix = tuple[
    TransformRow,
    TransformRow,
    TransformRow,
    TransformRow,
]


class SymmetryInstanceBase(StrictModel):
    """Provenance shared by symmetry-expanded runtime-independent objects."""

    source_id: Identifier
    orbit_id: Identifier | None = None
    transform_set_id: Identifier | None = None
    copy_index: int = Field(ge=0)
    transform_id: str
    transform: TransformMatrix

    @model_validator(mode="after")
    def validate_symmetry_provenance(self) -> "SymmetryInstanceBase":
        orbit_fields = (self.orbit_id, self.transform_set_id)
        if (orbit_fields[0] is None) != (orbit_fields[1] is None):
            raise ValueError(
                "orbit_id and transform_set_id must either both be set "
                "or both be absent"
            )
        if self.orbit_id is None and self.copy_index != 0:
            raise ValueError("Unsymmetrized instances must use copy_index 0")
        return self


class FragmentInstance(SymmetryInstanceBase):
    """One concrete symmetry copy of a source fragment."""

    id: str
    motion_group_instance_id: str


class MotionGroupInstance(SymmetryInstanceBase):
    """One independently addressable symmetry copy of a motion group."""

    id: str
    fragment_instance_ids: tuple[str, ...]


class InterfacePortInstance(SymmetryInstanceBase):
    """One concrete interface port attached to a motion-group instance."""

    id: str
    motion_group_instance_id: str
    fragment_instance_ids: tuple[str, ...]


class InterfaceEdgeInstance(StrictModel):
    """One resolved relationship between concrete interface-port copies."""

    id: str
    source_id: Identifier
    left_port_instance_id: str
    right_port_instance_id: str
    required: bool
    target_geometry: TargetGeometrySpec
    orbit_id: Identifier | None = None
    source_copy_index: int = Field(ge=0)
    target_copy_index: int = Field(default=0, ge=0)


class ScaffoldLinkInstance(StrictModel):
    """One concrete directed scaffold connection between fragment copies."""

    id: str
    source_id: Identifier
    from_fragment_instance_id: str
    from_terminus: Terminus
    to_fragment_instance_id: str
    to_terminus: Terminus
    minimum_length: int = Field(ge=0)
    maximum_length: int = Field(ge=0)
    tie_group: Identifier | None = None
    chain_break: bool = False
    orbit_id: Identifier | None = None
    copy_index: int = Field(ge=0)
    target_copy_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_link_instance(self) -> "ScaffoldLinkInstance":
        if self.minimum_length > self.maximum_length:
            raise ValueError("minimum_length cannot exceed maximum_length")
        if self.orbit_id is None and self.copy_index != 0:
            raise ValueError("Unsymmetrized links must use copy_index 0")
        if not self.chain_break:
            if self.from_terminus != Terminus.C:
                raise ValueError("Continuous links must start at C terminus")
            if self.to_terminus != Terminus.N:
                raise ValueError("Continuous links must end at N terminus")
        return self


class CompiledInstanceSet(StrictModel):
    """Deterministic result of expanding all groups, fragments, and ports."""

    motion_groups: dict[str, MotionGroupInstance]
    fragments: dict[str, FragmentInstance]
    ports: dict[str, InterfacePortInstance]
    interfaces: dict[str, InterfaceEdgeInstance] = Field(default_factory=dict)
    scaffold_links: dict[str, ScaffoldLinkInstance] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_instance_references(self) -> "CompiledInstanceSet":
        for instance_id, group in self.motion_groups.items():
            if instance_id != group.id:
                raise ValueError("Motion-group mapping key must match instance ID")
            for fragment_id in group.fragment_instance_ids:
                fragment = self.fragments.get(fragment_id)
                if fragment is None:
                    raise ValueError(
                        f"Motion group {instance_id!r} references unknown "
                        f"fragment instance {fragment_id!r}"
                    )
                if fragment.motion_group_instance_id != instance_id:
                    raise ValueError(
                        f"Fragment instance {fragment_id!r} has the wrong "
                        "motion-group owner"
                    )

        for instance_id, fragment in self.fragments.items():
            if instance_id != fragment.id:
                raise ValueError("Fragment mapping key must match instance ID")
            if fragment.motion_group_instance_id not in self.motion_groups:
                raise ValueError(
                    f"Fragment {instance_id!r} references unknown motion group"
                )

        for instance_id, port in self.ports.items():
            if instance_id != port.id:
                raise ValueError("Port mapping key must match instance ID")
            group = self.motion_groups.get(port.motion_group_instance_id)
            if group is None:
                raise ValueError(
                    f"Port {instance_id!r} references unknown motion group"
                )
            for fragment_id in port.fragment_instance_ids:
                if fragment_id not in group.fragment_instance_ids:
                    raise ValueError(
                        f"Port {instance_id!r} references fragment "
                        f"{fragment_id!r} outside its motion group"
                    )

        occupied_endpoints: set[tuple[str, Terminus]] = set()
        for instance_id, edge in self.interfaces.items():
            if instance_id != edge.id:
                raise ValueError(
                    "Interface-edge mapping key must match instance ID"
                )
            if edge.left_port_instance_id not in self.ports:
                raise ValueError(
                    f"Interface edge {instance_id!r} references unknown "
                    "left port instance"
                )
            if edge.right_port_instance_id not in self.ports:
                raise ValueError(
                    f"Interface edge {instance_id!r} references unknown "
                    "right port instance"
                )

        for instance_id, link in self.scaffold_links.items():
            if instance_id != link.id:
                raise ValueError(
                    "Scaffold-link mapping key must match instance ID"
                )
            for fragment_id, terminus in (
                (link.from_fragment_instance_id, link.from_terminus),
                (link.to_fragment_instance_id, link.to_terminus),
            ):
                if fragment_id not in self.fragments:
                    raise ValueError(
                        f"Scaffold link {instance_id!r} references unknown "
                        f"fragment instance {fragment_id!r}"
                    )
                endpoint = (fragment_id, terminus)
                if endpoint in occupied_endpoints:
                    raise ValueError(
                        f"Scaffold endpoint {fragment_id!r}:{terminus.value} "
                        "is used more than once"
                    )
                occupied_endpoints.add(endpoint)

        return self
