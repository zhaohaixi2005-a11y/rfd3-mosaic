"""Machine-readable maturity ledger for user-visible Mosaic capabilities."""

from __future__ import annotations

from enum import IntEnum

from rfd3_mosaic.schema.specs import StrictModel


class CapabilityMaturity(IntEnum):
    """Ordered validation ladder; larger values carry stronger evidence."""

    PLANNED = 0
    SCHEMA_ONLY = 1
    CPU_VALIDATED = 2
    GPU_CANARY = 3
    ENGINEERING = 4
    STABLE = 5
    SCIENTIFICALLY_VALIDATED = 6

    @property
    def label(self) -> str:
        return self.name.lower()


class CapabilityRecord(StrictModel):
    id: str
    title: str
    maturity: CapabilityMaturity
    public_interface: bool
    summary: str
    evidence: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


CAPABILITIES: tuple[CapabilityRecord, ...] = (
    CapabilityRecord(
        id="c3_fixed_xyz_central",
        title="C3 exact fixed central motif",
        maturity=CapabilityMaturity.STABLE,
        public_interface=True,
        summary="Generate N/C extensions while restoring a complete fixed motif orbit.",
        evidence=(
            "200-step LRZ runs 5729451-5729453 passed motif and scaffold audits",
            "exact joint-orbit and declared-frame regression tests",
        ),
    ),
    CapabilityRecord(
        id="c3_fixed_xyz_interface",
        title="C3 exact cross-protomer interface seed",
        maturity=CapabilityMaturity.STABLE,
        public_interface=True,
        summary=(
            "Preserve one atomic interface seed spanning symmetry-related "
            "protomers."
        ),
        evidence=(
            "C3 50/100/200-step motif, continuity and symmetry audit evidence",
            "unified exact motif-precedence sampler tests",
        ),
    ),
    CapabilityRecord(
        id="public_fixed_xyz",
        title="Public fixed_xyz design lowering",
        maturity=CapabilityMaturity.ENGINEERING,
        public_interface=True,
        summary=(
            "Lower topology-neutral selectors into independently or jointly "
            "coupled rigid-geometry components in the exact backend."
        ),
        dependencies=("c3_fixed_xyz_central", "c3_fixed_xyz_interface"),
        evidence=("structure-bound selector, compiler and CLI unit suite",),
    ),
    CapabilityRecord(
        id="static_pose_sampling",
        title="Rigid initial-pose sampling",
        maturity=CapabilityMaturity.ENGINEERING,
        public_interface=True,
        summary=(
            "Sample radius, axial offset and fixed or Haar-SO(3) "
            "orientation before diffusion."
        ),
        evidence=("pose ensemble, LHS, QD and static geometry unit tests",),
    ),
    CapabilityRecord(
        id="functional_geometry_schema",
        title="Functional constraint hypergraph schema",
        maturity=CapabilityMaturity.SCHEMA_ONLY,
        public_interface=False,
        summary=(
            "Represent rigid fragments, atom geometry, chirality, relative "
            "poses and cooperative coordination hyperedges independently "
            "of global architecture."
        ),
    ),
    CapabilityRecord(
        id="cooperative_site_orbit",
        title="Cooperative multi-subunit functional-site orbit",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary=(
            "Bind, compile, project and audit one functional site spanning "
            "multiple symmetry-related subunits throughout diffusion."
        ),
        dependencies=("functional_geometry_schema", "public_fixed_xyz"),
    ),
    CapabilityRecord(
        id="cyclic_relation_compatibility",
        title="Cyclic pairwise-relation compatibility",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=False,
        summary=(
            "Identify Cn group-element equivalence classes compatible with "
            "one relative SE(3) relation while exposing unobserved cosets."
        ),
        dependencies=("functional_geometry_schema",),
        evidence=(
            "LRZ complete unit suite: 393 tests passed on 2026-08-05",
            "inverse-direction, noise, subgroup and screw-motion tests",
        ),
    ),
    CapabilityRecord(
        id="symmetry_discovery",
        title="Symmetry and order discovery",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary=(
            "Rank compatible Cn/Dn group, order, orbit assignment and group "
            "frame from local functional geometry."
        ),
        dependencies=(
            "cyclic_relation_compatibility",
            "static_pose_sampling",
        ),
    ),
    CapabilityRecord(
        id="topology_inference",
        title="Assembly topology inference",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary=(
            "Infer component partition and generated connectivity for each "
            "compatible architecture hypothesis."
        ),
        dependencies=("symmetry_discovery",),
    ),
    CapabilityRecord(
        id="cylindrical_projector",
        title="Rigid cylindrical orbit projector",
        maturity=CapabilityMaturity.SCHEMA_ONLY,
        public_interface=True,
        summary=(
            "Constrain selected orbit-pose radius, azimuth or axial "
            "coordinates during diffusion."
        ),
        dependencies=("public_fixed_xyz",),
    ),
    CapabilityRecord(
        id="bounded_orbit_mobility",
        title="Bounded motif-orbit SE(3) mobility",
        maturity=CapabilityMaturity.GPU_CANARY,
        public_interface=True,
        summary=(
            "Move independently coupled rigid components under per-step and "
            "cumulative SE(3) bounds."
        ),
        dependencies=("public_fixed_xyz",),
        evidence=("default-off Cn mobility controller and paired pilot jobs",),
    ),
    CapabilityRecord(
        id="dn_static",
        title="Static Dn exact assembly",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=False,
        summary="Compile declared Dn frames and fixed motif orbits.",
        evidence=("D2/D3 registry properties and D3 two-orbit prevalidation",),
    ),
    CapabilityRecord(
        id="multi_orbit_joint_control",
        title="Simultaneous multi-orbit control",
        maturity=CapabilityMaturity.SCHEMA_ONLY,
        public_interface=False,
        summary=(
            "Jointly project or move several motif orbits without "
            "update-order dependence."
        ),
        dependencies=("bounded_orbit_mobility", "dn_static"),
    ),
    CapabilityRecord(
        id="diffusion_feedback_refinement",
        title="Diffusion-in-the-loop pose refinement",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary=(
            "Refine continuous orbit pose from scaffold feedback while "
            "keeping one discrete architecture fixed per trajectory."
        ),
        dependencies=(
            "topology_inference",
            "bounded_orbit_mobility",
        ),
    ),
    CapabilityRecord(
        id="local_neighbourhood",
        title="Local symmetry-neighbourhood execution",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=False,
        summary=(
            "Denoise local copies while reconstructing the complete exact "
            "output orbit."
        ),
        evidence=("C12 local-neighbourhood kernel and sampler unit tests",),
    ),
    CapabilityRecord(
        id="polyhedral_groups",
        title="T/O/I finite-group execution",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary=(
            "Typed finite rotation groups for tetrahedral, octahedral and "
            "icosahedral cages."
        ),
        dependencies=("multi_orbit_joint_control", "local_neighbourhood"),
    ),
    CapabilityRecord(
        id="helical_window",
        title="Finite-window helical execution",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary=(
            "Execute a bounded repeat window with explicit neighbours and "
            "boundary semantics."
        ),
        dependencies=("local_neighbourhood",),
    ),
    CapabilityRecord(
        id="sequence_fold_validation",
        title="Sequence and fold validation pipeline",
        maturity=CapabilityMaturity.PLANNED,
        public_interface=False,
        summary="Sequence design, multimer prediction and interface-energy ranking.",
    ),
)


def capability_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "maturity_order": [item.label for item in CapabilityMaturity],
        "capabilities": [
            {
                **item.model_dump(mode="json"),
                "maturity": item.maturity.label,
            }
            for item in CAPABILITIES
        ],
    }


def capability_by_id(capability_id: str) -> CapabilityRecord:
    for item in CAPABILITIES:
        if item.id == capability_id:
            return item
    raise KeyError(f"Unknown Mosaic capability {capability_id!r}")


def required_capabilities_for_design(
    design: object,
) -> tuple[CapabilityRecord, ...]:
    """Return the maturity records implied by one public design.

    The import is intentionally local so the global ledger remains usable by
    lightweight tooling without introducing a schema import cycle.
    """

    from rfd3_mosaic.schema.design import (  # noqa: PLC0415
        BoundedMobileConstraint,
        CylindricalConstraint,
        FixedXYZConstraint,
        UserDesignSpec,
    )

    if not isinstance(design, UserDesignSpec):
        raise TypeError("Expected a UserDesignSpec")
    identifiers: list[str] = []

    def require(capability_id: str) -> None:
        if capability_id not in identifiers:
            identifiers.append(capability_id)

    for constraint in design.constraints:
        if isinstance(constraint, FixedXYZConstraint):
            require("public_fixed_xyz")
            if constraint.pose.mode == "bounded_mobile":
                require("bounded_orbit_mobility")
        elif isinstance(constraint, CylindricalConstraint):
            require("cylindrical_projector")
        elif isinstance(constraint, BoundedMobileConstraint):
            require("bounded_orbit_mobility")
    if design.sampling.initial_pose is not None:
        require("static_pose_sampling")
    if design.sampling.execution_backend == "local_neighbourhood":
        require("local_neighbourhood")
    symmetry_id = (
        design.symmetry
        if isinstance(design.symmetry, str)
        else design.symmetry.id
    )
    if symmetry_id.startswith("D"):
        require("dn_static")
    elif symmetry_id in {"T", "O", "I"}:
        require("polyhedral_groups")
    return tuple(capability_by_id(item) for item in identifiers)


__all__ = [
    "CAPABILITIES",
    "CapabilityMaturity",
    "CapabilityRecord",
    "capability_by_id",
    "capability_manifest",
    "required_capabilities_for_design",
]
