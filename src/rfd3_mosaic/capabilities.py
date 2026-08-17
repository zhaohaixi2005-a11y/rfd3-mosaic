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
        id="ordinary_input_inspection",
        title="Ordinary-user input inspection and cage intent",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Split residue-disconnected interface patches in one PDB/mmCIF, "
            "report the observed component/port incidence graph and emit a "
            "short replayable cage-intent YAML supporting variadic "
            "participants, user-controlled physical multiplicities and "
            "basic assembly-size goals."
        ),
        dependencies=("public_fixed_xyz",),
        evidence=(
            "LRZ 772-test suite covers contact-patch, multi-fragment "
            "participant, component-incidence and intent replay contracts",
        ),
    ),
    CapabilityRecord(
        id="ordinary_binary_ring_resolution",
        title="Ordinary binary-interface Cn ring resolution",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Resolve one binary preserve-exact supplied interface into all "
            "chain-direction and adjacent-copy Cn ring hypotheses, then "
            "compile, rank, freeze and hash-replay standard public designs."
        ),
        dependencies=(
            "ordinary_input_inspection",
            "public_assembly_graph",
        ),
        evidence=(
            "real two-seed C3 resolutions on 2026-08-12 froze and strictly "
            "replayed standard public YAML candidates",
        ),
    ),
    CapabilityRecord(
        id="public_assembly_graph",
        title="Public component/interface/connection assembly graph",
        maturity=CapabilityMaturity.GPU_CANARY,
        public_interface=True,
        summary=(
            "Compile arbitrary rigid component nodes, reusable named "
            "interface faces, symmetry-neighbour relation edges and "
            "generated chain connections into the common "
            "AssemblySpecification."
        ),
        dependencies=("public_fixed_xyz",),
        evidence=(
            "50-step V100 T job 5735772 compiled one joint-rigid component, "
            "two reusable faces, one nonidentity neighbour edge and one "
            "generated connection through the public graph; all relation, "
            "constraint, continuity, clash and symmetry audits passed",
        ),
    ),
    CapabilityRecord(
        id="multi_chain_interface_seed_cage",
        title="Interleaved multi-interface-seed cage assembly",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Preserve any number of two-sided interface identities and "
            "compile protein units carrying arbitrary interface subsets "
            "such as A-C-D or B-C-D from "
            "one input while cross-pair scaffold links assemble polymer "
            "units such as A_i--B_(i+1), with explicit orbit ownership and "
            "no duplicated physical fragments."
        ),
        dependencies=(
            "public_assembly_graph",
            "multi_orbit_joint_control",
            "graph_pose_search",
        ),
        evidence=(
            "three supplied C3 interface identities resolved to replayable "
            "nine-chain public designs without inventing an interface",
        ),
    ),
    CapabilityRecord(
        id="multi_seed_polymer_path_cover",
        title="Bounded multi-seed polymer path-cover enumeration",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=False,
        summary=(
            "Enumerate deterministic rotation/reversal-unique alternating "
            "cycles for disjoint binary interface seeds while using every "
            "seed side exactly once. Results are topology-only and are "
            "explicitly non-executable until geometry, termini and symmetry "
            "actions are bound."
        ),
        dependencies=("ordinary_input_inspection",),
        evidence=(
            "LRZ path-cover, incidence-graph and strict-replay regressions",
        ),
    ),
    CapabilityRecord(
        id="ordinary_prepositioned_multi_seed_cn",
        title="Pre-positioned multi-seed Cn ordinary resolution",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Bind several disjoint binary preserve-exact seeds from one "
            "common reference frame into deterministic interleaved polymer "
            "units, enumerate chemical directions, closing seams and Cn "
            "windings, then require a valid expanded interface/unit graph "
            "before common static ranking and strict replay."
        ),
        dependencies=(
            "ordinary_binary_ring_resolution",
            "multi_seed_polymer_path_cover",
            "public_assembly_graph",
        ),
        evidence=(
            "two- and three-seed C3 resolution manifests accepted and froze "
            "strictly replayable public designs on LRZ",
        ),
    ),
    CapabilityRecord(
        id="graph_pose_search",
        title="Assembly-graph neighbour and pose search",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Compare requested finite symmetry groups, enumerate interface "
            "neighbours, sample declared component poses, rank complete "
            "static assemblies and freeze replayable public candidates."
        ),
        dependencies=("public_assembly_graph", "static_pose_sampling"),
        evidence=(
            "complete-assembly Cn/Dn/T/O/I initializer, bounded joint pose "
            "search and frozen replay regressions in the LRZ 772-test suite",
        ),
    ),
    CapabilityRecord(
        id="graph_interface_guidance",
        title="Symmetry-coupled graph interface guidance",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Guide generated residues on compiler-declared symmetry "
            "neighbour edges with joint contact, coverage, continuity, "
            "orientation, shape-proxy, anti-collapse and multi-interface "
            "objectives while fixed motif orbits remain authoritative."
        ),
        dependencies=("public_fixed_xyz",),
        evidence=(
            "CPU sampler tests cover patch identity, coverage, continuity, "
            "shape, orientation, clash rollback and final-output gating",
            "new multi-seed GPU replicate matrix remains pending",
        ),
    ),
    CapabilityRecord(
        id="joint_packing_mobility",
        title="Atomic packing-aware component mobility",
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Propose generated interface patches and all bounded mobile "
            "component-orbit SE(3) poses from one state, restore exact "
            "symmetry and fixed targets, and commit or roll back the whole "
            "packing transaction atomically."
        ),
        dependencies=(
            "graph_interface_guidance",
            "bounded_orbit_mobility",
            "multi_orbit_joint_control",
        ),
        evidence=(
            "LRZ sampler regressions cover radial-axial-rotation and free "
            "bounded-SE(3) proposals, joint energy acceptance and rollback",
            "locked/guided 50-step GPU replicate matrix is queued",
        ),
    ),
    CapabilityRecord(
        id="assembly_shape_contract",
        title="Assembly diameter and cavity contract",
        maturity=CapabilityMaturity.SCHEMA_ONLY,
        public_interface=True,
        summary=(
            "Use ordinary diameter/cavity ranges during full-assembly CPU "
            "pose search and check the same contract against final RFD3 CA "
            "morphology."
        ),
        dependencies=("graph_pose_search",),
        evidence=(
            "schema, objective lowering and final-audit regressions added; "
            "complete LRZ suite pending",
        ),
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
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=True,
        summary=(
            "Constrain selected per-atom radius, azimuth or axial "
            "coordinates during exact Cn/Dn diffusion."
        ),
        dependencies=("public_fixed_xyz",),
        evidence=(
            "Public YAML lowering emits exact runtime atom keys without "
            "mislabeling cylindrical atoms as Cartesian-fixed motifs",
            "RFD3 input prevalidation resolves complete Cn/Dn atom orbits "
            "and finite cylindrical masks",
            "The shared constraint lifecycle projects cylindrical DOFs at "
            "initialization, model prediction, guidance and finalization",
        ),
    ),
    CapabilityRecord(
        id="bounded_orbit_mobility",
        title="Bounded motif-orbit SE(3) mobility",
        maturity=CapabilityMaturity.ENGINEERING,
        public_interface=True,
        summary=(
            "Move independently coupled rigid components under per-step and "
            "cumulative SE(3) bounds."
        ),
        dependencies=("public_fixed_xyz",),
        evidence=(
            "V100 job 5733341 applied a non-zero bounded scaffold-driven "
            "SE(3) update and passed component, symmetry, continuity and "
            "clash audits",
            "V100 job 5733680 moved one complete two-fragment interface "
            "seed orbit while preserving all three C3 copies at 0.000015 A "
            "maximum per-copy internal RMSD",
            "V100 job 5733718 executed radial-only mobility with 0.074646 A "
            "translation, zero axial displacement and zero rotation; all "
            "required audits passed",
            "V100 job 5733719 executed radial-axial mobility with 0.161358 A "
            "translation including -0.157698 A axial displacement and zero "
            "rotation; all required audits passed",
        ),
    ),
    CapabilityRecord(
        id="dn_static",
        title="Static Dn exact assembly",
        maturity=CapabilityMaturity.GPU_CANARY,
        public_interface=False,
        summary="Compile declared Dn frames and fixed motif orbits.",
        evidence=(
            "D2/D3 registry properties and D3 two-orbit prevalidation",
            "50-step V100 job 5733912 denoised all six D3 group actions "
            "with two exact constraint orbits and passed input, constraint-"
            "orbit and scaffold-validity audits",
        ),
    ),
    CapabilityRecord(
        id="dn_dynamic_multi_orbit",
        title="Dynamic multi-orbit Dn control",
        maturity=CapabilityMaturity.GPU_CANARY,
        public_interface=False,
        summary=(
            "Jointly move multiple rigid motif orbits while reconstructing "
            "every declared Dn group action."
        ),
        dependencies=(
            "dn_static",
            "bounded_orbit_mobility",
            "multi_orbit_joint_control",
        ),
        evidence=(
            "50-step V100 D3 job 5733972 jointly moved two independent "
            "orbits over all six group actions, preserved 1158/1158 fixed "
            "heavy atoms at <=0.000014 A per-copy internal RMSD, and passed "
            "mobility, constraint, continuity, clash and symmetry audits",
        ),
    ),
    CapabilityRecord(
        id="multi_orbit_joint_control",
        title="Simultaneous multi-orbit control",
        maturity=CapabilityMaturity.GPU_CANARY,
        public_interface=False,
        summary=(
            "Jointly project or move several motif orbits without "
            "update-order dependence."
        ),
        dependencies=("bounded_orbit_mobility",),
        evidence=(
            "LRZ 423-test suite validates snapshot-synchronous proposals, "
            "atomic joint acceptance, declaration-order independence and "
            "two independently mobile compiled components",
            "50-step V100 job 5733788 jointly moved two independent C3 "
            "orbits by different bounded SE(3) increments, preserved both "
            "components at <=0.000015 A per-copy internal RMSD, and passed "
            "constraint, atomic mobility, continuity, clash and exact-"
            "symmetry audits",
            "50-step V100 job 5733972 extended the same atomic joint "
            "contract to two mobile D3 orbits across all six group actions "
            "and passed every required audit",
        ),
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
        maturity=CapabilityMaturity.CPU_VALIDATED,
        public_interface=False,
        summary=(
            "Typed finite rotation groups for tetrahedral, octahedral and "
            "icosahedral cages."
        ),
        dependencies=("multi_orbit_joint_control", "local_neighbourhood"),
        evidence=(
            "LRZ 468-test suite validates deterministic 12/24/60-element "
            "T/O/I proper-rotation registries, group closure, inverses, "
            "center preservation and complete AssemblySpecification "
            "instance expansion",
        ),
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
        FixedArrangementPolicy,
        TerminalGeneration,
        UserDesignTask,
        UserDesignSpec,
    )

    if not isinstance(design, UserDesignSpec):
        raise TypeError("Expected a UserDesignSpec")
    identifiers: list[str] = []
    task_optimizes_fixed_components = bool(
        design.task == UserDesignTask.CREATE_SYMMETRIC_INTERFACE
        and design.fixed_arrangement
        == FixedArrangementPolicy.OPTIMIZE_COMPONENTS
    )
    has_mobile_constraint = task_optimizes_fixed_components

    def require(capability_id: str) -> None:
        if capability_id not in identifiers:
            identifiers.append(capability_id)

    for constraint in design.constraints:
        if isinstance(constraint, FixedXYZConstraint):
            require("public_fixed_xyz")
            if constraint.pose.mode == "bounded_mobile":
                has_mobile_constraint = True
                require("bounded_orbit_mobility")
        elif isinstance(constraint, CylindricalConstraint):
            require("cylindrical_projector")
        elif isinstance(constraint, BoundedMobileConstraint):
            has_mobile_constraint = True
            require("bounded_orbit_mobility")
    if task_optimizes_fixed_components:
        require("bounded_orbit_mobility")
    if design.components:
        require("public_assembly_graph")
        require("public_fixed_xyz")
        if any(
            component.pose.mode == "bounded_mobile"
            for component in design.components.values()
        ):
            has_mobile_constraint = True
            require("bounded_orbit_mobility")
    if any(
        interface.required and interface.relation.mode == "contact"
        for interface in design.interfaces
    ) or any(
        isinstance(generation, TerminalGeneration)
        for generation in design.generation
    ):
        require("graph_interface_guidance")
        if task_optimizes_fixed_components:
            require("joint_packing_mobility")
    if design.assembly_shape is not None:
        require("assembly_shape_contract")
    if (
        design.sampling.initial_pose is not None
        or design.sampling.initial_poses
    ):
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
        if has_mobile_constraint:
            require("dn_dynamic_multi_orbit")
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
