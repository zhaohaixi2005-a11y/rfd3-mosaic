# RFD3-Mosaic Productization and Foundry Fork Plan

## Purpose

RFD3-Mosaic is a symmetry-aware protein-assembly diffusion engine built on a
maintained RFD3 fork. The project is not merely an external wrapper: exact
constraint restoration, coupled symmetry noise, orbit-state projection,
bounded motif motion and memory-aware symmetry execution require access to the
RFD3 timestep loop.

The goal is therefore not to remove the RFD3 modifications. The goal is to
turn them into a controlled, testable and upgradeable Mosaic-RFD3 engine while
providing a simple ordinary-user surface and a retained expert assembly-graph
surface above one shared compiler/runtime.

## Non-negotiable scientific invariants

The product refactor must preserve all currently validated behavior:

1. A complete fixed motif orbit has precedence over a symmetry projection.
2. Exact cyclic copies remain related by the declared transform registry.
3. Fixed motif atoms are restored during sampling, not only after inference.
4. Symmetry-coupled noise and orbit-average state remain available.
5. Central motifs and cross-protomer interface seeds use the same geometric
   compiler and exact runtime semantics.
6. Every completed run passes all audits declared by its compiled constraints.
7. Refactoring cannot silently relax clash, continuity, motif or symmetry
   thresholds.

## Product boundary

The target architecture is:

```text
input PDB/mmCIF -> SimpleCageIntentSpec -> inspect/resolver -> frozen candidate
                                      \
expert UserDesignSpec -----------------+-> UserDesignSpec
simple central/interface template -----/
        |
        v
AssemblySpecification + ConstraintPlan
        |
        v
RFD3 BackendPlan
        |
        v
Mosaic-RFD3 engine
        |
        v
Executor (local or Slurm)
        |
        v
Artifacts, provenance and audits
```

An optional functional-geometry frontend sits above this explicit execution
path:

```text
FunctionalGeometrySpec
        -> constraint hypergraph and symmetry-orbit compilation
        -> optional architecture discovery and ranking
        -> one or more explicit UserDesignSpec candidates
        -> the unchanged compilation/runtime path above
```

Discovery is additive: it must not hide or replace the explicit route used by
current golden regressions.

The simple and expert authoring levels are additive, not separate products.
The already executable central/interface templates are the narrow ordinary
motif-scaffolding path. The `simple_cage_intent` path now also resolves its
first conservative executable slice: one binary preserve-exact seed into
ranked Cn-ring `UserDesignSpec` candidates. A second, experimental slice now
bridges several disjoint binary preserve-exact seeds only when they are
already co-positioned in one reference frame: it binds deterministic path
covers to full-orbit Cn windings and submits them to the shared static
compiler/replay gate. The generic path-cover output remains topology-only;
unknown seed poses, hyperedges, heteromer/homomer inference and non-Cn cage
actions are still unresolved rather than guessed. Expert mode exposes
named components, ports, interfaces, connections and numerical overrides.
All three entrances must produce the same `AssemblySpecification`; no feature
may introduce a second sampler or topology-specific submission script.

The general scientific model is an interface **relation hypergraph** plus an
independent covalent scaffold graph.  One relation has two or more
participants; the original two-helix interface seed is the common binary
special case, not a maximum enforced by the product.  Interface type count,
ports per component and physical interface multiplicity are separate values.

## Current milestone: finish the product spine

The current phase is platform engineering, not selection of a specific
functional site or benchmark protein. Work proceeds through one invariant
spine:

```text
UserDesignSpec
    -> AssemblySpecification + ConstraintPlan + SamplingPlan
    -> BackendPlan
    -> Mosaic-RFD3
    -> execution + provenance + declared audits
```

Every public capability must cross the complete spine. A schema-only
constraint, a standalone preparation script or a sampler branch without an
independent audit is not a completed feature.

Product completion is incremental rather than all-or-nothing. Version 0.1 is
the protected, normally usable exact motif-scaffolding product: simple
supplied-interface and central-motif YAMLs, Cn execution, strict preflight,
provenance, required audits and portable reporting. Complex cage discovery,
new-interface packing, additional finite/infinite groups and sequence/fold
validation are versioned capability layers above this baseline. They do not
block routine use and may not destabilize it.

The practical release train is:

1. **v0.1 exact scaffolding** -- clean installation, simple C3 examples,
   supplied-interface and central-motif GPU golden replays, reports;
2. **v0.2 packing guidance** -- generated-interface GPU evidence, calibrated
   defaults and all-atom post-generation packing metrics;
3. **v0.3 cage graphs** -- multi-face/multi-orbit Dn/T engineering support,
   continuous graph-aware pose search and reproducible candidate replay;
4. **v0.4 broader groups** -- O/I and bounded high-order execution, followed
   separately by helical semantics;
5. **v0.5 designability pipeline** -- sequence design, multimer refolding and
   unified final ranking.

Each release must remain useful without the next one. Experimental options are
visible through the capability ledger but are not silently selected by the
ordinary-user templates.

## Flagship acceptance contract: interleaved interface pairs and protein units

The long-term cage target is an arbitrary **N**, not a hard-coded two or eight.
One input PDB/mmCIF may contain N fixed interface identities. `A/B/C/D/...`
name different interfaces; each has two physical sides such as
`Interface_A=(A.left,A.right)`. The supplied interface graph preserves each
complete `i.left<->i.right` geometry; a
separate scaffold graph connects halves from neighbouring pairs into polymer
units containing arbitrary ordered interface-side paths such as `A-C-D` or
`B-C-D`. They must become one connected protein assembly
while every required relative geometry and the declared global symmetry remain
valid. The original two-helix input is the one-pair primitive. This is
different from merely allowing N YAML interface entries or concatenating all
fixed fragments into one ordered chain.

The component/interface mapping is deliberately many-to-many. A component
type may own several local-frame ports, different component types may expose
different interface sets, and symmetry may instantiate one interface type
many times. Pairwise relations are the common case, but the final relation IR
must also permit three-or-more participant hyperedges for cooperative
multi-subunit junctions.

The ordinary-user surface should eventually require only one input structure,
fixed chains/residue ranges, optional symmetry preferences and generated
lengths.
The expert surface retains explicit components, ports, neighbour transforms,
orbit assignments and pose bounds. Both lower into the same assembly IR and
runtime.

The first input-driven slice now exists as `rfd3-mosaic inspect`. It detects
chain-pair interface patches from one PDB/mmCIF and writes a replayable
`simple_cage_intent`. The ordinary user may specify broad cage properties and
the physical usage of each interface identity (`auto`, exact or range), while
the inspection thresholds are retained as provenance. `plan` and `validate`
consume this intent. A conservative generic full-orbit filter already ranks
Cn/Dn/T/O/I group-order compatibility from usage and subunit bounds. Execution
for the general case remains fail-closed until the architecture resolver
converts interface-side ownership, polymer paths, symmetry and copy relations
into a normal public graph and the common `AssemblySpecification`; it must not
infer a one-to-one topology merely because an input relation is pairwise. The
pre-positioned binary Cn slice described below is the only current multi-seed
exception, and it uses supplied coordinates rather than solving them.

The intent-level interface is already variadic: `participants` contains two
or more chains/fragments and validation checks that their selected contact
graph is connected. Automatic inspection emits pairwise candidates as a
conservative starting point, while a user may merge several candidates into a
cooperative multi-participant interface. Lowering those hyperedges into the
runtime remains distinct from merely decomposing them into unrelated pairs.

The compiler/runtime contract is:

```text
one input containing Interface_A ... Interface_N
    -> bind component types, instances and their local-frame ports
    -> preserve every declared pairwise or multi-participant relation
    -> derive arbitrary polymer paths such as A-C-D or B-C-D
    -> non-duplicating symmetry orbit + stabilizer/coset assignment
    -> optional ordered paths inside each polymer unit
    -> joint group-closure, clash and linker-feasibility solve
    -> one frozen, replayable AssemblySpecification
    -> unified exact/bounded constraint-orbit diffusion
    -> per-seed relation audit + whole-cage scaffold audit
```

Current subset: Assembly IR already stores interface edges and generated
scaffold links as different instance types, and the adapter emits interface
audit metadata separately from contigs. A general interface--unit incidence
analyzer now validates already-expanded graphs, and the pre-positioned Cn
bridge uses it as a post-lowering invariant. It is a validator, not an inverse
ownership solver. The local ordered-path addition is useful only for
multi-fragment polymer units and still awaits LRZ regression. Remaining pieces
include unknown many-interface component ownership, relation hyperedges,
general mapping of already-present input chains to symmetry actions,
mixed vertex/edge/face multiplicities, general N-interface joint pose solving
and GPU evidence beyond two independently controlled orbits. Until those gates
pass, the software must not claim general multi-pair interface-seed cage
design.

The ordered implementation sequence is:

1. protect the Foundry base, Mosaic patch boundary, source snapshot,
   provenance and central/interface golden regressions;
2. finish the single public `UserDesignSpec` entry and fail-closed lowering;
3. complete `fixed_xyz`, then cylindrical projection, then bounded rigid
   mobility, including conflicts and composition;
4. unify fixed/grid/random/LHS/Sobol sampling, reproducible seeds, candidate
   manifests and replay;
5. expose a stable sampler lifecycle rather than adding conditional blocks to
   the timestep loop;
6. automate Foundry upgrade compatibility and golden replay;
7. only then promote multi-orbit, additional groups and scientific discovery
   layers.

The preliminary functional-geometry and cyclic-relation modules remain
internal extension points during this phase. They are not public CLI features
and do not redirect the current implementation sequence.

## Future scientific hierarchy: functional-site-first assembly design

A future scientific question is whether cooperative geometry formed by
several subunits can be compiled into exact or bounded constraint orbits and
maintained throughout all-atom diffusion while the surrounding symmetric
assembly is generated.

```text
scientific lead: cooperative functional-site design
method core:     constraint-orbit diffusion
search layer:    symmetry/topology compatibility discovery
```

The first internal `FunctionalGeometrySpec` now represents symbolic
functional atoms and rigid/soft-rigid fragments, distance, angle, periodic
dihedral and chirality residuals, relative-pose bounds and multi-fragment
hyperedges such as three-subunit metal coordination. A backend-independent
evaluator now produces relation-level observations, errors and non-negative
violations. Structure selector binding, a standalone artifact audit and
constraint-orbit lowering remain required before any GPU coupling.

The first flagship closed loop uses one cooperative site spanning three
symmetric subunits. A complete rigid site may be supplied initially. Mosaic
must restore its complete orbit at every timestep, generate the surrounding
assembly, pass continuity/clash/symmetry/site audits, and support downstream
sequence design and refolding evaluation. Bounded radial, axial, tilt and
twist refinement follows only after the rigid case is established.

## Inverse assembly search layer

The later search frontend asks an underdetermined inverse question:

```text
local functional geometry
    -> compatible global symmetry/topology hypotheses
    -> exact all-atom assembly realization
```

The software must return a ranked, auditable set of hypotheses and explicit
infeasibility reasons. It must not imply that one local motif uniquely
determines one global assembly.

One hypothesis contains three variable classes:

- discrete: group/order, orbit assignment, copy relation, component partition
  and generated connectivity topology;
- continuous: group frame, radius, azimuth, axial offset, tilt, twist and
  bounded rigid correction;
- generated: RFD3 coordinates, followed later by sequences and refolded
  structures.

The outer solver follows six stages:

1. bind supplied atoms into rigid functional fragments and pairwise or
   higher-order local-frame relations;
2. enumerate C2-C8 first, then D2-D6, including orbit offsets and compatible
   fragment assignments;
3. fit one common group frame and reject candidates that fail SE(3)
   compatibility, identity, inverse, composition or closure;
4. infer chain/component partitions and directed generated connections, then
   apply conservative contour-length, terminus, clash and cavity gates;
5. optimize only the admitted continuous pose degrees of freedom and compile
   ranked candidates into explicit `AssemblySpecification` objects;
6. realize a short list with the existing exact Mosaic-RFD3 runtime.

For a measured relative transform `T_hat`, candidate group action `Q` and
fitted global frame `H`, the core compatibility residual contains:

```text
d_SE3(T_hat, H Q H^-1)
```

Three-subunit metal sites and other cooperative functional geometries must be
represented as hyperedges rather than unrelated pairwise contacts.

Initial ranking uses explicit non-negative residuals:

```text
E = w_geometry     * E_geometry
  + w_closure      * E_closure
  + w_connectivity * E_connectivity
  + w_clash        * E_clash
  + w_pose_prior   * E_pose_prior
```

Hard gates precede ranking. Foldability, sequence designability and interface
energetics are downstream evidence and cannot be inferred from this CPU score.

This layer begins only after the functional-geometry compiler and a fixed
multi-subunit flagship are established. Its first bounded scope is one rigid
cross-subunit motif, C2-C8 enumeration, ranked order/orbit offset/group frame,
and SE(3), closure, clash and linker gates. Architecture recovery on known
assemblies is its first benchmark.

Discrete symmetry or topology must not switch inside one Euler trajectory in
the first implementation. Later beam/SMC exploration creates separately
identified trajectories. Bounded timestep mobility refines only continuous
pose variables inside one frozen architecture hypothesis.

Topology, constraints and execution are separate concerns:

- topology describes connectivity and where new residues are generated;
- symmetry describes relations between copies;
- constraints describe fixed, bounded or guided degrees of freedom;
- execution profiles describe machines and schedulers.

`central_motif` and `interface_seed` may remain convenience presets, but they
must not become separate samplers or permanent user pipelines.

## User contract

The canonical interface will be one design file and one executable:

```bash
rfd3-mosaic plan design.yaml
rfd3-mosaic run design.yaml
rfd3-mosaic status RUN_ID
rfd3-mosaic report RUN_ID
```

Constraints are optional, repeatable declarations. For example:

```yaml
constraints:
  - kind: fixed_xyz
    selector: A12-20,A26-37
    atoms: all
  - kind: cylindrical
    selector: A40-55
    atoms: ca
    axis: symmetry
    keep: [radius, azimuth]
```

If no motif constraint is declared, motif atoms receive no additional Mosaic
fixing. The selected topology and symmetry may still constrain the design.

## Constraint execution model

Every user constraint compiles to a topology-neutral `ConstraintPlan` with
four distinct categories:

1. hard projectors, such as exact Cartesian coordinate restoration;
2. bounded projectors, such as radial or angular trust regions;
3. guidance terms, such as contacts or scaffold objectives;
4. audit requirements, derived from the declared constraint semantics.

Conflicts are detected per degree of freedom. Overlapping selectors are legal
when they constrain compatible degrees of freedom. Runtime precedence must
never be guessed from declaration order.

The sampler lifecycle remains conceptually:

```text
denoise proposal
-> optional guidance
-> exact symmetry projection
-> hard and bounded constraint projection
-> closure validation
-> accept the next diffusion state
```

The implementation may use a unified joint solve, but its observable behavior
must be covered by golden tests before replacing the current exact path.

## Mosaic-RFD3 fork policy

Deep RFD3 changes are permitted and expected. They must be managed as an
ordered patch layer rather than accumulated as undocumented edits:

1. compatibility: declared frames and feature transport;
2. exact constraints: motif restoration, coupled noise and orbit state;
3. runtime hooks: stable sampler lifecycle integration points;
4. memory: chunking and local-neighbourhood execution;
5. experimental guidance: mobility and scaffold-derived objectives.

The long-term target is a thin, explicit integration surface in upstream RFD3
files. Mosaic algorithms remain first-class functionality but move behind
well-defined runtime interfaces. Code is migrated one component at a time,
with old and new paths retained until equivalence is demonstrated.

## Foundry upgrade procedure

Never merge a new Foundry version directly into the active development branch.
For every upgrade:

1. fetch the official Foundry repository;
2. create a clean compatibility branch at the proposed upstream commit;
3. replay the ordered Mosaic patch layers;
4. run API and feature-contract tests;
5. run the complete Mosaic unit suite;
6. run compiler golden fixtures and native prevalidation;
7. run a central-motif GPU canary;
8. run a cross-protomer interface-seed GPU canary;
9. run at least one 200-step golden regression;
10. promote the new compatibility record only after all required gates pass.

Generic RFD3 correctness fixes may be proposed upstream independently. Mosaic
domain concepts, compiler semantics, assembly objectives, execution and audit
logic remain in Mosaic.

## Validation tiers

### Stable

- exact C3 central motif;
- exact C3 cross-protomer interface seed;
- declared-frame symmetry;
- motif, symmetry, continuity and clash audit gates.

### Engineering

- static C5/C6/C7 sampling;
- D3 two-orbit compilation and prevalidation;
- local-neighbourhood kernels.

### Experimental

- bounded motif SE(3) mobility;
- scaffold-derived guidance;
- high-order cyclic execution;
- T/O/I symmetry.

Experimental code is preserved and tested, but remains disabled by default
until it passes the same end-to-end gates as stable functionality.

## Implementation phases

### Phase 0: freeze and identify the known-good engine

- keep `snapshot/validated-mosaic-2026-08-05` immutable;
- record the Mosaic commit and Foundry base commit;
- record imported Foundry reference changes;
- capture Git, environment, checkpoint and scheduler provenance for every run;
- preserve central-motif and interface-seed evidence as golden records;
- replay the unit suite from the exact snapshot commit on LRZ.

Exit gate: a result can be traced to source commit, compatibility manifest,
checkpoint identity, resolved configuration, input hashes and audit reports.

### Phase 1: stable user design interface

- replace manual experiment dictionaries with a strict `UserDesignSpec`;
- accept raw PDB/mmCIF inputs for supported presets;
- implement `plan`, `run`, `status` and `report`;
- keep CLI overrides small and deterministic;
- render the full execution plan before submission;
- preserve the existing exact sampler behavior.

Exit gate: the existing central and interface examples compile to the same
native RFD3 contracts and pass their current audits.

Current implementation slice (2026-08-05):

- immutable `UserDesignSpec` now separates raw input, symmetry, generated
  regions and optional constraints without a `topology.kind` field;
- `fixed_xyz`, `cylindrical` and `bounded_mobile` are repeatable operators;
- the earlier names `full_xyz_fixed`, `ca_cylindrical_fixed` and
  `bounded_mobile_interface` are accepted as compatibility spellings and
  compile to the same canonical operators;
- `rfd3-mosaic validate/plan` can inspect this public declaration;
- structure-aware selector binding resolves declared ranges to exact atom
  identities and detects partial-selector DOF conflicts;
- terminal and between-region designs using `fixed_xyz(atoms=all)` now lower
  into the common `AssemblySpecification` and existing exact RFD3 runtime;
- `fixed_xyz` now has explicit component semantics: comma-separated ranges in
  one declaration are jointly rigid, declarations sharing `coupling_group`
  are jointly rigid, and otherwise declarations remain independent. Each
  component is evaluated after one common Kabsch fit; laboratory-frame
  coordinate offsets are not part of the acceptance contract;
- every fixed component now declares its own public pose policy. `fixed` is
  the backward-compatible default; `bounded_mobile` maps to an independent
  orbit-rigid SE(3) controller with cumulative bounds, timestep window and
  per-step trust region. The worker enables mobility from compiler-emitted
  runtime features rather than from a hand-written Slurm flag. Mobile designs
  automatically require a second semantic audit proving that the controller
  ran and that final component poses remained within their declared bounds;
- render/submit reject unconstrained endpoints and the not-yet-bound
  `cylindrical`/top-level `bounded_mobile` operators. Nested
  `fixed_xyz.pose.mode: bounded_mobile` is executable. No new constraint can
  be silently accepted and then ignored by the sampler.
- a machine-readable capability ledger now exposes validation maturity,
  dependencies and public visibility through `rfd3-mosaic capabilities`;
- public designs now distinguish one rigid pre-diffusion `initial_pose` from
  diffusion seed/timestep sampling. Radius, axial offset and fixed or
  Haar-uniform SO(3) orientation lower into the existing assembly IR without
  adding a topology-specific sampler path;
- omitting `initial_pose` leaves input coordinates unchanged, and the pose
  seed is recorded separately from the diffusion seed.

All features advance through the same ordered evidence ladder:

```text
planned -> schema_only -> cpu_validated -> gpu_canary
        -> engineering -> stable -> scientifically_validated
```

The dependency order is fixed-XYZ golden regression, functional-geometry
schema/binding/audit, a rigid three-subunit functional-site GPU loop,
single-orbit cylindrical projection, bounded orbit mobility, simultaneous
multi-orbit control, then Cn relation compatibility plus topology/connectivity
ranking. Dn and finite T/O/I groups follow demonstrated multi-orbit control.
Helical symmetry is a separate finite-window backend and is not treated as a
large cyclic group.

### Phase 2: unified ConstraintPlan

- compile existing fixed motifs into `fixed_xyz` operators;
- give every constraint an explicit reference frame and orbit scope;
- separate hard, bounded and guidance semantics;
- detect conflicts by constrained degree of freedom;
- derive audit requirements from constraints;
- retain topology presets only as frontend conveniences.

Exit gate: the new plan is behaviorally equivalent to the current exact path
for fixed motifs and can represent a design with no motif constraint.

The first backend-independent `ConstraintPlan` compiler is now present. It
assigns deterministic operator IDs, records atom/orbit scope and reference
frame, separates hard and bounded projector stages, detects exact-selector
DOF conflicts, and requires each backend to declare supported operator kinds.
Structure-aware partial-selector overlap and the first exact fixed-XYZ
assembly/runtime binding are now implemented locally. Cylindrical projection,
bounded-pose binding and behavioral-equivalence GPU gates remain part of this
phase's exit gate.

### Phase 3: runtime boundary cleanup

- define versioned Mosaic sampler lifecycle hooks;
- route initialization, per-step projection and finalization through the same
  constraint runtime;
- move Mosaic algorithms behind the runtime interface without deleting them;
- retain feature flags for old and new paths during migration;
- compare timestep invariants and final outputs before switching defaults.

Exit gate: central and interface golden cases remain unchanged within declared
numeric tolerances after the runtime reorganization.

The first Phase 3 slice is now implemented behind the existing exact sampler.
`MosaicConstraintRuntime` owns initial projection, optional scheduled target
proposal, conditioning refresh, model-output projection, Euler-state
projection, post-guidance projection and finalization.  Every stage delegates
to the same `UnifiedJointProjector`; rejected proposals cannot mutate the hard
target.  The RFD3 denoiser call and EDM integration remain unchanged, and the
previous sampler helpers remain available as compatibility paths for legacy
symmetry mode.  Runtime phase counters are emitted with result diagnostics so
LRZ golden tests can prove that all exact stages traversed the new boundary.
This slice remains local until the complete LRZ unit suite and C3 golden replay
pass.

### Phase 4: execution and operational usability

- provide local and Slurm executors over the same compiled plan;
- add job arrays, resume, structured failure reasons and status discovery;
- replace normal-use shell scripts with generated execution plans;
- retain historical scripts as regression or study fixtures;
- add installation, wheel and CLI smoke tests.

Exit gate: users can run a design without writing an sbatch file or editing a
cluster-specific Python module.

The first Phase 4 product-shell slice is implemented locally. `run` is the
canonical submit entry point while `submit` remains compatible. `status`
resolves either a run directory, a submission receipt or a numeric Slurm JobID
and joins scheduler state with worker state, declared audits and output
artifacts. `report` writes one self-contained HTML dashboard plus its canonical
JSON payload. Scientific success is fail-closed: scheduler completion alone is
never promoted to `passed`. This slice still requires LRZ unit validation
before it is considered available in the synchronized checkout.

The second Phase 4 slice adds an explicit executor interface and a persistent
per-JobID index at `OUTPUT_ROOT/.rfd3-mosaic/jobs`. Slurm submission now uses
`sbatch --parsable`, validates the returned JobID, writes a durable submission
record, and lets the allocated worker advance that record through running,
completed or failed. `runs` lists the index and `status JOB_ID --root ...`
checks it before its backward-compatible filesystem scan. Index failure is
reported but cannot corrupt or retroactively fail a scientifically valid run.
Only the Slurm executor is enabled in this slice; local execution remains a
declared next step rather than an untested switch.

### Phase 5: advanced assembly capabilities

- public `components + interfaces + connections` assembly graph;
- contig-inferred public intent for the two common workflows: supplied fixed
  interface plus generated linker, or fixed central motif plus automatically
  designed generated interface;
- simultaneous multi-orbit mobility;
- explicit radial, axial, tangential, tilt, twist and bounded SE(3) subspaces;
- Dn GPU closure;
- T/O/I registry support;
- high-order local-neighbourhood execution;
- cage-aware objectives and selection.

Each capability advances from experimental to engineering to stable only via
recorded unit, GPU and scientific validation gates.

The first assembly-graph slice is now implemented at the public-schema and
compiler level. It accepts an arbitrary number of rigid or joint-rigid
components, reusable component-owned interface ports, `preserve_input` or
contact interface edges, and directed generated chain connections. Several
differently oriented ports may belong to one joint-rigid building block, while
each interface edge names its own symmetry-neighbour relation. These
declarations lower into the same fragments, motion groups, ports, interface
edges, generated segments and constraint plan used by the existing exact
runtime; no graph-specific sampler branch is introduced.
The first slice validates edge relations during static assembly preflight and
uses the existing final audits for component rigidity, motion bounds and
scaffold validity. The topology-neutral post-diffusion relation audit is now
implemented: the adapter freezes every concrete symmetry edge and its atom
mapping, and the worker gates `preserve_input` translation/rotation or declared
contact/distance constraints in `assembly_interface_relation_audit.json`.
Ordinary users do not choose contact counts or identify a packing patch. The
compiler emits an automatic quality contract and the runtime derives balanced
residue coverage and contiguous-patch targets from available generated chain
length. Explicit graph edges and thresholds are reserved for advanced
multi-face cages and reproducible ablations.

The runtime is also converging on the same single-path contract. An
input-stage `preserve_input` edge remains authoritative in the unified hard
projector. An output-stage `contact` edge automatically activates a bounded
graph-interface field in that same timestep loop; compiler-expanded neighbour
instances determine which generated chains interact, while fixed motif atoms
remain untouched and exact symmetry is reprojected after the update. The final
audit evaluates output-stage relations on generated heavy atoms rather than
the fixed port atoms. This is an experimental CA-level controller until the
LRZ unit suite and a designed-interface GPU canary pass.

The second controller revision adds balanced per-side residue coverage,
contiguous-patch formation, source-interface-balanced aggregation,
residue-normalized clash energy and same-chain token-gradient smoothing. It
therefore cannot satisfy a many-contact request with one contacting residue,
and its repulsive signal does not vanish quadratically as the assembly grows.
These terms remain part of the common sampler lifecycle and common runtime
audit; they are not a topology-specific execution branch.

The fourth diagnostics revision adds topology-neutral CA-level orientation,
contact-depth uniformity, adjacent-backbone protection and smooth
worst-interface pressure. This closes several failure modes of a pure contact
field: end-on approach, point protrusions, local token collapse and one good
interface masking another bad one. These are intentionally proxy objectives;
solvent burial, atomic shape complementarity, cavity analysis and hydrophobic
surface require an all-atom post-generation layer and must not be claimed from
the CA controller alone.

The fifth runtime revision aligns the optimization lifecycle with the final
quality gate. Required-interface guidance now retains a nonzero terminal
floor, uses one true sequence-contiguous patch for continuity, orientation and
shape, and performs a small bounded post-trajectory polish through the same
joint projector before finalization. Missing coverage or continuity scales the
gradient up to a declared fraction of the existing per-token trust region;
this changes neither the hard maximum step nor fixed/symmetry authority.
Automatic continuity is limited by the longest available generated segment in
both runtime and final audit, while explicit expert targets remain fail-closed.
Diagnostics schema v5 reports the actual final proxy state separately from
the execution audit so a controller that ran cannot be confused with an
interface that met its quality contract.

The static supplied-interface graph crossed its first GPU canary in T job
`5735772`. Promotion beyond `gpu_canary` still requires the full LRZ suite,
strict multi-component regression coverage and the new designed-interface
runtime canary.
The scope is intentionally one global finite symmetry action. Independent
stabilizers and mixed vertex/edge/face orbits require a later IR extension and
must not be simulated by topology-specific scripts. The static graph path is
`gpu_canary`; the output-stage designed-interface controller remains
`schema_only` pending its own LRZ tests and GPU evidence.

### Current next implementation order

The next product work is not another topology-specific run script:

1. harden ordinary inspection (separate contact patches, report observed
   component/interface incidence, freeze user size/multiplicity intent);
2. implement `SimpleCageIntentSpec -> ranked frozen UserDesignSpec`
   resolution over ownership, directed scaffold paths, symmetry neighbours
   and continuous poses;
3. promote the binary `InterfaceEdgeSpec` into a relation IR with two or more
   participants, while retaining binary YAML compatibility;
4. carry requested and realized physical multiplicity, stabilizer and coset
   provenance in the frozen AssemblySpecification and audits;
5. close three-or-more-interface GPU cases, then T dynamic, O and I;
6. add the sequence-design/refolding/ranking layer and release engineering.

Every resolved ordinary candidate must replay through the existing expert
compiler and produce the same AssemblySpecification. `inspect` or generic
group-order compatibility alone must never be presented as a solved cage.

The first part of item 2 is now implemented as
`simple_binary_cn_ring_v1`: exactly one pairwise preserve-exact seed is
converted into one joint-rigid component, two ports, one supplied-interface
edge and one adjacent-copy scaffold link. Direction/offset alternatives are
ranked and strict-replayed rather than hidden.

The next bounded part of item 2 is implemented locally as
`prepositioned_multi_binary_cn_v1`. It accepts several disjoint binary
preserve-exact seeds from one authoritative coordinate frame, requires
complete boundary backbone anchors, enumerates canonical path covers plus
chemical directions/closing seams/Cn windings, lowers them to ordinary expert
graphs, validates the expanded interface/unit topology, and then uses the
existing linker/clash/closure ranking and strict replay. This bridge is
experimental and `schema_only`; it has not yet crossed its complete LRZ and
real GPU evidence gates.

The distinction is non-negotiable: `PolymerPathCoverHypothesis` alone remains
`executable: false`, because it proves only that every seed side participates
in one alternating combinatorial cycle. The pre-positioned bridge becomes
executable only for the restricted Cn case by adding input-contact evidence,
backbone anchors, explicit symmetry winding, `UserDesignSpec` lowering,
expanded topology validation and strict replay. It does not optimize radius,
orientation, tilt or axial pose, prove homomer equivalence, lower hyperedges,
or cover Dn/T/O/I and stabilizer/coset architectures.

### 70% gate for pre-positioned multi-binary Cn resolution

This narrow capability may be called 70% engineering-complete only when:

1. the full LRZ suite passes from one frozen source snapshot;
2. at least one real two-seed input completes `inspect -> plan -> resolve`
   with deterministic enumeration, no partial candidate truncation, explicit
   `automatic_selection: false`, and zero replay failures among advertised
   runnable YAML files;
3. a chosen YAML passes public validation, runtime-feature prevalidation,
   expanded interface/unit topology, linker, clash and group-closure gates;
4. a newly rendered 50-step V100/P100 run passes every required fixed-seed,
   symmetry, continuity and scaffold audit, followed by a second input or Cn
   order without source-specific code; and
5. manifests, CLI output and documentation retain the word
   **pre-positioned** and list the unsupported pose/search semantics.

Crossing this gate does not promote general multi-interface cage solving.
The next work remains continuous multi-seed pose search, component-type and
homomer/heteromer inference, relation hyperedges, stabilizer/coset orbits,
Dn/T/O/I execution and downstream sequence/refolding validation.

## Historical first implementation order

The original productization slices deliberately did not modify sampler
behavior; that phase is complete, and the unified output-stage interface field
described above is now the active runtime-development boundary:

1. machine-readable Foundry compatibility manifest: implemented;
2. provisional validated-snapshot record: implemented, exact snapshot replay
   still pending;
3. Git and runtime provenance: implemented;
4. fail-closed hashes for source state, runtime inputs and checkpoint:
   implemented and LRZ unit-validated;
5. immutable source snapshot for queued work: implemented locally, LRZ
   validation pending;
6. strict `UserDesignSpec`: begin only after the identity layer passes.

This order protects the working scientific engine before reorganizing its
public API or internal runtime boundaries.
