# RFD3-Mosaic Productization and Foundry Fork Plan

## Purpose

RFD3-Mosaic is a symmetry-aware protein-assembly diffusion engine built on a
maintained RFD3 fork. The project is not merely an external wrapper: exact
constraint restoration, coupled symmetry noise, orbit-state projection,
bounded motif motion and memory-aware symmetry execution require access to the
RFD3 timestep loop.

The goal is therefore not to remove the RFD3 modifications. The goal is to
turn them into a controlled, testable and upgradeable Mosaic-RFD3 engine while
providing one simple user-facing design interface above it.

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
design.yaml / CLI overrides
        |
        v
UserDesignSpec
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
rfd3-mosaic evaluate RUN_ID
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
- implement `plan`, `run`, `status` and `evaluate`;
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

### Phase 4: execution and operational usability

- provide local and Slurm executors over the same compiled plan;
- add job arrays, resume, structured failure reasons and status discovery;
- replace normal-use shell scripts with generated execution plans;
- retain historical scripts as regression or study fixtures;
- add installation, wheel and CLI smoke tests.

Exit gate: users can run a design without writing an sbatch file or editing a
cluster-specific Python module.

### Phase 5: advanced assembly capabilities

- simultaneous multi-orbit mobility;
- explicit radial, axial, tangential, tilt, twist and bounded SE(3) subspaces;
- Dn GPU closure;
- T/O/I registry support;
- high-order local-neighbourhood execution;
- cage-aware objectives and selection.

Each capability advances from experimental to engineering to stable only via
recorded unit, GPU and scientific validation gates.

## Immediate work order

The first implementation slices deliberately do not modify sampler behavior:

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
