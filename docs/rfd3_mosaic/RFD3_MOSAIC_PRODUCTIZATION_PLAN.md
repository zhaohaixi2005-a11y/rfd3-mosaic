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
  - fixed_xyz:
      selector: A12-20,A26-37
      atoms: all
      orbit: complete
  - orientation_cone:
      selector: A40-55
      axis: symmetry
      maximum_angle_degrees: 15
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

### Phase 2: unified ConstraintPlan

- compile existing fixed motifs into `fixed_xyz` operators;
- give every constraint an explicit reference frame and orbit scope;
- separate hard, bounded and guidance semantics;
- detect conflicts by constrained degree of freedom;
- derive audit requirements from constraints;
- retain topology presets only as frontend conveniences.

Exit gate: the new plan is behaviorally equivalent to the current exact path
for fixed motifs and can represent a design with no motif constraint.

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

The first implementation slice deliberately does not modify sampler behavior:

1. add the machine-readable Foundry compatibility manifest;
2. add the provisional validated-snapshot record;
3. capture Git and runtime provenance automatically;
4. ignore local visualization artifacts;
5. add provenance regression tests;
6. replay the full suite on LRZ;
7. begin `UserDesignSpec` only after this identity layer passes.

This order protects the working scientific engine before reorganizing its
public API or internal runtime boundaries.
