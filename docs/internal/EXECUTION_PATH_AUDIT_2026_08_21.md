# RFD3-Mosaic execution-path and version audit (2026-08-21)

This document records an end-to-end audit of the maintained
`refactor/product-core-v1` line.  Its purpose is to prevent results from
different frontends, historical scripts, commits or scientific screens from
being interpreted as one execution path.

Repository and run retention rules are defined in
`docs/internal/VERSION_RETENTION_POLICY.md`.

## Audited revision and verification

- local branch: `refactor/product-core-v1`
- audited base revision: `87f3b3a`
- personal remote: `origin/refactor/product-core-v1`
- lab remote: `lab/hx/rfd3-mosaic-product-core`
- all three revision pointers agreed at the start of the audit
- CPU regression suite: 905 tests passed
- public documentation boundary: 12 documents passed
- GitHub Actions run `32475145212`: every step passed, including CPU
  contracts, public-document checks and wheel smoke installation

The local shell initially had `DEBUG=release`. Foundry accepts only boolean
values for that environment variable, so the first test invocation produced
environment-driven errors. With the documented boolean environment
(`DEBUG=false`, `TYPE_CHECK=false`, `NAN_CHECK=true`), the complete suite
passed. This is shell contamination, not a second Mosaic code path.

## One maintained execution spine

The maintained path is:

```text
public UserDesignSpec
        or
simple_cage_intent -> resolver -> strictly replayed UserDesignSpec
        |
        v
constraint/topology lowering
        |
        v
AssemblySpecification
        |
        v
compile_assembly_rfd3_input
        |
        v
per-design pose materialization + one multi-example RFD3 input
        |
        v
RFD3 sampler/runtime constraints
        |
        v
raw CIFs + provenance + audits + advisory report
```

`run` and `submit` are aliases. Local and Slurm executors change where the
same frozen worker is launched; they do not select different compilers.
Ordinary-intent replay, normal public designs, and compatibility frontends
ultimately converge on the same Assembly IR compiler and experiment worker.

Important source locations:

- CLI dispatch: `src/rfd3_mosaic/cli.py`
- shared assembly compiler: `src/rfd3_mosaic/assembly_compiler.py`
- constraint lowering: `src/rfd3_mosaic/design_compiler.py`
- public-to-RFD3 adapter: `src/rfd3_mosaic/output/rfd3_adapter.py`
- pose population and seeds: `src/rfd3_mosaic/sampling_plan.py`
- frozen execution worker: `src/rfd3_mosaic/experiment_worker.py`
- RFD3 runtime sampler: `models/rfd3/src/rfd3/model/inference_sampler.py`
- result audits and advisory screening:
  `src/rfd3_mosaic/result_auditing.py` and
  `src/rfd3_mosaic/advisory_screening.py`

## Legitimately different user paths

### Standard public design

`rfd3-mosaic run design.yaml` validates the public schema, lowers it once,
materializes the requested design population, freezes source/provenance and
executes it. This is the normal product path.

### Ordinary simple intent

A `simple_cage_intent` does not go directly to RFD3. It first enumerates the
unknown topology/connection choices, requires a strictly replayable selected
public YAML, and then dispatches that YAML through the normal product path.
This search is necessary because the user's topology is incomplete. It is not
an automatic aesthetic ranking stage applied to every normal design.

### Explicit `resolve` and `search`

These commands are opt-in tools for under-specified topology, unknown
relative seed poses, or expert candidate enumeration. Their CPU objectives
are feasibility/ranking proxies. They are not a universal definition of a
biologically good backbone and therefore are deliberately not inserted into
every `run designs=N` call.

### `central` and `interface`

These are compatibility convenience frontends. They create an internal
experiment request and still converge on the normal compiler/worker. They are
not the preferred way to describe new public designs.

## `designs=N` and pose diversity

Current semantics are not "one assembly pose copied N times" for a stochastic
pose declaration.

- A variable `initial_pose` receives one independently derived pose seed per
  design, subject only to hard feasibility rejection.
- Each design receives an independently derived diffusion seed.
- Pose-specific RFD3 examples are merged into one multi-example JSON, so the
  checkpoint is loaded once per shard.
- The worker writes `pose_manifest.json` with seeds, realized input digests,
  retry history and `selection_method=seeded_hard_feasibility_rejection`.
- RFD3 reseeds both stochastic input transforms and sampling from the
  per-example `mosaic_diffusion_seed`.

For a completely locked/fixed input there is intentionally one physical pose
and multiple diffusion trajectories. For a stochastic/mobile declaration,
there are independent feasible assembly poses. Optional QD/search remains
explicit because its proxy score is not a ground-truth quality oracle.

## Fixed geometry, relative mobility and packing are separate layers

The current schema/compiler makes the following distinctions:

1. `preserve_supplied_geometry` or locked arrangement fixes both internal
   motif geometry and the declared assembly arrangement.
2. A supplied interface with joint-rigid geometry fixes every atom-to-atom
   relation inside that interface. If its component is guided/free, the whole
   interface seed may translate and rotate as one rigid body; its two sides
   cannot separate or deform.
3. Locked generated-interface packing keeps the selected initial motif pose
   fixed during diffusion and guides only generated regions.
4. Guided generated-interface packing begins from the matched selected pose
   and additionally permits bounded radial/axial/rotational component motion.
5. `scaffold_packing: symmetric_generated` is only for creating a new
   generated interface. It is rejected for a preserve-supplied-geometry task.
6. Intra-chain and inter-chain guidance weights are passed independently of
   whether a graph/generated-interface controller is active.

Graph-interface guidance and automatic symmetric-generated packing cannot be
active simultaneously. Their mutual exclusion is checked before RFD3.
Runtime audit selection is derived from compiled Assembly IR, not inferred a
second time from the original YAML.

## Version and provenance isolation

Every submitted run freezes:

- resolved configuration and its digest;
- source snapshot under the run's `software/` directory;
- repository branch/commit and dirty-state identity;
- dependency identities;
- checkpoint path and hash;
- pose and diffusion seed provenance.

The worker verifies render/source identity before inference. Therefore an old
job does not silently import today's checkout halfway through execution.
Multiple commits can coexist in the run root, but they remain distinct frozen
runs. Confusion happens when results from different campaign manifests or
commits are compared without reading their provenance, not because a single
run combines those versions.

## Confirmed issue repaired by this audit

One real dispatch inconsistency was found: a one-command
`simple_cage_intent` replay did not forward `--defer-runtime-preflight` into
the selected public design. A high-order O/I simple intent could therefore
attempt complete feature construction on a constrained login node, despite
the user's request to defer it to the allocated worker.

The CLI now forwards the flag, and the one-command replay regression test
checks it. Complete RFD3 prevalidation remains mandatory; only its execution
location changes.

## Historical scripts: the main source of operational confusion

The repository retains old research scripts for provenance and comparison.
Several call `rfd3.run_inference` or the adapter directly and therefore bypass
parts of the current public path, such as ordinary schema lowering, source
snapshot generation, campaign indexing or current reporting. They must not be
used as evidence for the current product contract.

The exact maintained and historical script classification is recorded in
`scripts/rfd3_mosaic/README.md`. Direct historical scripts and their wrappers
are physically isolated below `scripts/rfd3_mosaic/archive/legacy_direct/`.
Current release-gate configuration is defined by `GATES` in
`scripts/rfd3_mosaic/submit_gpu_release_gates.py`, not by every YAML whose
filename contains `canary` or `smoke`.

The `experiments/` directory similarly contains current gate inputs alongside
development canaries. `experiments/README.md` records the current
source-of-truth set; directly superseded inputs live below
`experiments/archive/superseded/`. Historical inputs remain tracked because
deleting them would make old run provenance and bug regressions harder to
understand.

## Output status versus scientific advice

A generated CIF is retained even when advisory scientific targets are not
met. Current reporting separates:

- execution/generated status;
- hard geometry contracts (fixed geometry, seed integrity, symmetry and
  mobility bounds);
- advisory scaffold/interface metrics.

Older JSON fields and collectors may still use compatibility names such as
`passed`, `accepted`, `rejected` or the phrase "required result audits". Those
legacy names are an interpretation risk. The current user-facing status uses
`GENERATED` plus contract/advisory details and does not claim to know whether
the user likes a generated backbone.

## What remains genuinely open

The audit found no unresolved CPU wiring contradiction among fixed geometry,
joint-rigid mobility, per-design poses, multi-example RFD3 execution and
reporting. The remaining major blockers are evidence and scientific quality,
not parallel software paths:

- 50-step icosahedral fixed-orbit runtime closure;
- corrected independent-pose locked/guided generated-interface GPU evidence;
- calibration of broad, continuous generated interfaces;
- broader real-input multi-seed GPU validation.

Automatic symmetry/topology inference, cooperative functional-site design,
unrestricted quotient-edge multiplicities and a fully general native polymer
path language are outside the current product scope. They are not release
blockers and must not be included in completion estimates. Existing validated
subsets remain available and continue to fail closed outside their executable
domain.

These open items must not be described as already validated. Conversely, old
10-step failures and direct-script outputs must not overwrite claims proved by
newer frozen 50-step runs.

## Operator rules

1. Start new work with `rfd3-mosaic run`/`submit`, or one of the three
   maintained campaign launchers.
2. Identify a result by campaign manifest, run directory and repository
   commit, not only by job number or filename.
3. Read `pose_manifest.json` before claiming pose diversity.
4. Treat `input/presymmetrized_input.cif` as compiled pre-diffusion input and
   `*_model_0.cif(.gz)` as generated output.
5. Do not rerun a historical direct `.sbatch` script to validate the current
   product.
6. Keep all raw CIFs; use advisory screens for comparison, not destructive
   deletion or universal biological acceptance.
7. Use boolean `DEBUG`, `TYPE_CHECK` and `NAN_CHECK` values.
