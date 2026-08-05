# RFD3 Mosaic Development Status

Last updated: 2026-08-04

This file is the persistent project memory for resuming development after a
new login or a new Codex session. Update it whenever a milestone changes.

## Project identity

- Repository: `zhaohaixi2005-a11y/rfd3-mosaic`
- Upstream: `RosettaCommons/foundry`
- Active branch: `feat/interface-seed-compiler`
- Server working tree: `/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic`
- Local mirror: `/home/haixi/Documents/mosaic`
- Goal: build a generator-independent `AssemblySpecification` compiler and a
  constraint-aware RFD3 symmetry sampler in which fixed or bounded-mobile
  motif orbits, generated segments and exact group actions are updated through
  one runtime contract. LHD101 and individual Cn/Dn runs are regression
  fixtures, not product-specific code paths.

## 2026-08-04 architecture migration

The architecture has been reset around three planes:

```text
compile:   user intent -> AssemblySpecification -> RFD3 runtime features
infer:     (assembly state X_t, orbit poses G_t) -> denoise/control/project
evaluate:  motif/symmetry/scaffold/assembly audits -> acceptance gate
```

The backwards-compatible migration now has three completed local slices:

- `AssemblySpecification` is now the topology-neutral public schema name;
- `InterfaceSeedSpec` remains an alias, so existing configs and imports keep
  working;
- `load_assembly_config` accepts new `assembly:`, legacy `interface_seed:`, or
  an unwrapped payload and rejects ambiguous dual wrappers;
- ports and interface edges may be empty for non-interface motifs;
- `generated_segments` can represent N/C terminal extensions using the same
  assembly schema used by between-motif scaffold links;
- central and interface compatibility frontends now return one immutable
  `CompiledAssembly` artifact, so prevalidation and inference no longer branch
  on how the input was compiled;
- each `CompiledAssembly` now carries immutable, topology-neutral
  `CompiledAudit` command descriptions; the experiment worker no longer
  branches on central motif versus interface seed during evaluation;
- the runtime public API is now `ConstraintGroup`, `ConstraintOrbit` and
  `ConstraintOrbitLayout`; the original `InterfaceConstraint*` names remain
  compatibility aliases, and the motif-mobility controller consumes the
  neutral API;
- `UnifiedJointProjector` now owns the ordered runtime contract
  `symmetry projection -> complete constraint restore -> closure validation`;
  exact Euler updates, fixed-motif projection and final exact projection are
  wired through it while the previous sampler methods remain compatibility
  wrappers;
- mobility is now a property of the topology-neutral symmetry orbit contract,
  not intrinsically of an interface edge. `OrbitMobilitySpec` declares the
  allowed subspace (`radial`, `radial_axial`, `tilt_only`, `bounded_se3`),
  cumulative bounds, timestep window, per-step translation/rotation trust
  region, proposal source and objective references;
- `ConstraintOrbitInstance` is now part of the compiled assembly IR. Legacy
  non-fixed interface mobility is migrated into that IR with conflict checks;
  the adapter reads mobility from the compiled orbit rather than directly
  from an interface edge;
- `hoyeung_drag_compat` is represented only as one optional proposal source.
  It does not define the core state model; the formal state remains one
  bounded rigid `SE(3)` pose per master constraint orbit;
- the experiment compiler no longer calls separate central-probe and
  interface-seed builders. Legacy/simple frontends first lower to one
  `AssemblySpecification`; one adapter then expands `CompiledInstanceSet`,
  consumes either `ScaffoldLinkInstance` or `TerminalExtensionInstance`, and
  emits the same native RFD3 feature contract;
- `rfd3_central_motif_probe` remains only as a diagnostic compatibility tool
  and regression fixture. It is no longer the central-motif backend used by
  `compile_experiment_assembly`;
- this third slice has passed local syntax and diff checks. After the latest
  synchronization, the complete LRZ unit suite passed **318/318 tests in
  10.666 seconds**. Native prevalidation plus one central and one interface
  GPU regression through this new compiler path are still required before
  calling the compiler migration end-to-end validated.

The next code milestone is not another topology-specific script. It is:

```text
validate native AssemblySpecification lowering on LRZ
-> move all remaining initialization/finalization projection sites through
   UnifiedJointProjector
-> multi-orbit objective aggregation
```

The existing `OrbitRigidMotifController` already carries bounded translation
and SO(3) rotation and refreshes conditioning inside the RFD3 timestep loop.
It remains experimental because scaffold-derived guidance currently assumes
one cyclic orbit and uses a ring-specific axis/tilt objective. Generalization
requires multi-orbit simultaneous residual aggregation, optional DOF
subspaces, objective composition and Cn/Dn/T/O/I transform-registry support.

## Current project state

| Capability | Current evidence | Status |
|---|---|---|
| Interface-seed compiler | Generic fragment, topology, Cn/Dn registry, pose provenance, adapter and prevalidation paths are implemented and unit-tested | Implemented |
| Static exact-C3 RFD3 | 50/100/200-step inference preserves the complete cross-chain seed, continuous copies and declared C3 symmetry | End-to-end demonstrated |
| Central-motif exact C3 | Registry-v4 runs 5729451--5729453 passed the central-orbit and scaffold gates with no breaks or CA clashes and sub-0.00005 A symmetry maximum error | End-to-end demonstrated on the compatibility frontend; unified-compiler GPU regression pending |
| C5/C6/C7 static RFD3 | A 48-structure extracted set produced 37 strict seed+scaffold passes across all three orders | Cross-order engineering generalization demonstrated on the screened set |
| Pose exploration | Haar SO(3), joint LHS, topology-aware endpoint descriptors and quality-diversity morphology coverage are implemented | Implemented; still a GPU-budget selector, not a biological score |
| Scaffold-aware motif mobility | Default-off bounded controller and conditioning refresh path exist; the boundary-representation correction is included in the 318-test LRZ pass, but no applied-mobility GPU result has passed the full audit gate | Unit-validated; GPU/scientific behavior not validated |
| Multiple interface orbits | The adapter can emit multiple disjoint ASU scaffold segments and multiple constraint orbits; the D3 two-orbit CPU build/prevalidation passed, but its P100 GPU attempt exceeded memory | CPU plumbing demonstrated; GPU inference pending |
| Dn inference | Schema, registry and D3 two-orbit CPU prevalidation pass, but no native Dn GPU end-to-end result has passed | Not demonstrated on GPU |
| C12/C20 | Artificial order-10 guards are removed and the local-neighbourhood kernel is unit-tested, but explicit preprocessing/output and quadratic pair-state costs remain; no explicit/local GPU equivalence run has passed | Outside the validated production architecture |
| Sequence/design validation | ProteinMPNN, multimer prediction, interface-energy ranking and experimental validation are not part of the completed evidence | Not started |

The defensible current claim is:

> RFD3 Mosaic can compile a cross-protomer interface seed and preserve it
> while generating a continuous, exact-symmetry cyclic scaffold. C3 has been
> demonstrated end to end. In the first audited C5/C6/C7 set, 37/48
> structures passed the same strict seed and scaffold gates. This supports
> engineering generalization across C5--C7, but does not yet establish
> sequence-level designability or experimental assembly.

The extracted-structure screen is:

```text
scripts/rfd3_mosaic/screen_extracted_cn_structures.sh
    -> src/rfd3_mosaic/rfd3_batch_screen.py
    -> C5/C6/C7 JSON + CSV reports
```

Its acceptance order is deliberately strict:

```text
seed integrity
-> chain continuity and compactness
-> zero hard CA clashes
-> declared-transform Cn symmetry
-> only then compare ring shape and neighbouring-chain packing
```

Ring radius, axis clearance, shape aspect and contact counts are diagnostics
for diversity and ranking. They are not calibrated folding, assembly or
experimental-success scores.

### 2026-07-31 C5/C6/C7 extracted-structure screen

The first complete batch screen used:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c5_full/extracted_cif
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c6_full/extracted_cif
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c7_full/extracted_cif
```

Results:

| Order | Screened | Seed passed | Continuous | Zero CA clash | Declared symmetry | Strict passed |
|---|---:|---:|---:|---:|---:|---:|
| C5 | 15 | 15 | 15 | 12 | 15 | 12 (80.0%) |
| C6 | 18 | 18 | 17 | 13 | 18 | 13 (72.2%) |
| C7 | 15 | 15 | 14 | 12 | 15 | 12 (80.0%) |
| Total | 48 | 48 | 46 | 37 | 48 | 37 (77.1%) |

This establishes that seed preservation and declared cyclic symmetry are no
longer C3-only behavior in the screened runs. The dominant rejection is local
CA clash rather than loss of the interface seed. Two structures also fail
continuity.

Initial candidates for downstream inspection are:

| Order | Recommended job | Reason |
|---|---:|---|
| C5 | `5722385` | Strict pass, 127 minimum neighbouring contacts and 4.055 A minimum inter-chain CA distance |
| C6 | `5722398` or `5722341` | Safer 3.940/4.223 A minimum distance than the contact-ranked jobs `5722400` and `5722401`, which lie close to the 3 A hard-clash boundary |
| C7 | `5722413` or `5722344` | Strict pass with 71/66 minimum neighbouring contacts and 3.987/4.223 A minimum distance |

Contact-ranked C6 jobs `5722400` and `5722401` remain valid strict passes, but
their 3.006 and 3.229 A minimum inter-chain CA distances make them dense
packing controls rather than automatic first choices. The screened shape
aspect ratios are broadly ring-like and relatively flat; they do not by
themselves demonstrate cage-like three-dimensional closure.

The next scientific gate is not more CIF geometry ranking. It is to take a
small, shape-diverse strict-pass shortlist through sequence design, multimer
prediction and atom-level interface/clash assessment while retaining failed
and borderline candidates as negative controls.

### Core RFD3 method-development direction (prospective)

Benchmark expansion is deferred while the tool is still being built. The
primary development objective is no longer another task-specific sampler
patch. The current hypothesis is an RFD3-native interface-orbit sampler with
two coupled runtime states:

```text
X_t = complete all-atom scaffold/assembly diffusion state
g_t = one bounded, symmetry-reduced pose state per mobile interface orbit
```

The target per-step contract is:

```text
denoise the complete assembly
-> aggregate copy-wise scaffold signals in the master frame
-> infer one bounded, copy-equivariant update for g_t
-> preserve immutable cross-protomer interface cores
-> expand every master pose through the declared group actions
-> refresh all motif and constraint conditioning
-> project X_t back to exact group closure
-> continue denoising
```

This is currently best described as **symmetry-constrained alternating
inference**, not joint diffusion. No forward noise process, reverse transition
or learned score has yet been defined for `g_t`.

The first model-context audit is complete. The symmetry sampler passes the
full length-`L` noisy assembly directly to `diffusion_module`; symmetry is not
implemented by denoising an isolated ASU and copying it only after the network.
Both full and chunked pair representations are built over that assembly.
However, the `>3`-chain sparse-attention helper incorrectly iterated over
unique chain IDs as if they were query-atom indices, so its reserved
inter-chain keys covered only a few rows. The local fix now constructs reserved
inter-chain neighbours for every query atom and keeps small attention budgets
non-negative. The new targeted tests and the full LRZ unit suite pass. A paired
C5 GPU rerun is still required before claiming any improvement in generated
inter-subunit packing.

The implementation milestones are deliberately incremental:

1. Promote the existing constraint-orbit tensors into a validated, first-class
   RFD3 `InterfaceConstraintOrbit` runtime object. **Implemented and unit-test
   validated.**
2. Make the existing mobile-pose controller consume that object while
   preserving backwards compatibility with current emitted features.
   **Implemented and unit-test validated; static restoration remains on its
   legacy-compatible path until a separate change is justified.**
3. Audit the true RFD3 model context and identify gauge freedoms before
   choosing the pose parameterization. A raw six-degree-of-freedom `SE(3)`
   state must not be adopted if axial translation or rotation only changes the
   world frame or copy labels. **Full-assembly entry is confirmed; gauge
   analysis remains open.**
4. Treat interface pose as an explicit, bounded state with a copy-equivariant
   proposal: evaluate all equivalent copies, inverse-map their signals to the
   master frame, aggregate once, and regenerate copies only by group actions.
5. Add hierarchical roles only after the pose state is sound:
   immutable `interface_core`, orbit-following `rigid_support`, and a small
   time-scheduled `flexible_boundary` next to generated scaffold.
6. Generalize from one cyclic orbit to multiple non-equivalent Cn/Dn
   interface orbits with simultaneous, order-independent conflict handling.

Moving a complete interface orbit does **not** optimize the interface core's
internal packing: its two sides retain their validated relative geometry. It
can only improve junctions to generated scaffold, seed-external contacts,
clashes and global morphology. Weak or symmetry-incompatible seeds still need
to be rejected upstream.

This is a prospective method contribution, not a validated project claim.
More Cn samples, more penalties,
or direct reproduction of RFdiffusion1 motif dragging are useful validation or
engineering work but are not by themselves the core innovation. Large
benchmarks, sequence design and refolding remain required evidence later; they
are not the immediate construction priority.

### Target symmetry scope and scalable backends

The target is not limited to C3--C7. The intended finite-group scope is
`Cn`, `Dn`, `T`, `O`, and `I`; `H` is treated here as helical/screw symmetry
and requires a finite runtime neighbourhood rather than a finite group
multiplicity.

Current limitations are distinct and must not be treated as one unchecked
constant change:

- The local branch now removes both artificial 10-transform guards: the
  Mosaic adapter accepts C12/D6, and Foundry motif-frame recovery no longer
  rejects more than 10 transforms. The C12/D6 closure, adapter and frame
  regressions have passed on LRZ.
- Foundry's named-frame parser recognizes only `Cn`, `Dn`, and
  `input_defined`.
- Mosaic's typed symmetry schema currently enumerates only cyclic and
  dihedral transform sets.
- Explicit all-copy atom state grows linearly with copy count, while token
  pair state and several audits can grow quadratically; `O` (24 copies), `I`
  (60 copies), high-order `Cn/Dn`, and long helical windows therefore cannot
  be enabled responsibly by deleting the guards.

Development should separate two execution backends behind the same
interface-orbit contract:

1. `explicit_all_copy`: exact current representation for small finite groups;
   retain as the reference implementation and numerical oracle.
2. `local_symmetry_neighbourhood`: keep one independent master ASU/orbit,
   materialize only symmetry copies that can interact with it, denoise that
   local assembly context, inverse-map copy-wise updates to the master frame,
   aggregate once, and expand the complete requested assembly only for exact
   projection/output/audit.

The first backend-independent local-neighbourhood kernel is now implemented in
`rfd3/inference/symmetry/local_neighbourhood.py`:

- Cn selects `master, -1, +1, ...` with network copy count bounded by the
  configured neighbour radius rather than group order. C200 with radius one
  therefore exposes three copies to the future network view.
- Dn can select the same local cyclic neighbourhood in both dihedral cosets.
- explicit global-to-local atom maps reject incomplete transform coverage;
- local copy predictions are inverse-transformed, averaged in canonical orbit
  coordinates, and expanded to every global copy. Omitted global copies never
  dilute the denoiser update.

The same module now also constructs a fail-closed local feature view: it keeps
whole atomized tokens, reindexes `atom_to_token_map`, crops atom/token and pair
features together, and handles Mosaic motif-constraint atom axes explicitly.
The standalone C12/C20/C200/D100 neighbourhood, feature-crop, coordinate
expansion and sequence-logit expansion tests have passed on LRZ.

An experimental integration is now wired before `TokenInitializer` and into
`SampleDiffusionWithSymmetry`. It is selected only by
`symmetry_execution_backend=local_neighbourhood`, requires low-memory mode,
exact orbit-average state, coupled noise and fixed-motif preservation, and
currently rejects dynamic motif mobility. The default remains
`explicit_all_copy`. On 2026-07-31 the complete LRZ unit suite, including the
new sampler integration and fail-closed configuration tests, passed 273/273 in
9.189 seconds. A real small-order explicit/local A/B run remains pending, so no
production script enables the backend by default. In addition, preprocessing
still constructs the complete assembly before this crop and final output still
expands every copy; this integration bounds the initializer/denoiser network
view but is not yet an end-to-end C200 memory guarantee.

The staged order is:

1. remove task-script assumptions such as `C5|C6|C7` while retaining safe
   resource guards;
2. validate parameterized high-order `Cn` and `Dn` registries, closure,
   provenance, attention neighbourhoods, and audits on CPU;
3. add proper finite `T/O/I` transform registries and generic finite-group
   prevalidation;
4. validate the experimental local-neighbourhood sampler integration and
   demonstrate agreement with explicit all-copy results on small groups before
   using it for C12+, O or I;
5. add helical screw transforms, a declared repeat window and boundary-aware
   audits as a separate infinite-symmetry mode.

The former `>10` guards have been removed locally, but production high-order
submission remains blocked by staged CPU construction, bounded GPU memory and
scientific audit gates rather than by a hard-coded order constant.

## Maintained documentation

Only the following five documents are active and should receive future
updates:

1. `DEVELOPMENT_STATUS.md` — single source of truth for current evidence,
   limitations, cluster-operation boundary and resume point.
2. `docs/rfd3_mosaic/RFD3_MULTI_INTERFACE_SEED_FINAL_PLAN.md` — stable method
   architecture, data model, invariants and success criteria.
3. `docs/rfd3_mosaic/C5_C6_C7_200STEP_RUNBOOK.md` — executable pose,
   inference, audit and extracted-structure screening workflow.
4. `docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md` — the only
   active experimental-method note; it must stay separate from the validated
   static claim.
5. `docs/rfd3_mosaic/USER_CLI.md` — concise public experiment configuration,
   execution-profile and submission contract.

One additional document is retained but not actively maintained:

- `docs/rfd3_mosaic/INTERFACE_SEED_RFD1_UPGRADE_AUDIT.md` — read-only
  historical code audit used to compare Interface-Seed 1.0 with this project.

Do not create another progress handoff, evolution plan or duplicate runbook.
Put new project evidence in this file, stable architectural decisions in the
final plan, executable C5/C6/C7 commands in the runbook, and mobility-only
results in the pilot note.

The superseded `RFD3_MULTI_INTERFACE_SEED_EVOLUTION_PLAN.md` was removed on
2026-07-31 after its still-relevant decisions had been incorporated into the
final plan and this status file.

## Project reading order

To understand the project without replaying the full development history, read
these files in order:

1. `DEVELOPMENT_STATUS.md` — current evidence, limitations, operational
   boundary, and exact resume point.
2. `docs/rfd3_mosaic/RFD3_MULTI_INTERFACE_SEED_FINAL_PLAN.md` — method
   architecture, data model, compiler/runtime separation, and success criteria.
3. `docs/rfd3_mosaic/USER_CLI.md` — routine validated configuration and
   submission workflow.
4. `docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md` — the current
   opt-in experiment that allows bounded scaffold-guided seed motion.
5. `docs/rfd3_mosaic/C5_C6_C7_200STEP_RUNBOOK.md` — reproducible C5/C6/C7
   pose generation, P100 inference, and audit commands.

For historical comparison with Interface-Seed 1.0, then read
`docs/rfd3_mosaic/INTERFACE_SEED_RFD1_UPGRADE_AUDIT.md`.

## Environment contract

The shared `rc-foundry` environment must not be modified with editable
installs. Activate it and point Python at this checkout:

```bash
source ~/software_paths.sh
source "$SHARED_MAMBAFORGE/etc/profile.d/conda.sh"
conda activate "$RC_FOUNDRY_ENV"
cd /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:$PYTHONPATH"
export FOUNDRY_CHECKPOINT_DIRS=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints
```

Checkpoint used:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/foundry/checkpoints/rfd3_latest.ckpt
```

Cluster-operation boundary:

- Codex may modify local code, scripts, tests, and documentation, but cannot
  operate the LRZ server directly.
- The user performs every local-to-LRZ synchronization, `sbatch` submission,
  job cancellation, and GitHub push.
- After every local file change, Codex must provide directly executable
  synchronization commands and the necessary server-side verification
  commands.
- Documentation-only changes do not require a standalone LRZ synchronization.
  They remain pending locally and travel with the next code/script sync batch;
  Codex must identify them as pending instead of repeatedly asking the user to
  sync documentation by itself.
- Providing a command must never be described as having synchronized,
  submitted, cancelled, pushed, or executed it.

## 2026-07-30 end-to-end milestone achieved

The static exact-C3 Interface-Seed pipeline has now completed end to end on
LRZ, including compilation, deterministic linker materialization, native RFD3
input construction, runtime prevalidation, checkpoint inference, result
serialization, seed-integrity auditing, transform-aware scaffold auditing, and
the final audit gate.

The principal 200-step result is job `5721371`:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/native_c3_full/5721371
```

Key result:

- all three cross-protomer seed pairs passed;
- maximum all-atom seed RMSD was `0.000511 A`;
- maximum CA seed RMSD was `0.000489 A`;
- atom completeness and contact retention were both `1.0`;
- all three 146-residue chains were continuous with zero chain breaks;
- compactness passed with maximum CA radius of gyration `23.460 A`;
- declared-transform C3 symmetry passed with coordinate RMSD `0.000119 A`
  and maximum coordinate error `0.000236 A`;
- copy-internal distance-matrix RMSD was `6.18e-6 A`.

This establishes the end-to-end engineering result: the complete cross-chain
interface seed can remain fixed while RFD3 generates a continuous, compact,
exact-C3 scaffold around it. The strict scaffold report is a near-pass rather
than a clean final acceptance because one local generated-linker/right-motif
CA overlap (`2.253 A`) is reproduced in all three C3 copies, producing three
reported clashes. This remaining candidate-quality issue does not invalidate
the demonstrated end-to-end pipeline, but a clash-free pose or diffusion seed
is still required before calling a design scientifically final.

## 2026-07-29 exact orbit-state refactor

The previous hard-motif patches preserved the interface seed but did not keep
the generated scaffold in one C3-invariant state. The current local refactor
replaces that bridge with an explicit orbit-space path:

- Fixed and generated coordinates use the same row-vector convention as
  AtomArray expansion: `x_copy = x_master @ R.T + T`. The previous runtime
  projector used `R` instead of `R.T`; for C3 this silently exchanged the
  `r1/r2` direction even though the unordered copy set still looked symmetric.
- `symmetry_state_mode=orbit_average` inverse-maps every copy, averages all
  copies in the master frame, and re-expands them. Noise is sampled once per
  master atom and rotation-copied to the other transforms. Initial, noisy,
  denoised, Euler-updated, compactness-guided, and final states are all checked
  for orbit closure.
- Exact operations require atom-key-verified `sym_orbit_slot` correspondence,
  run outside bfloat16 autocast, and retain float32 state precision instead of
  quantizing a 1e-3 A closure check back to bfloat16.
- Runtime frames reconstructed from RFD3's epsilon-stabilized virtual-frame
  encoding are validated and projected to the nearest proper SO(3) rotation
  before orbit operations. Online closure retains the 1e-3 A absolute gate at
  molecular scale and adds a float32/coordinate-scale roundoff floor only at
  the very large initial EDM noise scale.
- Arbitrary per-step realignment is disabled because the stored symmetry
  operators are not conjugated into the augmented frame.
- Adapter output now embeds the actual registry matrices and explicit
  constraint-orbit actions. Prevalidation calls the real
  `DesignInputSpecification.build -> AddSymmetryFeats.forward` path, matches
  runtime matrices one-to-one to the registry, and evaluates the same
  all-copy orbit projector used during inference. Missing groups/orbits,
  malformed matrices, non-finite targets, mask non-closure, or target
  residuals above threshold fail before checkpoint loading.
- A continuous linker range is materialized once by the adapter as an exact
  `N-N` contig length. The default is the configured integer midpoint
  (`70--100 -> 85-85` for LHD101), and `RFD3_LINKER_LENGTH` can select another
  in-range value. This prevents prevalidation and the separate inference
  process from independently sampling different AtomArray lengths. The
  standalone endpoint spans are rechecked against that exact materialized
  length rather than only the configured maximum.
- Scaffold symmetry acceptance is transform-aware. Copy-internal CA distance
  matrices remain a diagnostic only; final acceptance compares each copy's CA
  coordinates with the declared C3 transform of the master chain. A translated
  or wrongly rotated but internally congruent copy therefore fails. The current
  output-chain association is validated for the one-chain-ASU C3 baseline; it
  still uses sorted output chain order and is not yet a provenance-aware
  multi-chain/Dn audit.
- Smoke and full scripts write both seed and scaffold reports, then a separate
  audit gate fails the job if either report is scientifically rejected.
- The 10-step smoke job runs the complete repository unit suite on its
  allocated compute node before adapter/prevalidation or checkpoint loading.
  This avoids the observed DSS-backed `torch` import stalls on LRZ login
  nodes and makes a unit failure stop the same job before GPU inference.
- Complete-interface `orbit_rigid` schema, atom correspondence, and bounded
  SE(3) controller are connected through an explicit, default-off experimental
  sampler hook. Formal scripts do not enable it and keep the interface static:
  moving coordinates without
  updating RFD3's precomputed motif pair-conditioning creates contradictory
  geometry, and raw fixed-atom denoiser output is not yet a validated scaffold
  pose signal.

Local `compileall`, Slurm shell parsing, and `git diff --check` pass. The local
base Python lacks `pydantic` and the RFD3 runtime, so the new suite and the real
C3 build-forward-prevalidation chain still require LRZ validation. Job
`5720844` is retained as historical evidence for the old sampler only; it
cannot validate the corrected transform direction or exact-state path.

The next gate is: targeted CPU tests -> complete CPU discovery -> adapter plus
prevalidation (including deterministic linker provenance) -> 10-step static
smoke -> 50-step static pose screen -> 200-step static validation. Dynamic
motif mobility remains blocked until its
conditioning and pose signal are designed and tested explicitly.

LRZ smoke validation update: the synchronized exact-path suite reached 216
tests; 214 passed and two regressions stopped the job before inference. One
was a real aliasing bug in the legacy atomwise projector (`Tensor.to()` can
return the input, so its in-place COM correction mutated the caller); the
projector now clones before centering. The other was a test harness error: the
full fake sampler test called an inference API that correctly requires
`torch.no_grad()` without that context. Both fixes are local and require one
fresh LRZ smoke rerun; no GPU/scaffold result from that stopped job is valid.

## 2026-07-29 group-aware constraints and symmetry diagnosis

- The adapter now compiles every concrete interface-edge instance into a
  two-sided motif constraint group and records both standalone atom indices
  and post-symmetry RFD3 source-component/transform membership.
- `AddSymmetryFeats` resolves these groups only after RFD3 has constructed and
  expanded the AtomArray. It requires both interface sides to match and every
  fixed motif atom to be covered.
- The symmetry sampler restores complete groups at every opted-in denoising
  step. Overlapping groups are order-independent when their coordinates agree
  and fail explicitly when hard constraints conflict.
- C3 entry scripts require group metadata. `rfd3_prevalidate` now resolves the
  same runtime memberships before checkpoint loading and reports group counts,
  sizes, and fixed-atom coverage.
- Symmetry projection now works out-of-place, verifies equal atom counts for
  ASU/target subunits, and transform-frame construction no longer assumes
  atoms are consecutively ordered by transform ID.
- Local compilation, shell parsing, and whitespace checks pass. LRZ
  `rc-foundry` validation subsequently passed all 168 unit tests on
  2026-07-29, including runtime cross-chain group resolution, transform-order
  independence, complete fixed-atom coverage, and conflicting-overlap
  rejection.
- LRZ adapter prevalidation then passed with three resolved C3 constraint
  groups of 496 atoms each, covering all 1488 fixed motif atoms, with no
  failures. One unguided 200-step LRZ run remains required; these CPU checks
  are not yet an end-to-end scientific result.
- The single-design entry point accepts `RFD3_NUM_TIMESTEPS` in the range
  2--200 while retaining 200 as its default. Shorter runs are diagnostic only;
  the next intermediate screen uses three geometrically distinct poses at
  50 steps before the definitive 200-step gate.
- Full 200-step job `5720844` passed all three interface-seed pairs
  (maximum CA RMSD 0.038 A, maximum all-atom RMSD 0.041 A, minimum contact
  retention 0.975, and complete atom coverage) but failed scaffold geometry.
  Chain A had only three continuity failures, whereas chains B and C each had
  94, mostly implausibly short generated-region bonds; the output also had
  3031 intra-chain and 613 inter-chain CA clashes. This chain-asymmetric
  collapse is incompatible with rigid C3 copies and points to a remaining
  ASU-to-copy atom/coordinate correspondence problem, not merely insufficient
  diffusion or a need for motif rigid-body mobility.
- Pairwise CA distance-matrix comparison confirmed the loss of rigid C3
  geometry (A--B RMSD 14.636 A; A--C RMSD 19.128 A). The concrete sampler
  cause is that upstream projects the denoised prediction, then combines it
  with independently noised coordinates in the Euler update; the actual
  state advancing to the next step is therefore not guaranteed symmetric.
  Hard-motif mode now reprojects that updated state at every step and restores
  the complete cross-chain groups afterwards, including when compactness is
  disabled. LRZ subsequently passed all 169 unit tests, including the
  compactness-disabled updated-state projection regression. A single
  pose-2131 50-step controlled rerun was the next gate at that historical
  stage; the exact orbit-state refactor above now requires CPU/preflight
  validation and a fresh 10-step smoke first.

## 2026-07-29 difficulty assessment and revised sampler architecture

The problem is not a single motif-write ordering bug. Three constraints must
hold simultaneously:

1. each cross-chain interface retains its internal all-atom geometry;
2. the full generated assembly remains exactly C3;
3. generated scaffold junctions remain continuous and physically plausible.

The earlier patches exposed the following interactions:

- A final native symmetry projection restores C3 but may reconstruct the two
  sides of an indexed cross-chain interface separately.
- Final-only motif reinsertion preserves the seed but creates a late
  motif-linker coordinate jump.
- Stepwise static motif restoration preserves the seed throughout diffusion,
  but skipping the native final projection allowed non-symmetric Euler-state
  errors to survive.
- Projecting the denoised prediction alone is insufficient because
  `X_noisy_L` contains independent atomwise noise.
- The pre-refactor C3 scripts set `allow_realignment=True`. That applied an
  arbitrary global rotation/translation every step, while the stored symmetry
  operators remain in their original frame. Unless the operators are
  conjugated by the same global transform, the scaffold projector and motif
  targets are expressed in incompatible coordinate systems. Upstream defaults
  `allow_realignment` to false; this is a local integration hazard, not
  evidence that ordinary upstream C3 generation is broken.
- The 169 passing tests cover group resolution and update-state projection,
  but do not yet prove true C3 closure with real transforms or the
  realignment/frame invariant.

The LHD101 constraints themselves are mathematically C3-compatible. With
`B=right(0)` and `C=left(1)` in the selected ASU, the runtime interface groups
must be:

```text
group(0) = B@0 + C@2
group(1) = B@1 + C@0
group(2) = B@2 + C@1
```

The adapter emits exactly this relation. The remaining risk is transform
matrix numbering/direction inferred by RFD3, which is not established merely
by observing transform IDs `[0, 1, 2]`.

The recommended formal design is orbit-space constrained diffusion:

- maintain the full-copy tensor projected onto one master-equivalent degree
  of freedom rather than three independently drifting copies;
- sample Gaussian displacement noise once on the ASU and rotate-copy it to
  the other C3 members (translations do not apply to displacement vectors);
- project predictions by inverse-transforming all copies to the master frame,
  averaging them, and re-expanding them, rather than copying Chain A alone;
- keep all diffusion states inside the C3-invariant subspace so the Euler
  update is closed by construction;
- initially disable arbitrary realignment; a future augmented frame must use
  conjugated operators `A S_k A^-1`;
- represent one cross-chain master interface with a pose `Q_t in SE(3)` and
  generate its orbit as `S_k Q_t P`;
- first hold `Q_t` static, then add bounded Kabsch-derived rigid motion so the
  interface can adapt during diffusion without changing its internal geometry
  or breaking C3.

Required new preflight/online gates:

- match Mosaic transforms to actual RFD3 matrices, allowing explicit
  permutation/direction resolution;
- fixed-target projection RMSD <= 0.01 A and maximum atom error <= 0.03 A;
- per-step orbit-equivariance residual below numerical tolerance;
- final transform-aware C3 coordinate RMSD <= 0.01 A and maximum error
  <= 0.03 A; copy-internal distance matrices are diagnostic only;
- motif rigidity, contact retention, continuity, compactness, and clash gates
  all pass independently.

Implementation should proceed in three stages:

1. static master orbit with realignment disabled and exact C3 invariants;
2. bounded master-interface SE(3) mobility with an early/middle/late schedule;
3. multiple independent motif orbits with overlap merging or explicit conflict
   rejection.

The historical post-Euler-only patch was a diagnostic bridge. It is now
superseded by the exact all-copy orbit-average implementation described above;
that implementation has completed LRZ/GPU end-to-end validation for the
LHD101 one-chain-ASU C3 baseline. It is not yet a scientifically final general
architecture for Dn, multiple independent motif orbits, or dynamic mobility.

## Historical archive boundary

All dated sections below preserve the sequence of earlier experiments. Words
such as “current”, “latest”, and “next” inside those sections describe that
historical stage and are superseded by the exact-orbit status and gate at the
top of this file.

## 2026-07-24 scaffold-generation diagnosis and guided experiment

- Job `5712555` is a positive seed-preservation result only.  Its complete
  generated protomers are structurally invalid, so it is not an end-to-end
  Interface-Seed success.
- The native RFD3 contig is parsed as intended: one continuous
  `B1-31,70-100,C1-30` ASU chain is expanded into three independent C3-related
  chains.  The observed failure is therefore not an absent contig or an
  unintended covalently closed trimer.
- A separate scaffold audit now reports chain continuity, per-chain CA radius
  of gyration, and coarse nonlocal CA clashes.  Seed integrity and scaffold
  validity are intentionally independent acceptance gates.
- `SampleDiffusionWithSymmetry` now has optional, default-off Interface-Seed
  compactness guidance.  It applies a capped, token-rigid translation to
  generated residues toward the fixed anchors of their own chain, leaves all
  fixed motif atoms untouched, fades to zero before the final denoising
  quarter, and reprojects coordinates to native symmetry after each guided
  update.
- At that stage, `lhd101_c3_full_single.sbatch` enabled the first conservative
  diagnostic
  setting with `RFD3_COMPACTNESS_WEIGHT=0.02`,
  `RFD3_COMPACTNESS_END_FRAC=0.75`, and
  `RFD3_COMPACTNESS_MAX_STEP=0.5`.  All are environment-overridable and the
  sampler defaults remain zero/off for ordinary RFD3 jobs.
- Local syntax compilation, shell parsing, and `git diff --check` pass.  The
  local base Python lacks project dependencies (`torch`, `pydantic`, and
  `pytest`), so the full unit suite must be run in the LRZ `rc-foundry`
  environment before GPU submission.

## 2026-07-24 fixed-interface result and evidence boundary

- The upstream RFD3 symmetry sampler finalized structures in the order
  `motif reinsertion -> symmetry projection -> global rigid alignment`.
  Upstream symmetry handling also distinguishes indexed motifs from entities
  treated as fully fixed.
- The conclusion that this ordering caused the observed cross-protomer
  interface displacement is a Mosaic source-code analysis, not an upstream
  Foundry statement or officially documented bug.
- The local correction finalizes in the order
  `symmetry projection -> complete-motif reinsertion -> global rigid
  alignment`, so the complete cross-protomer seed receives the final
  coordinate write.
- LRZ smoke job `5712555` is the local positive validation of that correction:
  all three recovered interface pairs passed the seed-integrity audit. These
  measurements are local experimental results and must not be presented as
  official Foundry data.
- The next scientific gate is 200-step sampling across the selected,
  geometrically diverse pose manifests, followed by independent seed-integrity
  and scaffold-validity acceptance.

## 2026-07-24 junction-failure diagnosis and stepwise motif preservation

- Guided 200-step job `5713652` preserved all three cross-protomer seed pairs,
  but failed scaffold validation with 114 CA clashes and nine chain breaks.
- Every chain broke at the same motif-linker boundaries: residues 31--32 were
  separated by about 33.9 A and residues 124--125 by about 24.1 A. This shows
  that final-only motif reinsertion restored the interface by making a large
  last-step coordinate change after the linker had already been generated.
- The sampler now has an opt-in
  `preserve_fixed_motif_during_symmetry` mode. Every symmetry projection
  restores the complete fixed motif in the current augmented frame, and the
  coordinate update keeps those atoms equal to the motif coordinates seen by
  that denoising step. Later steps can therefore adapt the generated linker to
  the true interface geometry.
- In stepwise-preservation mode, finalization no longer performs another native
  symmetry projection before motif reinsertion. This avoids recreating the
  final motif-linker coordinate jump.
- All C3 entry points enable stepwise motif preservation. The full single-GPU
  script returns compactness guidance to a default weight of zero; guidance is
  now an explicit environment-enabled experiment rather than the baseline.
- New unit coverage requires symmetry projection to modify generated atoms
  while leaving the complete multi-fragment fixed motif unchanged, and
  requires stepwise-mode finalization not to invoke a second final projection.
- This correction still requires LRZ CPU tests followed by one controlled
  200-step GPU run using pose 2131, RFD3 seed 42, and compactness disabled.
  Acceptance requires both seed integrity and junction continuity; seed
  preservation alone is insufficient.

## Completed

- Forked Foundry and established the feature branch.
- Confirmed the fork's `models/rfd3/src/rfd3` is imported through
  `PYTHONPATH`.
- Completed a native RFD3 smoke test on one P100 GPU using `dsDNA_basic`, one
  sample, ten diffusion steps, and low-memory settings.
- Defined strict Interface-Seed v2 schema objects for fragments, motion
  groups, interface ports, target geometry, symmetry orbits, interface edges,
  and directed scaffold links.
- Added cross-reference validation for ownership, ports, interfaces, orbits,
  and scaffold endpoints.
- Added and validated the LHD101 C3 configuration.
- Implemented core SE(3) operations and tests: validation, construction,
  inverse, composition, coordinate application, and axis-angle rotation.
- Implemented a generic cyclic Cn transform registry with stable transform
  IDs, arbitrary axis/center, signed orbit-offset resolution, group-element
  composition, and closure checking.
- Implemented the proper rotational Dn transform registry locally. It emits
  stable `Dn:e/rk/sk` IDs, accepts a configurable perpendicular two-fold axis,
  contains all `2n` proper rotations, preserves the requested center, and
  keeps cyclic orbit offsets inside each Dn coset.
- Added C3/C4/C5 closure, center, offset, composition, and master-copy drift
  tests.
- Server verification passed on 2026-07-21: all 38 unit tests passed in the
  shared `rc-foundry` environment (`Ran 38 tests ... OK`).
- Symmetry-orbit expansion into motion-group, fragment, and port instances was
  subsequently verified without failures in `rc-foundry`.
- The object-level `MappingRegistry` and its provenance tests were subsequently
  verified without failures in `rc-foundry`.
- Directed scaffold expansion and topology validation were subsequently
  verified without failures in `rc-foundry`.
- Deterministic PDB parsing, atom selections, and synthetic interface-frame
  tests were subsequently verified without failures in `rc-foundry`.
- The real LHD101 reference fixture, fragment selections, and port-frame
  integration were subsequently verified without failures in `rc-foundry`.
- Standalone CIF/mapping/manifest emission was verified on 2026-07-22 as part
  of the complete server suite (`Ran 80 tests in 0.429s`).
- Corrected C3 motif placement was regenerated and visually inspected on
  2026-07-22. The three motif pairs are separated C3-related copies rather
  than a collapsed, overlapping cluster.
- Audited RFD3's symmetry input path: RFD3 builds one ASU from the contig and
  then expands it with frames inferred from the full pre-symmetrized motif.
- Implemented the first static native-RFD3 adapter. It emits
  `rfd3_input.json` with an ASU contig, strict all-atom motif coordinate
  fixing, fixed motif sequence identity, and native C3 symmetry metadata.

## In progress

- Visual and quantitative inspection found that the first emitted C3 artifact
  was invalid: the unplaced seed centroid was only 2.52 A from the symmetry
  axis, giving a 0.182 A minimum inter-copy distance and 1,545 atom pairs below
  2.0 A across the three copy pairs.
- Corrected master-pose initialization (COM centering, explicit orientation,
  radial/axial placement) and a mandatory inter-group clash gate are
  implemented; the regenerated artifact passed visual separation inspection.
  The complete updated server test count still needs recording.
- Topology audit corrected an important semantic error: the preserved 7mwr
  A/B interface is same-copy (`interface orbit_offset: 0`), while each designed
  protomer connects one interface half to the geometry-selected neighboring
  copy. For the current LHD101 fixture this resolves to
  `right(k) -> left(k+1)` (`scaffold orbit_offset: +1`); the direction must not
  be hard-coded as a universal rule.
  InterfaceEdge instances and required-edge geometry diagnostics are
  implemented. The complete updated server test count still needs recording.
- Five static-adapter unit tests are implemented locally and pass syntax and
  whitespace checks. They await `rc-foundry` testing and native RFD3 input
  construction/prevalidation on LRZ.
- Added an RFD3-runtime prevalidation command that loads the emitted JSON and
  CIF, runs `DesignInputSpecification.build(return_metadata=True)`, verifies
  C3 chain/transform multiplicity, equal per-chain residue counts, recognized
  motif/fixed atoms, and ASU annotations, then writes
  `rfd3_prevalidation.json`. Four dependency-independent report-logic tests
  were added; the complete suite now contains 96 tests.
- Server verification passed on 2026-07-22: the complete updated suite ran all
  96 tests in the shared `rc-foundry` environment (`Ran 96 tests ... OK`).
- The static adapter successfully generated `presymmetrized_input.cif`,
  `mapping.json`, `manifest.json`, and `rfd3_input.json` on LRZ with the
  intended cross-copy ASU topology.
- The first runtime prevalidation exposed a residue-number namespace bug:
  AtomWorks selects mmCIF residues by `label_seq_id`, while the first adapter
  emitted original PDB `auth_seq_id` values (`B211-241` and `C165-194`). The
  adapter now correctly emits `B1-31,70-100,C1-30`; original author numbering
  remains preserved in `mapping.json` for provenance. Server revalidation is
  complete.
- Native RFD3 atom-array construction passed on LRZ on 2026-07-22. It produced
  three chains (`A`, `B`, `C`), 155 residues per chain, 1,488 recognized motif
  atoms, 732 fixed backbone atoms, and symmetry transform IDs `[0, 1, 2]`.
  The sampled ASU linker length was 94 residues (`31 + 94 + 30 = 155`).
- Re-audited the original Interface-Seed oligomer topology after questioning
  whether the entire ring should be covalently connected. The original code
  expands one contig per symmetry copy and separates those contigs as distinct
  chains; downstream notebooks explicitly design chains `A B C`. For C3 the
  intended topology is therefore three independent protomer chains:
  `B0-linker-A1`, `B1-linker-A2`, and `B2-linker-A0`. The preserved seed
  interfaces `A0:B0`, `A1:B1`, and `A2:B2` are noncovalent and assemble the
  three chains into the ring. The current native RFD3 adapter matches this
  topology; it does not create one covalently closed chain.
- Chain-colored inspection of the native smoke output confirmed this topology
  visually: cyan, magenta, and yellow form three separate protomer chains;
  each chain spans between two interface lobes, while each lobe contains a
  noncovalent motif contact between two differently colored chains. The loose
  appearance is therefore a sampling/linker-quality issue, not an accidental
  covalently closed C3 chain.
- Added a tracked Slurm script for a one-design, ten-timestep native C3 smoke
  test. The script repeats prevalidation inside the allocation before loading
  the checkpoint.
- The first smoke job (`5711261`) was submitted to the A100 partition but
  remained pending for priority. The tracked script was changed to allow the
  available V100/P100 partitions requested for faster scheduling, while
  retaining batch size one and RFD3 low-memory mode. P100 still carries an OOM
  risk for the 465-residue complex.
- The native ten-timestep C3 GPU smoke test subsequently completed and emitted
  a structure. Visual inspection shows three C3-related lobes, but the sampled
  94-residue linkers are loose and loop-rich. This confirms the execution path,
  not design quality: ten diffusion steps are far below the normal 200-step
  inference schedule. Output metadata metrics and motif preservation still
  need quantitative review before selecting full-run parameters.
- Quantitative review confirmed the ten-step output is an unconverged smoke
  structure: 209 chain breaks, 611 inter-residue clashes with sidechains, 70
  backbone clashes, 100% loop assignment, zero secondary-structure elements,
  and 13.39 A maximum CA deviation. The inference build sampled a 90-residue
  linker (`453 / 3 - 61 = 90`), independently of the earlier prevalidation
  sample. These failures justify a normal 200-step baseline before changing
  the original 70--100 linker range.
- Added a tracked, fixed-seed (`42`), one-design, 200-timestep Slurm script for
  the first full-quality baseline. It retains low-memory mode and the GPU
  partitions proven to schedule the smoke run.
- Added a second full-quality script for LRZ's `lrz-hgx-h100-94x4` partition.
  It runs one 200-step sample with seed `43`, records PyTorch/CUDA compute
  capability at startup, disables low-memory mode to use the H100's available
  memory, and writes to a separate `native_c3_full_h100` run tree. This makes
  it a useful replicate rather than duplicating the seed-42 legacy-GPU job.
- Live `sinfo` output on 2026-07-22 confirmed the current partition spelling is
  `lrz-hgx-h100-94x4` (the earlier public training material showed `92x4`), so
  the tracked H100 script was corrected before submission. At the same time,
  the seed-42 full baseline job `5711276` was running on `p100-001`.
- Reduced the H100 single-design walltime request from 12 hours to 2 hours.
  Walltime is only an upper bound, but the shorter request is more realistic
  for one 200-step H100 inference and may improve backfill scheduling.
- Keeping local, GitHub, and LRZ server copies synchronized.
- The new Dn registry, schema dispatch, D2/D3/D5 closure tests, and D3
  instance-expansion test are implemented locally. Syntax and whitespace
  validation pass; the local system Python lacks Pydantic, so the complete
  suite still needs to be rerun in the server `rc-foundry` environment.
- Implemented named group-element copy relations locally. Configuration can
  now use `copy_relation.transform: D3:s0`; relations act as
  `target = relation @ source`, allowing deterministic pairing between the
  two Dn cyclic cosets. The schema now accepts canonical transform IDs with a
  colon, and both interface and scaffold compilation resolve them.
- Extended standalone prescreen diagnostics locally. Clash reports are now
  separated into cyclic, Dn intra-coset, and Dn inter-coset pair classes;
  scaffold links report terminal-anchor distance and a conservative maximum
  contour feasibility estimate; each symmetry orbit reports central void and
  principal-axis clearance descriptors. These diagnostics are written to the
  manifest and do not add uncalibrated rejection thresholds.
- Extended the static RFD3 adapter and prevalidation logic from Cn-only to
  native Cn/Dn symmetry, with an explicit guard for RFD3's current
  10-transform symmetric-motif limit. Adapter metadata records multiplicity
  and Mosaic transform order. Added an end-to-end D2 adapter fixture.
- Audited RFD3's native dihedral frame generator and found that D3 produced
  six entries but only four unique rotations. The fork now constructs Dn from
  one fixed perpendicular two-fold generator, preserving all `2n` unique
  proper rotations. Added D3/D6 uniqueness and D2/D3/D5 closure tests. No
  model architecture or checkpoint was changed.
- Added adapter-side Cn/Dn registry preflight for multiplicity, uniqueness,
  proper rotations, and group closure. Added a real no-linker adapter mode:
  `chain_break: true` emits `/0` and records a two-chain ASU instead of
  silently creating a continuous linker. Prevalidation now supports repeated
  multi-chain asymmetric units with unequal chain lengths.
- Extended the standalone CLI summary to print symmetry/copy count, hard
  clashes, linker-span feasibility, central void radius, and axis clearance
  without requiring manual inspection of `manifest.json`.
- Implemented the first backend-independent objective/scoring layer locally.
  Configurable minimize/maximize, upper/lower bound, target-with-tolerance,
  and range terms emit per-objective diagnostics and deterministic ranking
  keys that prioritize required-constraint feasibility. Static compiler
  diagnostics are exposed through stable metric names.
- Added relaxed standalone compilation (`strict_validation=False` or CLI
  `--allow-infeasible`) so pose search can retain, diagnose, and rank invalid
  candidates. The RFD3 adapter continues to use strict validation.
- Reframed the implementation order around general software capabilities:
  objective API -> static pose search -> symmetry feasibility screening ->
  conflict diagnostics -> dynamic controller. The final plan now explicitly
  separates reusable features from C3/D2/D3 benchmark fixtures.
- The user reported that the normal-timestep native C3 RFD3 run completed and
  supplied a structure image on 2026-07-22. Visual inspection shows three
  assembly lobes and substantially formed secondary structure, but also long,
  extended inter-motif scaffold regions.
- Quantitative audit of the fixed-seed (`42`) 200-step P100 result from job
  `5711276` is complete. Relative to the ten-step smoke result, maximum CA
  deviation improved from 13.39 A to 0.876 A, internal chain breaks from 209
  to 3, sidechain-inclusive clashes from 611 to 9, and backbone clashes from
  70 to 3. The structure contains 18 secondary-structure elements with 38.7%
  helix, 29.7% sheet, and 31.5% loop instead of the smoke result's 100% loop.
  This validates normal-timestep convergence and strong motif preservation.
- The full result contains 462 residues, or 154 residues per C3 protomer. With
  61 indexed motif residues per protomer, the sampled linker is 93 residues,
  confirming that its long visual appearance follows the configured `70-100`
  linker range rather than a contig parsing failure. Compactness therefore
  remains an objective/configuration question.
- RFD3's `n_chainbreaks` metric explicitly zeroes normal inter-chain
  boundaries before counting deviations greater than 0.75 A from the standard
  3.8 A CA spacing. The remaining count of 3 therefore represents internal
  continuity defects, not the expected boundaries between chains A, B, and C.
  The run passes execution, motif-fidelity, and fold-formation checks, but
  topology continuity and sterics remain partial rather than fully accepted.
- After the complete local-to-LRZ source synchronization on 2026-07-22, the
  `rc-foundry` environment passed all 127 discovered Mosaic unit tests in
  3.677 seconds. This includes the Cn/Dn registry and RFD3 frame tests,
  adapter/prevalidation tests, standalone output tests, and objective/scoring
  tests.
- A subsequent method-level audit identified a critical acceptance gap. The
  legacy Interface-Seed implementation applies one rotation and translation
  to the complete two-fragment reference interface and only then symmetry-
  copies that intact rigid seed. Mosaic's schema expresses the same intent by
  placing `left` and `right` in one rigid `primary_seed` motion group, and its
  standalone interface-edge check validates the compiled relative pose.
  However, the current tests do not yet prove that this intact contact survives
  the full `presymmetrized_input.cif -> native RFD3 symmetric-motif build ->
  generated structure` path. In particular, supplying an already expanded
  motif together with native RFD3 symmetry may be interpreted differently
  from the legacy single-seed expansion. Until fixed-motif contact retention
  is measured at each boundary, the completed 200-step run is an execution
  and folding baseline, not a validated Interface-Seed reproduction.
- Hardened all tracked C3 Slurm entry points against stale compiler artifacts.
  Each allocation now recompiles its own adapter JSON, pre-symmetrized CIF,
  mapping, and manifest under the job-specific run directory before
  prevalidation and inference. All three scripts explicitly select
  `inference_sampler.kind=symmetry`; reusing the earlier shared
  `lhd101_c3_adapter` directory is no longer allowed. This does not replace the
  pending RFD3-built and final-model seed-contact audits.
- Compared the legacy smoke output from job `5711263` with the fresh-adapter,
  symmetry-sampler, fixed-seed (`45`) smoke output from job `5711563`. They are
  not identical: the sampled scaffold lengths are 90 and 78 residues (453 and
  417 total residues), and their coordinates and metrics differ. Nevertheless,
  both exhibit the same method-level failure. Their cyclic endpoint pairing is
  consistent (`A_start:B_end`, `B_start:C_end`, `C_start:A_end`), but the seed
  halves are catastrophically overlaid. The reference LHD101 seed has a minimum
  inter-fragment CA distance of 4.223 A and 34 CA pairs below 8 A; job `5711263`
  gives minima of 0.55--0.66 A with 130--132 pairs below 8 A, while job
  `5711563` gives 1.17--1.24 A with 112--114 pairs below 8 A. Fixed motif
  backbones must not acquire such geometry even in a ten-step smoke run. This
  disproves stale input or an unlucky random seed as the sole cause and points
  to the adapter/RFD3 symmetric-ASU coordinate interpretation. Further GPU
  sampling is blocked until RFD3-build seed geometry is audited and corrected.
- Tightened the reproduction baseline from backbone-only motif conditioning to
  strict all-atom seed freezing. Both LHD101 interface fragments now compile as
  `select_fixed_atoms: ALL`, while `redesign_motif_sidechains` remains false,
  for C3 and the D2/D3 dry-run configurations. Rigid-body initialization may
  still rotate/translate the complete two-fragment seed before a job is built,
  but no atom within that placed seed may move during RFD3 denoising. The RFD3
  prevalidator now rejects a job unless every recognized motif atom has both a
  fixed coordinate and fixed sequence identity. This is a pre-GPU hard gate,
  not merely a reported metric.
- Server verification after the strict all-atom update passed on 2026-07-22:
  the `rc-foundry` environment discovered and passed all 129 Mosaic unit tests
  in 3.244 seconds.
- Replaced the single fixed `[0, 0, 0]` LHD101 pose with reproducible
  Haar-uniform SO(3) rigid orientation sampling and a 20--30 A radial interval.
  The complete two-fragment seed remains all-atom fixed; sampling applies one
  shared SE(3) transform and therefore cannot alter its internal PPI geometry.
  Job-specific `--pose-seed` overrides are now recorded together with the
  sampled quaternion, rotation matrix, radius, axial offset, and centers.
- Added a CPU-only `rfd3_mosaic.pose_ensemble` compiler. It generates many
  deterministic pose candidates and ranks them before GPU use, rejecting hard
  clashes, unsatisfied required interfaces, infeasible continuous link spans,
  and required-objective failures. The C3 smoke/full Slurm scripts now pass
  their job seed into rigid-pose compilation instead of silently reusing one
  fixed pose for every run.
- The first 64-pose server ensemble (pose seeds 1000--1063) completed with
  64/64 candidates accepted. This validates deterministic SO(3)/radius
  sampling, but it also exposed an under-discriminating first-pass score: the
  70--100-residue contour gate is only a necessary reachability check, no
  objectives were configured, and the old final tie-break incorrectly
  preferred greater inter-group separation. The scorer now exposes minimum,
  mean, and maximum linker endpoint spans plus the maximum contour-derived
  residue requirement. The LHD101 example applies two explicitly soft
  shortlist heuristics (minimize the worst endpoint span and constrain the
  central-axis opening to a configurable soft window after hard gates), and
  the generic fallback tie-break minimizes the
  worst linker span rather than maximizing seed separation. These scores rank
  GPU candidates; they are not evidence of foldability or designability.
- The corrected 64-pose rerun produced nonzero discriminating scores. Its old
  top pose (seed 1010) combined a 25.640 A worst linker span with only 2.267 A
  axis clearance, exposing a second ranking-direction issue: unboundedly
  minimizing the central opening rewards near-axis placement. The LHD101
  example now uses an explicitly heuristic 6--14 A soft clearance window
  instead. This range is example configuration, not a universal Cn rule. Seed
  1058 (24.623 A worst span, 10.639 A clearance) is the provisional geometric
  leader before orientation-diversity selection and RFD3 validation.
- Added `rfd3_mosaic.pose_select`, which reads an existing ensemble without
  recompiling candidates. It preserves geometry-score order but suppresses
  near-duplicate orientations using the sign-invariant geodesic angle between
  sampled unit quaternions. Pool size, shortlist size, minimum SO(3) angular
  separation, and (for future multi-group inputs) the diversity group are all
  explicit CLI parameters. It never silently fills a shortlist with candidates
  that violate the requested diversity threshold.
- Visual review and the v3 ranking showed selection collapse toward the lower
  half of the 20--30 A radius interval; the top ten all lay below 25 A. This is
  not a failure of random-number generation: Haar SO(3) naturally places more
  principal axes near transverse orientations, while a single compactness
  score couples radius and orientation by preferentially retaining poses with
  short linker spans. The ensemble compiler now supports reproducible joint
  Latin-hypercube sampling of radius, axial offset, and the three Shoemake SO(3)
  unit variables. This preserves Haar orientation marginals while providing
  space-filling finite-sample coverage across all pose inputs.
- Added a coordinate-invariant visual-tilt diagnostic. Each rigid motion group
  receives a deterministic longest PCA axis from all source-seed coordinates;
  the manifest records its source/world vectors and its sign-invariant 0--90
  degree tilt relative to the symmetry axis. Degenerate PCA cases are reported
  as unavailable rather than assigned an arbitrary axis.
- Added `rfd3_mosaic.pose_stratify`. It reads an ensemble and retains the best
  accepted pose independently in each configurable radius-by-principal-tilt
  cell. The LHD101 defaults use four radius strata across 20--30 A and four
  equal tilt strata across 0--90 degrees. Empty cells and candidates outside
  configured bins remain explicit in the coverage report. This prevents one
  compact, highly tilted family from monopolizing the shortlist; the bins are
  exploration controls, not biological acceptance thresholds.
- Server verification of the joint sampler initially found one stale unit-test
  assumption rather than a compiler defect: the standalone test still required
  an exact 25 A radius even though the LHD101 configuration now samples 20--30
  A. The test now compares the emitted structure center with the provenance
  `sampled_radius` and independently checks the configured interval, preserving
  both deterministic auditability and the intended variable-radius behavior.
- The v4 256-pose joint ensemble occupied all 16 configured radius-by-tilt
  cells. Principal tilts span 1.542--74.685 degrees in the reported cell
  representatives, and radii span 20.144--29.837 A. This is the first direct
  evidence that finite-sample coverage no longer collapses to one compact,
  highly tilted pose family.
- Closed a provenance gap between CPU search and GPU inference. A Latin-
  hypercube candidate cannot be reconstructed by passing its integer pose seed
  alone because its explicit unit samples override the ordinary RNG stream.
  The RFD3 adapter now accepts a candidate manifest, validates the config hash,
  recovers the exact per-group unit samples, rebuilds the structure, and fails
  unless the rebuilt CIF SHA256 exactly matches the searched candidate. All C3
  Slurm entry points accept `RFD3_POSE_CANDIDATE_MANIFEST` and otherwise retain
  their legacy seed-based behavior.
- Historical, now superseded: the earlier C3 inference entry points explicitly
  set
  `inference_sampler.allow_realignment=True` and
  `+inference_sampler.insert_motif_at_end=True` (Hydra append syntax), in
  addition to compiling every
  motif atom as fixed-coordinate/fixed-sequence. In the RFD3 symmetry sampler,
  this is required to reinsert the ground-truth indexed motif during diffusion
  and at the final step. `allow_realignment=False` only suppresses coordinate
  noise; it does not make an indexed motif a hard positional constraint.
  The post-generation coordinate/contact audit verified the cross-chain
  interface seed directly. Current exact-orbit entry points instead disable
  realignment and do not rely on end-only motif insertion.
- Added `rfd3_mosaic.rfd3_seed_audit`, a generator-output audit that combines
  the adapter mapping with RFD3's `diffused_index_map`, recovers the two
  original source fragments, and searches for the best one-to-one cross-chain
  pairing among generated protomers. It reports per-seed all-heavy-atom RMSD,
  CA RMSD, atom completeness, reference-contact retention, and contact-distance
  RMSE. Same-chain fragment pairs are never accepted. Default acceptance
  requires CA RMSD <= 0.5 A, all-heavy-atom RMSD <= 0.75 A, at least 99% atom
  completeness, and at least 90% retention of reference contacts within
  4.5 A.
- Added a dependency-light RFD3 mmCIF/mmCIF.gz atom-site reader for this audit.
  A recovered pre-fix ten-step result was used as a negative control: all three
  inferred cross-chain seeds failed, with maximum CA RMSD 1.835 A, maximum
  all-atom RMSD 3.246 A, and minimum contact retention 0.473. This confirms the
  audit detects the original fixed-motif failure instead of passing it through
  symmetry alone.
- All three C3 Slurm entry points run seed and transform-aware scaffold audits
  after inference. Both JSON reports are written before a separate audit gate
  marks the job failed when either scientific check is rejected.
- Corrected seed-2153 job 5712416 reached the first realignment step but P100
  rejected `torch.linalg.svd` on a bfloat16 covariance
  (`svd_cuda_gesvdjBatched not implemented for BFloat16`). The shared Kabsch
  utility now promotes only the alignment solve to float32 for float16/bfloat16
  callers, retains float64 when requested, and casts aligned coordinates back
  to the caller dtype. This preserves hard motif reinsertion instead of
  disabling realignment. CPU bfloat16 regression coverage was added both to
  Foundry's alignment tests and to the Mosaic unittest discovery suite.
- Smoke job 5712530 showed that dtype promotion alone was insufficient:
  RFD3's outer bfloat16 autocast converted the float32 covariance-producing
  `einsum` back to bfloat16. The complete fix now disables autocast only around
  the small Kabsch covariance/SVD/rotation block; regression tests execute the
  bfloat16 call from inside an autocast context to reproduce the actual sampler
  call path.
- On 2026-07-23 the latest sampler finalization, BF16 alignment, seed audit,
  and associated local changes were synchronized to the LRZ working tree.
  Server-side unittest discovery completed successfully: 151 tests ran in
  3.719 seconds and all passed. This cleared the CPU test gate. The subsequent
  10-step GPU smoke job `5712555` passed all three cross-chain seed-integrity
  checks; full 200-step scaffold-quality validation remains outstanding.

### Why the fixed-interface finalization patch is necessary

The previous symmetry-sampler finalization order was:

```text
reinsert ground-truth fixed motif
-> apply native symmetry projection
-> globally rigid-align the result to the motif
```

An indexed interface motif can contain fragments on different protomers.
Native symmetry projection can therefore apply different transforms to the
two fragments. Each fragment remains internally correct, but their relative
cross-chain pose—and consequently the original interface contacts—is
destroyed. A final global rigid alignment cannot repair this: one global
rotation and translation cannot simultaneously invert two different
per-protomer transforms.

`SampleDiffusionWithSymmetry._finalize_with_fixed_motif()` changes the order
to:

```text
apply native symmetry projection to the generated scaffold
-> reinsert the complete ground-truth fixed motif as one coordinate set
-> globally rigid-align using all fixed motif atoms
-> return without another symmetry projection
```

This gives the complete fixed interface the final coordinate-write
precedence. The patch does not alter the contig, chain topology, checkpoint,
network weights, or C3 transform definitions.

The accompanying Kabsch change in `src/foundry/utils/alignment.py` performs
only the covariance/SVD/rotation solve in float32 (or preserves float64),
explicitly outside the outer bfloat16 autocast context, and casts aligned
coordinates back to the caller dtype. This avoids the P100 bfloat16 SVD
failure without converting the full inference calculation to float32.

Passing CPU tests proves the intended ordering and mixed-precision code path,
not the scientific result. GPU acceptance still requires all three recovered
cross-chain interface copies in `seed_integrity_audit.json` to satisfy atom
completeness, RMSD, and reference-contact-retention thresholds. Linker
junction geometry and final C3 consistency must also be checked after motif
reinsertion.

## Canonical pose-to-200-step workflow

Canonical ensemble:
`.../runs/rfd3-mosaic/lhd101_c3_joint_lhs_v4` (256 joint Latin-hypercube
poses; 16/16 radius-by-tilt cells occupied).

The first 200-step multi-pose batch is fixed to six v4 stratified candidates:

| pose seed | rank | radius (A) | tilt (deg) |
| ---: | ---: | ---: | ---: |
| 2131 | 1 | 20.499 | 41.153 |
| 2153 | 2 | 20.144 | 65.476 |
| 2003 | 15 | 22.428 | 6.144 |
| 2248 | 25 | 25.380 | 33.776 |
| 2213 | 31 | 25.618 | 57.702 |
| 2200 | 35 | 27.782 | 74.685 |

Key rules:

- all pairwise SO(3) distances are >=30 degrees (minimum 31.241 degrees);
- pass each candidate's `manifest.json`, not its integer pose seed;
- hold `RFD3_SEED=42` constant for the first comparison;
- each job must pass `seed_integrity_audit.json`;
- after seed preservation, evaluate junctions, chain breaks, clashes, fold
  quality, and C3 consistency.

- GPU smoke job 5712555 is the first positive seed-preservation validation of the
  corrected finalization order. Its `seed_integrity_audit.json` passed all
  three one-to-one cross-chain interface pairs (`A:B`, `B:C`, and `C:A`).
  All pairs had 496/496 matched heavy atoms (1.0 completeness), maximum CA
  RMSD 0.053180 A, maximum all-heavy-atom RMSD 0.048469 A, and minimum
  4.5 A reference-contact retention 0.978799. Contact-distance RMSE was at
  most 0.035735 A. This is a decisive positive control against the previous
  approximately 12.1 A combined CA RMSD and zero-contact-retention failure.
  The complete cross-chain interface is therefore preserved for this local
  10-step test pose. This does not establish full scaffold quality or an
  upstream RFD3 bug fix. The next phase is independent 200-step validation across
  multiple geometrically and orientationally diverse candidate manifests;
  each run must pass its own seed audit.

- Direct audit of the newly downloaded generated CIF
  `rfd3_input_lhd101_c3_interface_seed_0_model_0.cif.gz` on 2026-07-23 proved
  that fixed-atom annotations and end-of-run reinsertion were still
  insufficient. Each fragment was internally preserved (approximately
  0.05--0.09 A CA/all-heavy-atom RMSD), but no cross-chain fragment pairing
  retained the reference interface: the best cyclic pairs were approximately
  12.1 A CA RMSD with zero retained 4.5 A reference contacts. Thus RFD3 had
  fixed two isolated fragment shapes, not the complete two-fragment interface
  seed. The output is a scientific failure even though inference completed.
- Root cause was found in the symmetry sampler's final operation order. It
  reinserted the ground-truth fixed motif and then applied the native symmetry
  projection, allowing that projection to move the two protomer-spanning
  fragments independently. The local sampler now projects the generated
  scaffold into symmetry first and reinserts/aligned the complete fixed motif
  last. A regression test deliberately separates two motif fragments during a
  mock symmetry projection and requires finalization to recover their original
  4 A cross-fragment separation. The correction is present on LRZ and passed
  all 151 server-side CPU tests. GPU smoke job 5712555 subsequently passed the
  local interface-preservation gate.
- Three P100 submissions (5711682--5711684) exited before Python startup even
  though their stderr files were empty. Their stdout terminated immediately
  after `nvidia-smi` reported a corrupted infoROM, and `set -e` interpreted its
  nonzero diagnostic exit as a fatal job error. GPU inventory logging is now a
  nonfatal conditional in every C3 Slurm entry point; PyTorch/CUDA loading and
  inference remain fatal, so real compute failures are still surfaced.
- Each C3 Slurm entry point now redirects stdout and stderr, after creating its
  job directory, to `$RUN_ROOT/$SLURM_JOB_ID/slurm-$SLURM_JOB_NAME-$SLURM_JOB_ID.{out,err}`.
  Adapter files, RFD3 outputs, validation reports, and logs therefore stay
  together; Slurm's bootstrap streams are sent to `/dev/null`.
- Upstream Foundry's RFD3 symmetry documentation supports pre-symmetrized
  C/D motifs via `inference_sampler.kind=symmetry`, `diffusion_batch_size=1`,
  and `symmetry.is_symmetric_motif=true`; it does **not** provide an
  Interface-Seed / `asy_motif` / `motif_drag` example. Mosaic therefore uses
  the upstream symmetry entry point while supplying the missing
  Interface-Seed-specific pre-expansion, cross-copy contig topology, and
  fixed-motif reinsertion explicitly. This is an adapter layer, not a claim
  that RFD3 natively implements the earlier RFdiffusion extension.

## Not completed yet

Current priority order:

1. Exact-sampler targeted and complete LRZ CPU validation passed on
   2026-07-29: all 216 tests passed. Intermediate and final symmetry checks
   call the production scale-aware orbit-closure gate, while the independent
   fixed-motif coordinate comparison retains its strict `1e-5 A` regression
   check.
2. Rebuild the LHD101 adapter input and pass real
   `DesignInputSpecification.build -> AddSymmetryFeats.forward` prevalidation.
   This passed on LRZ: the linker materialized to 85 residues, three 496-atom
   constraint groups covered all 1488 fixed atoms, maximum fixed-target orbit
   error was `6.93e-5 A`, RMSD was `3.28e-5 A`, and both transform and orbit
   audits reported no failures.
3. Run one 10-step static exact-C3 smoke. Both seed and transform-aware
   scaffold audits must pass; the smoke is a wiring gate, not a fold-quality
   claim. Job `5721328` stopped before denoising because Lightning transported
   C3 feature matrices as bfloat16: the resulting ~`2e-3` raw orthogonality
   error was rejected before bounded polar normalization. The duplicate raw
   gate is removed locally; strict prevalidation of the original frame,
   maximum `1e-3` polar correction, and strict normalized-SO(3) checks remain.
   A bfloat16 C3 transport regression test was added before rerunning.
   Job `5721335` then reached fixed-target validation and exposed the same
   Fabric conversion on coordinates (RMS `0.040 A`, max `0.148 A`). The local
   engine now retains the pre-transfer geometry and restores only exact-orbit
   coordinates, noise, transforms, and constraint targets as float32 on the
   accelerator; the neural network remains bf16 mixed precision and no
   scientific threshold is relaxed.
   Job `5721339` reproduced the identical residual because the first engine
   implementation retained only a shallow alias to the nested batch. The
   correction now takes detached tensor clones before Fabric transfer and has
   a regression test that replaces tensors inside the same nested object.
   Job `5721344` showed the same residual and no precision-restoration log:
   the engine-side Hydra override copy was not a reliable runtime-mode
   detector. Exact geometry is now detected from the verified batch contract
   (`sym_transform`, `sym_orbit_slot`, and `sym_orbit_slot_verified=true`),
   which is symmetry-family independent and leaves ordinary batches unchanged.
   The synchronized correction passed all 220 LRZ unit tests.
   Job `5721348` still lacked the restoration log because the full orbit
   contract is not yet present at the engine precision boundary. Geometry
   preservation is therefore now unconditional for RFD3 inference:
   coordinates/noise and any present transforms/targets are restored as
   float32 after Fabric transfer, while model operations remain under bf16
   autocast.
   Job `5721355` proved from the stack that Lightning `_FabricModule.forward`
   still sits after the engine restoration point and reapplies the trainer
   precision policy to model arguments. Exact-orbit inference now overrides
   Fabric trainer precision to `32-true` at engine construction; non-exact
   samplers retain checkpoint precision. This is keyed by orbit-average mode,
   not by a C3 symmetry ID. A constructor-wiring regression test now checks
   the actual `RFD3InferenceEngine -> BaseInferenceEngine` trainer override,
   rather than testing only the mode predicate.
   Job `5721362` then crossed the runtime compatibility gate and preserved the
   complete interface seed. Declared-transform symmetry passed with maximum
   coordinate RMSD `1.21e-4 A` and maximum error `2.37e-4 A`; copy internal
   distance-matrix RMSD was `6.57e-6 A`. Compactness also passed and CA clashes
   fell to 9. The 10-step scaffold itself remained chemically under-denoised:
   every symmetry-identical 146-residue chain had 73 continuity failures
   (219 total), so the audit correctly failed. This result validates exact
   C3 state propagation but is not a valid final scaffold.
   Convergence jobs `5721369`, `5721370`, and `5721371` then tested the same
   seed-45 pose at 50, 100, and 200 steps. All three preserved the complete
   interface seed with 100% contact retention, had zero chain breaks, passed
   compactness, and retained declared-transform C3 coordinate RMSD near
   `1.2e-4 A`. Each failed only one intra-protomer CA clash copied through the
   exact C3 orbit: a generated-linker residue contacted the terminal residue
   of the right fixed motif. The 200-step case was best (`2.253 A`) but still
   below the hard `3.0 A` cutoff. The threshold must not be relaxed; the next
   scientific task is pose/diffusion-seed screening for a clash-free scaffold.
4. Screen selected pose manifests and diffusion seeds at 50 steps for a
   clash-free scaffold, then promote the best candidate to 200 steps.
   The next screening set should use the experimental morphology-aware
   `rfd3_mosaic.pose_qd` shortlist. It preserves the validated Haar SO(3) and
   joint Latin-hypercube generator, keeps the existing ensemble rank as the
   quality order, and distributes GPU candidates across axis-clearance and
   axial/radial-aspect cells with a global SO(3) separation gate. The standalone
   manifest now records axial span, radial thickness, aspect ratio, covariance
   eigenvalues, and shape sphericity for each symmetry orbit. These descriptors
   are exploration coordinates, not designability thresholds.
   The first 512-pose trial accepted 506 candidates and covered 13 morphology
   cells, but unconstrained cell filling admitted ensemble ranks 480 and 492.
   QD eligibility is therefore now restricted by default to the top 25% of
   accepted ensemble-ranked poses before morphology and SO(3) diversification.
   This top-quarter rule remains a compute-priority heuristic, not a claim that
   shorter generated-scaffold endpoint spans are universally better. These
   spans connect fixed fragments belonging to one protomer across adjacent
   interface positions; they are not flexible linkers between assembled units.
   The preserved cross-protomer interface seeds mediate unit self-assembly.
   Without an explicit target assembly size, morphology cells are parallel
   experimental conditions.
   Position quality must be estimated with a paired 50-step screen that uses
   the same set of at least three diffusion seeds for every pose; a replicate
   succeeds only if seed integrity, declared-transform symmetry, continuity,
   and hard-clash audits all pass. Promote positions by replicate success rate
   and declared morphology goals, not by endpoint span alone.
CPU pre-screening now goes beyond span/contour: every generated-protomer
   boundary reports C/N terminal-tangent-to-chord angles, tangent and peptide
   plane relative angles, chord axial fraction/out-of-plane angle, minimum
   chord-to-axis clearance, and an interior straight-chord clearance from the
   other fixed motif atoms. These are configurable boundary-condition and path
   risk descriptors, not claims that the generated 70--100-residue scaffold
   will follow a straight line or fold successfully.
   A C5/C6/C7 capability suite is now prepared. Each order has an explicit
   config, the same Haar-SO(3) plus Latin-hypercube pose generator, the same
   QD selection policy, and one generic P100 200-step entry point. The radial
   distributions are not copied from C3: they use
   `R_n = R_3 sin(pi/3) / sin(pi/n)` so the sampled adjacent-copy chord range
   is preserved when the cyclic order changes. Absolute cavity objective
   windows are scaled by the same factor, while QD uses the dimensionless
   `minimum_axis_clearance / sampled_radius` descriptor. This prevents larger
   cyclic orders from being penalized or collapsed into one morphology bin
   merely because their ring radii are larger. The C5/C6/C7 runs remain
   capability experiments until their adapter prevalidation, full inference,
   seed, continuity, clash, compactness, and declared-transform symmetry
   audits all pass.
   A tracked H100 robustness-screen entry point now submits the controlled
   matrix C5/C6/C7 x top three QD poses x five diffusion seeds x 50 steps
   (45 jobs). It records every job ID and exact pose manifest in a timestamped
   TSV. This is the first large-scale estimate of pose- and diffusion-seed
   robustness; it does not replace the existing convergence controls or
   downstream sequence/structure validation.
5. Replace sorted-chain output association with provenance-aware copy mapping
   before claiming general multi-chain or Dn scaffold auditing.
6. Validate D2/D3 through the real build/prevalidation and GPU paths.
7. Design dynamic motif pair-conditioning and a scaffold-derived pose signal
   before enabling the experimental orbit-rigid hook in formal scripts.
8. Only after those gates, extend to multiple independent motif orbits,
   soft-rigid motion, ligand/metal constraints, negative design, and additional
   symmetry families.

## Current limitations

- Cyclic Cn and the proper rotational Dn registry are implemented and have
  passed the complete 127-test server suite. Dn has not yet passed native RFD3
  generation validation.
- Polyhedral T/O/I, helical screw symmetry, and user-supplied explicit
  transform sets are not implemented yet.
- `schema/states.py` and `topology/pose_graph.py` remain placeholders for the
  later dynamic-guidance phase.
- Standalone CIF output contains motif coordinates only. The three configured
  70--100 residue scaffold links are recorded in the manifest but do not yet
  have generated coordinates.
- Standalone atom/residue indices are not claimed to be RFD3 indices until the
  adapter reads the CIF and verifies its own mapping.
- Radial placement removes catastrophic overlap but does not by itself prove
  that every requested cross-copy interface pose is optimal; explicit
  interface-edge geometry validation remains required before RFD3 inference.
- No RFD3 model architecture or checkpoint has been modified or retrained.
- The exact static C3 sampler path has passed LRZ runtime and GPU end-to-end
  validation for the one-chain-ASU LHD101 C3 baseline.
- Native C5/C6/C7 configs and run scripts exist, but no C5/C6/C7 GPU result
  has yet been validated. The requested P100 entry point uses low-memory
  mode; memory feasibility, especially for C7, remains an explicit runtime
  gate. These orders must not be described as established capability before
  the full audit gate passes.
- The schema, symmetry registry, and instance compiler can express higher
  orders such as C12 and C20. The local branch has removed the adapter and
  Foundry frame-recovery limit of 10 transforms, with C12/D6 CPU regressions;
  this removes the artificial input boundary but does not yet establish GPU
  inference support. Dense token-pair memory remains quadratic in assembly
  size, the checkpoint's relative-chain encoding saturates beyond nearby
  copies, high-order chain-ID paths are unvalidated, and the current
  seed-integrity audit has factorial pairing cost. C12/C20 must not yet be
  submitted as production native P100 diffusion jobs until the new CPU
  construction tests pass on LRZ and a bounded GPU probe is defined.
- Transform-aware output auditing currently assumes transform-major sorted
  chain IDs for the one-chain-ASU C3 baseline.
- Orbit-rigid mobility is an unvalidated, explicit opt-in experiment and is
  disabled in every formal Slurm entry point.
- Native C3 input construction and 10/50/100/200-step inference have
  succeeded. The 200-step candidate preserves the complete two-fragment
  interface, exact C3, continuity, and compactness, but retains one local
  linker/motif CA overlap copied threefold. A clash-free candidate is still
  required before claiming a scientifically final design, robustness, or
  generalization to Dn and multi-interface cases.

## 2026-07-30 scaffold-aware mobility pilot

- A separate, default-off experiment now closes the missing dynamic
  conditioning loop: a moved interface target also refreshes RFD3
  `motif_pos` and group target coordinates before the next denoising step.
- The pilot fails closed unless it uses one design, the low-memory/chunked
  pair path, exact orbit-average state, coupled noise, and fixed-motif
  preservation. Input mobility declarations and sampler opt-in must agree.
- The proposed scaffold-derived controller treats the complete cross-chain
  seed as one master SE(3) object, expands its copies through the declared Cn
  actions, and scores generated/fixed junctions, coarse CA clashes, excessive
  axis tilt, and displacement from the sampled pose.
- The first C5 configuration permits at most `1 A / 5 deg` cumulative motion.
  Proposal-only is the default; applying motion requires an explicit flag.
  Formal static C3/C5/C6/C7 entry points remain unchanged.
- This is local refinement, not a high-tilt rescue mechanism and not a
  retrained RFD3 model. Targeted LRZ tests, a full unit run, and paired
  static/mobile GPU validation are still required before treating the
  experiment as successful.
- The first real C5 proposal-only attempt, job `5722585`, passed all 247
  repository tests but stopped before its first denoising step because the
  boundary finder assumed ordinary peptide neighbours were present in
  `token_bonds`. Foundry runtime features do not require those polymer edges.
  The local fix now combines explicit token bonds with same-chain consecutive
  `residue_index` CA tokens. Regression tests cover an empty-token-bond C5
  contig and reject false adjacency across chains or residue gaps. This fix is
  syntax-checked locally and remains pending LRZ unit and real-feature
  validation; job `5722585` is not a mobility result.

The concise design and validation boundary is recorded in
`docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md`.

### 2026-08-04 orbit-owned SE(3) control migration

- Motif mobility is now represented on `SymmetryOrbitSpec` and lowered to a
  topology-neutral `ConstraintOrbitInstance`; it is no longer conceptually
  owned by an interface edge. Existing edge-level mobility remains a
  compatibility input and is migrated fail-closed into the owning orbit.
- The assembly IR now carries constraint orbits plus generated scaffold links
  and N/C terminal extensions. This is the common representation needed to
  compile an interface-spanning motif and a central motif without separate
  sampler algorithms.
- Orbit mobility metadata now survives the complete compiler-to-RFD3 runtime
  path: allowed subspace, proposal source, cumulative translation/rotation
  bounds, timestep window, response, per-step trust region and objective IDs
  are encoded as RFD3 features and parsed by `ConstraintOrbitLayout`.
- `OrbitRigidMotifController` now applies the schedule belonging to each orbit
  rather than silently using only global sampler defaults. Its diagnostics
  report the effective subspace, proposal, objectives and trust region for
  every orbit. Legacy inputs that do not explicitly declare an orbit schedule
  retain their sampler-level schedule, so existing pilot commands remain
  reproducible during the migration.
- Native runtime execution currently accepts `bounded_se3` with
  `denoiser_fit` or `scaffold_objectives`. `radial`, `radial_axial`,
  `tilt_only` and `hoyeung_drag_compat` are valid IR contracts but fail closed
  until a topology-defined reference frame and proposal backend are wired;
  they are not falsely treated as arbitrary Cartesian SE(3).
- Ho-Yeung's original per-step COM dragging remains the compatibility design
  reference. It is not the native algorithm: the formal controller moves one
  master rigid pose and regenerates every copy through declared group actions,
  preserving the exact motif and symmetry orbit.
- Local `py_compile` and whitespace validation pass. After synchronization,
  the complete LRZ `rc-foundry` unit suite passed **313/313 tests in 9.804
  seconds**. This validates the orbit-owned mobility schema/IR, feature
  transport, runtime parsing, per-orbit scheduling, legacy schedule fallback
  and all preceding exact-symmetry regressions at the unit-test level. It is
  not yet a GPU validation of dynamic motif movement.

### Extracted C5/C6/C7 batch screening

`python -m rfd3_mosaic.rfd3_batch_screen` now audits an extracted structure
directory and resolves each `<job-id>__*.cif` back to its sibling run. The
strict rank requires the seed report plus continuity, compactness, zero CA
clashes, and declared-transform symmetry. Ring-plane and coarse inter-chain CA
packing descriptors are reported only for ranking, not as calibrated
acceptance thresholds. A local diagnostic over 18 copied C6 structures found
13 with zero chain breaks and zero CA clashes, four with replicated clashes,
and one severely discontinuous result; seed provenance and declared-transform
strict status must be recomputed on LRZ where the original job directories are
available.
`scripts/rfd3_mosaic/screen_extracted_cn_structures.sh` runs the C5/C6/C7
screens together and prints one compact comparison of the hard-gate counts and
top-ranked candidates.

### Selected low-tilt P100 comparison

The first retained C5 mobility candidate is pose seed `3419`:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/lhd101_c5_mobile_lhs_v1/candidate_0419_seed_3419/manifest.json
```

`scripts/rfd3_mosaic/submit_lhd101_c5_mobile_pair_p100.sh` submits a controlled
comparison on only the P100 partitions:

```text
same candidate + same diffusion seed 42
-> proposal-only 50 steps
-> applied-mobility 50 steps
-> matching 200-step jobs, each held by afterok on its own 50-step result
```

The wrapper records every job and dependency in
`c5_mobile_seed3419_p100_v1.tsv` under the run base, rejects incompatible
resume files, and can resume safely after the Slurm QOS submission limit. Its
experiment fingerprint includes the manifest, config and pilot-script SHA256,
diffusion seed, mobility interval, target tilt, and linker length, so a changed
shell environment cannot silently mix conditions. These jobs are prepared but
are not recorded as executed or validated until their result audit files are
inspected.

### 2026-08-03 C5/C6/C7 low-tilt production campaign

- Added `submit_lhd101_cn_low_tilt_8_3.sh` for one strictly accepted
  0--15-degree interface-seed pose per C5/C6/C7 order and 100 independent
  200-step diffusion seeds per pose (300 logical jobs).
- The full pose ensembles, rather than the QD shortlists, are searched. The
  three selected manifests are frozen with SHA256 provenance under the `8.3`
  campaign root before submission.
- Submissions are balanced across orders and capped at 36 per invocation for
  the LRZ QOS limit. Resume is state-aware: active/completed/audited tasks are
  skipped, while infrastructure-incomplete failures can receive one retry.
- The shared Cn batch script now accepts an isolated campaign root and no
  longer treats a non-zero diagnostic `nvidia-smi` exit as a model failure;
  the following PyTorch CUDA probe remains fatal.
- This campaign is prepared locally but is not recorded as synchronized,
  submitted, or scientifically validated until the LRZ selection table, job
  accounting, and both result audits are inspected.

### 2026-08-03 multi-interface D3 engineering checkpoint

- Removed the adapter's one-copy-zero-link restriction. It now compiles one
  or more disjoint scaffold segments, emits a deterministic chain per segment,
  and preserves legacy single-link JSON fields unchanged.
- Runtime motif constraint groups are now assembled from every selected ASU
  link. Multiple configured interfaces therefore become separate constraint
  orbits under one native symmetry registry.
- Seed-integrity mapping is no longer restricted to exactly two indexed
  fragments. The audit CLI derives all unambiguous fragment pairs from
  `interfaces -> ports -> fragments`, evaluates them independently, and
  retains the old report schema for a single interface. Chain association now
  uses minimum-cost bipartite matching instead of factorial permutation
  enumeration, so D3 multi-chain output does not make the audit intractable.
- Added `lhd101_d3_two_orbit_engineering.yaml` as an explicit plumbing test:
  two duplicated LHD101 interface orbits, two disjoint ASU chains and D3
  expansion. This is not yet a connected protein cage or a biological design.
- Added `validate_lhd101_d3_two_orbit.sh` as the fail-closed LRZ CPU gate. It
  runs adapter and seed-audit regressions, compiles the exact D3 input, and
  invokes the real Foundry input builder/prevalidation. GPU inference must not
  be added until this gate passes in `rc-foundry`.
- Local syntax compilation, shell parsing and whitespace checks pass. The
  local system Python lacks `pydantic`, so behavioral tests remain pending on
  LRZ; no server execution is claimed here.
- LRZ adapter and seed-audit regressions subsequently passed (19 adapter tests
  and 5 seed-integrity tests), but the first real prevalidation exposed an
  upstream frame-recovery ambiguity: two duplicated, sequence-identical motif
  occurrences per ASU were grouped into 12 entity instances, while D3 has six
  unique transforms. Mosaic already owns and validates the exact transform
  registry, so multi-chain Mosaic inputs now explicitly request declared
  frames instead of re-inferring them from entity multiplicity. The default
  Foundry behavior remains unchanged for ordinary inputs. This correction is
  syntax-checked locally and awaits LRZ regression/prevalidation.
- A separate 200-step run reached the network on a 16 GB P100 but failed in
  token initialization with CUDA OOM (13.07 GiB allocated and a further
  2.93 GiB requested). P100 is therefore excluded from this two-orbit D3
  experiment; subsequent runtime probes should use H100 or 80 GB A100 only.

### 2026-08-04 central-motif bidirectional-growth diagnosis

The product target now explicitly contains two symmetric fixed-motif
topologies, both of which must be supported rather than conflated:

1. a cross-subunit interface seed at the ends of an ASU scaffold segment,
   with the intervening segment generated; and
2. one fixed motif in the middle of each protomer, with new N- and C-terminal
   regions generated around it.

The second topology is not represented as a fake left/right interface.
Runtime constraint metadata now accepts an explicit `fixed_motif` group with
one `motif` role while retaining the strict two-role schema for interface
groups. `rfd3_central_motif_probe` derives a controlled central-motif input
from an existing C3 adapter input, and
`lhd101_c3_central_motif_probe_p100.sbatch` provides four paired arms:

```text
A  true original RFD3 state, realignment enabled, no complete-orbit restore
B  exact orbit-average/coupled-noise Mosaic state with complete restore
C  same original RFD3 state as A, but realignment disabled
D  legacy state, realignment disabled, complete motif-orbit restore
```

This matrix separates configuration/realignment effects from symmetry
projector overwrite and exact-state effects. A central-orbit audit compares
the complete output motif orbit to the source registry using a single joint
rigid alignment and an invariant all-pairs distance-matrix residual. The
experiment is prepared locally but no LRZ result is claimed yet. Unlike the
larger D3 two-orbit case, this C3 single-orbit probe is intentionally small
enough to test on P100 first.

The implementation target is arm D (`exact_mosaic`), not attribution of an
older external run. Arms A--C remain optional regression controls and should
not consume routine GPU budget. Production acceptance requires the central
motif orbit audit plus scaffold continuity, clash and symmetry audits; passing
only the motif audit is not sufficient evidence of a usable design.

### 2026-08-04 unified user-facing experiment CLI

Routine use no longer requires writing a long Slurm script. A strict,
versioned experiment YAML selects either `interface_seed` or `central_motif`,
while a separate execution profile owns partitions, resources, environment
activation and checkpoint paths. The public `exact_mosaic` preset expands to
the validated correctness contract: no realignment, motif-precedence restore,
orbit-average state, coupled noise and required complete motif groups. These
invariants are not exposed as casual user toggles.

The new `rfd3-mosaic validate/render/submit` flow freezes resolved paths and
hash provenance, emits a short generated sbatch, delegates execution to one
Python worker and applies the topology-specific motif audit plus the common
scaffold audit gate. Built-in P100, H100 and A100-80G profiles and two example
experiment files are present. Local syntax/render checks are required before
handoff; LRZ environment tests and GPU submission remain pending until the
changes are synchronized. User instructions live in
`docs/rfd3_mosaic/USER_CLI.md`.

For routine interactive use, the still simpler `central` and `interface`
commands generate that versioned experiment YAML internally. A central-motif
user supplies only the validated input JSON, motif selector, optional N/C
lengths, execution profile and output root; exact-symmetry settings remain
software-owned. The explicit YAML commands remain the auditable advanced and
batch interface rather than a prerequisite for ordinary use.

## Verification commands

```bash
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:$PYTHONPATH"
python -m unittest discover -s tests/rfd3_mosaic/unit -p 'test_*.py' -v
python -c "from rfd3_mosaic.compile import load_interface_seed_config; load_interface_seed_config('configs/rfd3_mosaic/single_interface/lhd101_c3.yaml'); print('LHD101 config OK')"
git status --short --branch
```

## Resume point

The exact all-copy orbit-average implementation has passed CPU,
adapter/prevalidation, and C3 GPU end-to-end validation. The first-class
interface-orbit parser and corrected `>3`-chain inter-chain attention path have
passed their targeted tests and the full LRZ unit suite. Resume with one paired
C5 static 50-step P100 rerun matching the former job `5722385` inputs; compare
seed, continuity, clashes, exact symmetry and neighbouring-chain contacts
before and after the attention fix. Only after this runtime gate should work
continue on a symmetry-reduced mobile pose state. Keep C12/C20 as a separate
high-order architecture problem.
