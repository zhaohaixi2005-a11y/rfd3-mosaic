# RFD3 Mosaic Development Status

Last updated: 2026-07-24

This file is the persistent project memory for resuming development after a
new login or a new Codex session. Update it whenever a milestone changes.

## Project identity

- Repository: `zhaohaixi2005-a11y/rfd3-mosaic`
- Upstream: `RosettaCommons/foundry`
- Active branch: `feat/interface-seed-compiler`
- Server working tree: `/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic`
- Local mirror: `/home/haixi/Documents/mosaic`
- Goal: preserve Interface-Seed 1.0 behavior while generalizing it into an
  RFD3-native, generator-independent framework for Cn/multi-chain and
  multi-interface design.

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
- `lhd101_c3_full_single.sbatch` enables the first conservative diagnostic
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
- The C3 inference entry points explicitly set
  `inference_sampler.allow_realignment=True` and
  `+inference_sampler.insert_motif_at_end=True` (Hydra append syntax), in
  addition to compiling every
  motif atom as fixed-coordinate/fixed-sequence. In the RFD3 symmetry sampler,
  this is required to reinsert the ground-truth indexed motif during diffusion
  and at the final step. `allow_realignment=False` only suppresses coordinate
  noise; it does not make an indexed motif a hard positional constraint.
  The post-generation coordinate/contact audit now verifies the cross-chain
  interface seed directly.
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
- All three C3 Slurm entry points run the seed audit after inference and write
  `seed_integrity_audit.json` inside the job directory. They use report-only
  mode so a scientifically rejected design remains distinguishable from a
  failed GPU job.
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

Priority order follows the final project plan:

1. Run 200-step C3 jobs for the selected diverse pose manifests while holding
   the RFD3 diffusion seed constant for the first comparison.
2. Require both `seed_integrity_audit.json` and the independent scaffold audit
   for every result; reject seed-preserving outputs with broken, clashing, or
   excessively dispersed generated scaffolds.
3. Compare the default sampler against the conservative, default-off
   compactness-guidance experiment before tuning guidance strength or linker
   ranges.
4. Extend the invariant report to compare the standalone compiled copies and
   RFD3-built fixed motif explicitly, in addition to the now-implemented final
   model versus reference audit.
5. Determine whether native RFD3 expects one ASU seed or a fully expanded seed
   when `is_symmetric_motif=True`, and remove any double-expansion semantics.
6. Locate the three internal chain breaks and three backbone clashes in the
   completed seed-42 result and determine whether they lie in generated linker
   regions or at motif/linker junctions.
7. Run native versus legacy versus corrected LHD101 comparisons.
8. Complete native RFD3 generation validation for the general Dn
   implementation while C3 sampling experiments run independently; its full
   server unit-test suite has passed.
9. Continue validating the optional compactness sampler hook before treating
   it as a design method rather than a diagnostic experiment.
10. Implement a single-interface controller, then generalize it to multiple
   non-equivalent interfaces.
11. Add soft-rigid motion, ligand/metal constraints, negative design, and an
   explicit transform registry; then extend finite polyhedral and helical
   transform families.

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
- Native C3 input construction and one 10-step seed-preservation validation
  have succeeded. A structurally valid 200-step scaffold that also preserves
  the complete two-fragment interface has not yet been demonstrated. This must
  pass before claiming Interface-Seed reproduction, design robustness, or
  generalization to Dn and multi-interface cases.

## Verification commands

```bash
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:$PYTHONPATH"
python -m unittest discover -s tests/rfd3_mosaic/unit -p 'test_*.py' -v
python -c "from rfd3_mosaic.compile import load_interface_seed_config; load_interface_seed_config('configs/rfd3_mosaic/single_interface/lhd101_c3.yaml'); print('LHD101 config OK')"
git status --short --branch
```

## Resume point

Run the selected diverse pose manifests at 200 diffusion steps. Accept a result
only when the complete cross-chain seed passes `seed_integrity_audit.json` and
the independently generated scaffold passes continuity, compactness, clash,
junction, and C3-consistency review. Compare the native sampler with the
conservative compactness-guidance experiment before tuning linker ranges.
