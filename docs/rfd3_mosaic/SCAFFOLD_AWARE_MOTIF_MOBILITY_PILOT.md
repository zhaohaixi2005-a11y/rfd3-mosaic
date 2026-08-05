# Scaffold-aware interface-seed mobility pilot

## Purpose

The validated production path keeps every cross-protomer interface seed
static and generates an exact-symmetry scaffold around it.  That path remains
the baseline.

This pilot asks a narrower question: can one already reasonable seed pose make
small rigid-body adjustments during RFD3 denoising when the developing
scaffold indicates poor junction geometry, a local clash, or excessive tilt?
It does not retrain RFD3 and it does not alter the internal geometry of the
interface seed.

## Why this is a separate experiment

Moving only the coordinates is internally inconsistent.  RFD3 also conditions
on motif pair geometry.  A valid dynamic loop must therefore update both:

```text
denoise the scaffold
-> evaluate scaffold/seed geometry
-> propose one bounded master-orbit SE(3) step
-> regenerate all Cn copies from the symmetry operators
-> refresh motif pair conditioning and hard-constraint targets
-> continue denoising
```

The move is applied to the complete cross-chain interface orbit, never to its
two fragments or symmetry copies independently.

## First-pilot objective

The external pose controller uses a small, explicit objective:

```text
E = w_junction E_junction
  + w_clash E_clash
  + w_tilt E_tilt
  + w_prior E_prior
```

- `E_junction` penalizes fixed/generated CA boundary distances away from a
  peptide-like target.
- `E_clash` penalizes non-bonded mobile-seed/generated-scaffold CA distances
  below the coarse clash cutoff.
- `E_tilt` is an interval penalty above a declared maximum angle relative to
  the cyclic axis.  It does not force every seed to one exact orientation.
- `E_prior` keeps the pose close to the sampled initialization.

This objective is a transparent inference-time heuristic, not an RFD3-learned
score and not evidence that a candidate is experimentally designable.

## Safety boundary

The first implementation is default-off and accepts only:

- one cyclic `Cn` mobile motif orbit;
- one diffusion design per process;
- low-memory/chunked pair conditioning;
- exact `orbit_average` state, coupled noise, and fixed-motif preservation;
- a bounded early/middle update window followed by a frozen late window.

The C5 experimental configuration limits cumulative motion to `1 Å / 5°`;
per-update limits are smaller.  This can refine a low- or moderate-tilt
starting pose.  It is deliberately unable to rescue a 50--85° sideways seed.
Those poses belong in the static screening/control set, not in local
refinement.

`proposal-only` is the default pilot mode.  It records the suggested motion
and energy terms without moving the seed.  Applying movement requires an
explicit sampler flag.  The pilot script also rejects an applied run when the
manifest's initial tilt is more than `target maximum tilt + 5°`, because such
a pose is mathematically unreachable under the declared cumulative rotation
bound.  Proposal-only diagnostics may still use those high-tilt poses.

Candidate provenance stays strict: the adapter compares the complete config
SHA256.  A manifest generated from the formal static C5 config is therefore
not interchangeable with this experimental config, even when its initial
coordinates would be identical.  Generate a separate CPU pose ensemble from
`configs/rfd3_mosaic/experimental/lhd101_c5_mobile.yaml`.

## Validation ladder

1. Generate an experimental-config C5 pose ensemble and choose an accepted
   manifest with initial principal-axis tilt at most about `25°`.
2. Run targeted controller, scaffold-guidance, and dynamic-conditioning unit
   tests on an allocated LRZ node.
3. Run the complete repository unit suite.
4. Run a 10-step C5 proposal-only diagnostic with a low-tilt pose.
5. Compare static and proposal-only diagnostics for the same pose and
   diffusion seed.
6. Run paired 50-step static versus applied-mobility jobs.
7. Require seed integrity, exact C5 symmetry, continuity, compactness, and
   hard-clash audits; also verify every SE(3) update and cumulative bound.
8. Promote only a stable, audit-passing pair to 200 steps.

The tracked entry point is
`scripts/rfd3_mosaic/lhd101_c5_mobile_pilot_p100.sbatch`.  It defaults to a
50-step, proposal-only P100 job.  The formal C3/C5/C6/C7 scripts and configs do
not enable mobility.

### Runtime boundary representation

The first real C5 pilot exposed a distinction hidden by the original synthetic
tests: Foundry's `token_bonds` matrix does not have to contain ordinary peptide
neighbour edges. Scaffold guidance therefore obtains candidate junctions from
the union of:

- explicit same-chain protein `token_bonds`; and
- consecutive same-chain protein `residue_index` values with one CA-bearing
  token per residue.

Only fixed/generated transitions become junctions. Chain identity and residue
continuity prevent the fallback from bridging chain breaks or sequence gaps.
Job `5722585` stopped at the old boundary initialization before diffusion and
is retained only as failure evidence. The revised runtime path must pass LRZ
tests and a real C5 proposal-only run before the pilot advances.

## Selected C5 P100 comparison

The retained low-tilt candidate for the first paired experiment is pose seed
`3419`:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/lhd101_c5_mobile_lhs_v1/candidate_0419_seed_3419/manifest.json
```

Submit the complete P100-only comparison from the repository root:

```bash
bash scripts/rfd3_mosaic/submit_lhd101_c5_mobile_pair_p100.sh
```

The wrapper submits four controlled jobs:

```text
proposal-only, 50 steps
applied mobility, 50 steps
proposal-only, 200 steps, afterok on proposal-only 50
applied mobility, 200 steps, afterok on applied-mobility 50
```

All four runs use the same pose manifest and diffusion seed (`42` by default).
The paired design isolates the effect of applying the controller. It also
prevents a failed 50-step mode from automatically consuming a full 200-step
allocation. Jobs are restricted to
`lrz-dgx-1-p100x8,lrz-hpe-p100x4` and are recorded in:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/c5_mobile_seed3419_p100_v1.tsv
```

If the QOS submission limit interrupts the wrapper, rerun the same command. It
validates the TSV and skips combinations that already have a recorded job ID.
The TSV experiment fingerprint locks the manifest/config/pilot-script SHA256,
diffusion seed, update interval, target tilt, and linker length. Resuming with
different scientific parameters fails instead of silently mixing conditions.
An alternative manifest or diffusion seed can be supplied explicitly with
`RFD3_POSE_CANDIDATE_MANIFEST` or `RFD3_SEED`, but it must use a different
`RFD3_MOBILE_JOB_FILE`.

For a broader proposal-only control, the optional
`submit_lhd101_c5_low_tilt_p100.sh` selects the highest-ranked accepted pose
from each of `[0,10)`, `[10,20)`, and `[20,30]` degrees and submits 50/200-step
P100 jobs. Each 200-step job is held by `afterok` on its matching 50-step job.
This six-job control is separate from the seed-3419 proposal/applied comparison.

## Interpretation

A positive pilot result means that a bounded, symmetry-safe external
controller can improve local scaffold compatibility while RFD3 is denoising.
It does not yet establish support for Dn, multiple independent motif orbits,
large pose rescue, ligands, or a generally optimal ring/cage morphology.
