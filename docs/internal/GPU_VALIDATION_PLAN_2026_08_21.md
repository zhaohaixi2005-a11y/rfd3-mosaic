# Minimal GPU validation plan (2026-08-21)

This plan separates missing execution evidence from scientific calibration.
It intentionally avoids rerunning every historical smoke test.

## Required jobs

### 1. Icosahedral 50-step runtime closure

Run exactly one frozen job from:

```text
experiments/lrz_public_i_static_runtime_t50_large_gpu_canary.yaml
```

This is not the old `public-i-static-short-t10-s961` experiment. The current
design declares `timesteps: 50`, `task: preserve_supplied_geometry`, one fixed
180 A pose, 60 explicit symmetry actions and no generated-interface objective.
It asks whether a complete fixed orbit and continuous generated scaffold
survive a production-length trajectory. More random I replicates are not
useful until this one closes the exact fixed-orbit contract.

Submit on the AI cluster:

```bash
cd /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic

PY=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/conda_environment/rc-foundry/bin/python
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"

grep -nE 'name:|timesteps:|task:' \
  experiments/lrz_public_i_static_runtime_t50_large_gpu_canary.yaml

"$PY" scripts/rfd3_mosaic/submit_gpu_release_gates.py \
  --gate i-static --submit
```

Do not run the full `validate` command for this 60-copy design on a constrained
login node. The release-gate launcher performs schema, selector and constraint
binding there, then uses `--defer-runtime-preflight` so complete 60-copy RFD3
construction and prevalidation run inside the 440 GB A100/H100 allocation.
The preflight remains mandatory and inference cannot start if it fails; only
its execution location changes.

Required evidence is: one 60-copy CIF, fixed-orbit constraint recovery,
continuity, clashes and symmetry diagnostics. Scientific compactness is
reported separately because this file is a runtime canary, not a designed I
cage.

#### Result of the 50-step I run

Job `5756755` completed 50 diffusion steps and produced one 60-chain CIF.  A
post-run investigation found that its reported `62.535 A` fixed-orbit RMSD
was an audit false positive, not a runtime loss of the fixed motif.  Native
RFD3 concatenated copies in registry order, but the audit sorted legacy chain
labels; above chain `Z` this exchanged the `[` and blank identifiers.  Keeping
coordinate-file encounter order gives a joint fixed-orbit RMSD of
`0.000132387 A` and a maximum error of `0.000241701 A`.  The runtime fixed
target independently reports `0.0 A` RMSD.

The remaining continuity flag is real but localized.  The ASU has one
generated-to-fixed peptide-junction defect between residues 10 and 11
(`C--N = 3.1511 A`), which exact I symmetry reproduces once in every chain.
Thus `chain_break_count: 60` means one ASU defect copied 60 times, not 60
independent failures.  The current `10-10,A1-9,10-10` input with packing off is
a runtime canary and should not be presented as a scientifically optimized I
cage.  After the chain-order audit correction, re-audit this existing output;
do not spend another 36-minute GPU run to re-prove fixed-orbit preservation.

The separate follow-up
`experiments/lrz_public_i_long_scaffold_t50_large_gpu_canary.yaml` uses
`20-20,A1-9,20-20` semantics (49 residues per physical chain) while retaining
the same I frame, fixed motif and 180 A pose. Its purpose is to test whether a
less pathologically short scaffold improves continuity and backbone
morphology. It does not declare a generated I interface and must not be
presented as a complete designed I cage. Submit it once with:

```bash
"$PY" scripts/rfd3_mosaic/submit_gpu_release_gates.py \
  --gate i-long-scaffold --submit
```

### 2. Generated C3 interface, paired locked/guided poses

Run six independently seeded poses for each mode (12 outputs total):

```bash
"$PY" scripts/rfd3_mosaic/submit_packing_replicates.py \
  --profile configs/rfd3_mosaic/sites/lrz/any_gpu.yaml \
  --seed 73000 --seed 73002 --seed 73004 \
  --designs-per-job 2 \
  --defer-runtime-preflight \
  --submit
```

`--defer-runtime-preflight` keeps lightweight planning on the login node and
runs complete RFD3 feature construction inside the GPU allocation. It does
not skip prevalidation. This is required when the login-node memory limit
kills the broad all-pair contact feature build before `sbatch` is reached.

Each campaign seed creates one matched two-pose population:

```text
locked pose i -> pose fixed throughout diffusion
guided pose i -> same initial pose, bounded radial/axial/rotation allowed
```

Different design indices receive different pose and diffusion seeds. The
campaign manifest records `pose_seed_start`; each run records the realized
pose/input digest in `pose_manifest.json`.

Submitted on 2026-08-21 from revision `a935529`:

| mode | campaign seed | job | requested outputs |
| --- | ---: | ---: | ---: |
| locked | 73000 | 5756760 | 2 |
| locked | 73002 | 5756761 | 2 |
| locked | 73004 | 5756762 | 2 |
| guided | 73000 | 5756763 | 2 |
| guided | 73002 | 5756764 | 2 |
| guided | 73004 | 5756765 | 2 |

All six public designs passed complete CPU geometry and RFD3 feature
prevalidation before submission. The frozen campaign manifest is:

```text
/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/_campaigns/packing-replicates/20260821T105510Z/campaign_manifest.json
```

Submission is not a quality verdict; collect and compare the twelve outputs
after all six jobs reach a terminal state. The collector reports executable
contracts, runtime CA-window targets and post-hoc backbone-heavy-atom
observations separately.

After all jobs finish:

```bash
"$PY" scripts/rfd3_mosaic/collect_packing_campaign.py \
  /absolute/path/to/campaign_manifest.json \
  --run-root \
  /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic
```

All CIFs are retained. Geometry contracts determine execution completion;
packing, interface and scaffold diagnostics remain scientific recommendations
for comparing the resulting backbones.

## Optional local replication

If the RTX 3070 is free, run one two-design job per mode from the same revision
and checkpoint. This is a portability check, not additional statistical
power, and should not delay the AI-cluster jobs.

## Jobs not required now

- Static T already has a complete twelve-action GPU closure.
- Static O is closed by job `5755569`: one accepted 24-copy output from the
  50-timestep `o-static-release-gate`, with constraint-orbit, scaffold-validity
  and RFD3-prevalidation audits all passing. Do not rerun O for this gate.
- Do not repeat the old 10-step I canary or rerun job `5756755` merely to
  resolve its fixed-orbit flag.  Use post-hoc re-audit after the chain-order
  correction.  A separate longer-scaffold experiment is required if the goal
  changes from runtime closure to scientific I-backbone quality.
- Do not submit another same-pose 12-output packing campaign.
- The 40 produced LHD101 backbones are retained evidence and do not need to be
  regenerated for these two gates.

## Decision after collection

- If pose manifests are distinct and at least some generated interfaces are
  broad, continuous and clash-free, retain the controller and estimate yield
  with a larger campaign.
- If outputs start far apart despite the 20--30 A pose envelope, revise the
  pre-diffusion pose envelope/ranking, not the fixed semantics.
- If near-capture outputs repeatedly miss continuity or contain heavy-atom
  clashes, revise the local all-atom-safe packing proposal.
- If guided commits motion but is no better than its matched locked pose,
  revise the joint objective/acceptance rule rather than increasing movement
  bounds blindly.
