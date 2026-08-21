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

### 2. Generated C3 interface, paired locked/guided poses

Run six independently seeded poses for each mode (12 outputs total):

```bash
"$PY" scripts/rfd3_mosaic/submit_packing_replicates.py \
  --profile configs/rfd3_mosaic/sites/lrz/any_gpu.yaml \
  --seed 73000 --seed 73002 --seed 73004 \
  --designs-per-job 2 \
  --submit
```

Each campaign seed creates one matched two-pose population:

```text
locked pose i -> pose fixed throughout diffusion
guided pose i -> same initial pose, bounded radial/axial/rotation allowed
```

Different design indices receive different pose and diffusion seeds. The
campaign manifest records `pose_seed_start`; each run records the realized
pose/input digest in `pose_manifest.json`.

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
- Do not repeat the old 10-step I canary.
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
