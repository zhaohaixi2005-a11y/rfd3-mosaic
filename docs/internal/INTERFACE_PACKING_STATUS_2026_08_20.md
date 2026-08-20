# Interface packing status and GPU decision (2026-08-20)

## Decision

The interface-packing implementation is real and active, but it is not yet a
scientifically reliable release claim.

The latest frozen 50-step H100 evidence used revision `1e58d1e`:

- job `5755028`, locked motif arrangement: 2 structures, 0 accepted;
- job `5755029`, guided mobile motif arrangement: 2 structures, 0 accepted.

All four structures preserved the declared motif/symmetry contracts and the
packing controller accepted multiple bounded updates.  This rules out a dead
code path.  It does **not** prove useful interface generation.

The best locked result was close: 10/9 contacting residues on the two sides,
maximum contiguous runs 6/5 against a 6/6 target, and 19 reciprocal residue
pairs.  It nevertheless contained two heavy-atom contacts below 2 A and failed
the required edge relation.  The remaining three structures were either too
distant or too sparse.  Therefore thresholds must not be weakened merely to
turn the current result green.

Revision `c0f7060` subsequently normalized the bounded motif SE(3) search, and
revision `3b41f95` made variable assembly poses independent per requested
design.  Neither revision has yet received a statistically useful 50-step
locked/guided packing campaign.  The graph-interface objective itself did not
materially change after the failed H100 gates.

## What is implemented

The current graph controller includes:

- explicit symmetry-expanded interface edges;
- paired sequence-contiguous patch discovery;
- coverage, continuity, orientation and shape objectives;
- CA clash, global safety, junction and backbone protection;
- stateful patch locking after physical capture;
- local rigid patch translation/rotation;
- line search and atomic multi-interface rollback;
- final post-denoising polish;
- a separate heavy-atom output interface audit.

The fixed semantics remain hierarchical:

- `fixed_arrangement: locked` keeps the complete motif arrangement fixed and
  moves generated regions only;
- `fixed_arrangement: optimize_components` preserves each motif internally as
  one joint-rigid symmetry orbit but permits bounded orbit translation and
  rotation;
- exact symmetry copies are always regenerated/projected together.

## Diagnosed limitation

The online optimizer is a local CA/backbone proxy.  It can capture a useful
patch when diffusion already places two generated surfaces near one another,
but it does not reliably recover outputs ending 10--17 A from contact.  A
post-diffusion local patch correction should not be turned into a large rigid
drag, because that would fight the RFD3 denoiser and can create structures the
model never refined.

There is also an intentional mismatch of resolution: runtime optimization is
primarily CA/backbone based, while the final relation audit uses real heavy
atoms.  The closest locked result demonstrates why both are necessary: its CA
minimum remained just above 3.5 A while two side-chain/heavy-atom clashes were
still present.  This is a candidate for a later all-atom safety refinement,
not a reason to remove the heavy-atom gate.

## Comparison with RFdiffusion1, Ho-Yeung and native RFD3

The Ho-Yeung interface-seed implementation does not provide a stronger
all-atom interface optimizer.  For every design it:

1. randomly rotates the supplied A/B interface seed;
2. samples an initial displacement;
3. symmetrizes the seed;
4. applies a heuristic COM drag before each RFdiffusion1 denoising step;
5. combines this with the native `olig_contacts` potential (`intra=1`,
   `inter=0.1`, quadratic decay);
6. generates many designs and filters them downstream.

Its public code is deliberately heuristic: the documented displacement is not
a true radial distance, and the drag is not a mathematically exact joint-rigid
SE(3) update.  Mosaic should retain the useful statistical idea--independent
pose and diffusion sampling--without replacing exact motif/orbit semantics by
that drag.

Native RFdiffusion1 generates a random asymmetric-unit initialization,
symmetrizes the noise/input at every step and benefits from auxiliary intra-
and inter-chain contact potentials.  Native RFD3 is an all-atom generator from
one supplied conditioning pose; it supports multiple input specifications but
does not provide Mosaic's assembly-level pose search, exact multi-orbit
constraint compiler or output interface contracts.

The intended combination is therefore:

```text
RFdiffusion1 statistical breadth
  + one independently sampled assembly pose per design
RFD3 all-atom backbone/side-chain generation
Mosaic exact fixed/joint-rigid/symmetry constraints
  + local packing guidance
  + strict heavy-atom scientific audits
```

## Next GPU evidence gate

Do not tune weights again from four outputs.  First run the current revision
as 6 locked and 6 guided outputs on LRZ, plus 2 locked and 2 guided outputs on
the RTX 3070 development server.  This distinguishes a low-yield stochastic
method from a systematically broken controller and tests the post-gate motif
mobility and per-design pose changes.

### LRZ / AI cluster (12 outputs, six jobs, any compatible GPU)

```bash
cd /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic

PY=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/software/conda_environment/rc-foundry/bin/python
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"

"$PY" scripts/rfd3_mosaic/submit_packing_replicates.py \
  --profile configs/rfd3_mosaic/sites/lrz/any_gpu.yaml \
  --seed 63000 --seed 63002 --seed 63004 \
  --designs-per-job 2 \
  --submit
```

`any_gpu.yaml` asks Slurm to choose H100, A100, V100 or P100 and does not submit
the same sample once per accelerator type.

### mightymorphin RTX 3070 (4 outputs, two sequential jobs)

Create a machine-local profile outside the repository:

```yaml
schema_version: 1
name: mightymorphin-rtx3070
executor: local
setup_commands: []
checkpoint: /scratch2/haixi/checkpoints/rfd3_latest.ckpt
foundry_checkpoint_dirs: /scratch2/haixi/checkpoints
```

Then run:

```bash
cd /scratch2/haixi/rfd3-mosaic
source .venv-gpu/bin/activate
export PYTHONPATH="$PWD/src:$PWD/models/rfd3/src:${PYTHONPATH:-}"
export DEBUG=false TYPE_CHECK=false NAN_CHECK=true

python scripts/rfd3_mosaic/submit_packing_replicates.py \
  --profile /scratch2/haixi/mightymorphin-rtx3070.yaml \
  --run-root /scratch2/haixi/runs/rfd3-mosaic \
  --seed 64000 \
  --designs-per-job 2 \
  --submit
```

Each launch writes a frozen `campaign_manifest.json`.  After completion:

```bash
python scripts/rfd3_mosaic/collect_packing_campaign.py \
  /absolute/path/to/campaign_manifest.json
```

The collector writes `packing_campaign_summary.json` and
`packing_campaign_summary.md` beside the manifest.  A scientific packing
result is accepted only when graph guidance, final heavy-atom interface
relations and scaffold validity all pass.

## Stop/go rule after this campaign

- If either mode has reproducible accepted yield and the accepted structures
  show broad continuous interfaces, retain the controller and calibrate its
  yield with a larger campaign.
- If locked remains near-contact but repeatedly fails by heavy-atom clashes or
  one-residue continuity deficits, add all-atom/backbone safety to the local
  proposal and rerun a focused gate.
- If guided remains far from contact despite distinct initial poses and active
  committed SE(3) transactions, revise the joint pose/patch objective; do not
  increase the allowed motion blindly.
- A zero-yield 12-output current-revision campaign is sufficient evidence for
  another algorithm change.  A zero-yield 2-output campaign is not.

## Multi-design audit follow-up

The first RTX 3070 `designs=2` run exposed an execution-layer regression after
both CIFs had been generated: a fixed arrangement has one compiled pose, and
the multi-example engine input overwrote that pose's one-example audit input.
The constraint audit then correctly failed closed because it was handed two
examples.  This was not a packing rejection and the generated structures were
retained.

The worker now keeps every pose-specific one-example compiler artifact under
`input/pose_<index>/` whenever a run requests several designs, while the merged
`input/rfd3_input.json` remains dedicated to the one-load multi-example RFD3
engine call.  Each output is audited against its exact pose-specific input.
Regression coverage explicitly checks that fixed-pose diffusion replicates do
not overwrite the audit contract.

## Recovered RTX 3070 evidence

The two first multi-design RTX jobs completed RFD3 and produced four CIFs
before the audit-input collision described above stopped post-processing.  The
outputs were re-audited without rerunning diffusion after the fix in
`4df4fcc`.  This separates the integration failure from the scientific result.

| mode | output | minimum edge distance (A) | heavy-atom coverage per side | contiguous patch per side | heavy clashes per symmetry edge | result |
|---|---:|---:|---:|---:|---:|---|
| locked | 0 | 3.500 | 7 / 7 | 4 / 3 | 2 | FAIL |
| locked | 1 | 5.775 | 3 / 3 | 1 / 3 | 0 | FAIL |
| guided | 0 | 3.504 | 7 / 7 | 4 / 5 | 2 | FAIL |
| guided | 1 | 5.614 | 3 / 4 | 1 / 4 | 0 | FAIL |

All four outputs passed exact constraint recovery and scaffold validity; both
guided outputs also passed the component-mobility audit.  None passed graph
guidance or the final assembly-interface relation audit, so the scientific
yield is 0/4.  The two near-contact outputs already reach seven contacting
residues on each side but fail by an incomplete contiguous patch plus two
heavy-atom clashes on every C3 edge.  The other two outputs remain sparse and
have poor shape/coverage.

This campaign intentionally used the already-usable input pose for both
diffusion replicas.  It tests the locked versus bounded-mobile runtime
controller, not the new per-design stochastic initial-pose path.  Guided
mobility was active but changed the final geometry only slightly.  Together
with the earlier H100 0/4 gate, current evidence is 0/8 accepted structures.
The next algorithmic change should therefore target continuity-aware,
all-atom-safe final proposals for near-capture interfaces; pass thresholds
must not be weakened to convert clashes into successes.
