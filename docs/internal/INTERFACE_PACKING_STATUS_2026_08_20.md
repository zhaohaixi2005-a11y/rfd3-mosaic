# Interface packing status and GPU decision (2026-08-20)

## Decision

The interface-packing implementation is real and active, but it is not yet a
scientifically reliable release claim.

## Broad-contact integration (2026-08-21)

The zero-yield 12-output campaign showed that distinct initial poses and an
active local patch controller were not sufficient: many outputs finished
10--17 A from the declared edge, where the selected-patch objective had too
little capture range. The CPU/runtime implementation has therefore been
changed, without adding a new task mode:

- every declared generated-interface edge now receives the public
  RFdiffusion `olig_contacts` coordination-number prior over all generated CA
  pairs;
- the prior uses the RFdiffusion-style `r0=8 A`, `d0=2 A`, guide scale `2`
  and quadratic early-to-late decay;
- the prior is normalized per physical interface side and balanced by source
  interface identity, so T/O/I multiplicity does not silently dominate Cn;
- Mosaic's existing contiguous-patch coverage, continuity, orientation,
  shape, junction, clash, symmetry and rollback contracts remain the second
  refinement level;
- `guidance.inter_chain_weight` now controls this broad inter-chain prior
  rather than multiplying every graph-refinement loss;
- diagnostics/audit schema v9 records the resolved prior parameters, its
  timestep schedule and per-edge energy.

This is the mature part borrowed from RFdiffusion: early broad contact
formation plus annealing. The exact fixed/joint-rigid semantics, finite-group
edge accounting and post-generation audits remain Mosaic-specific. The code
change is CPU-testable; useful interface yield still requires a new frozen
50-step GPU campaign and must not be claimed before that result.

Implementation evidence is traceable to the RFdiffusion Nature paper
(<https://doi.org/10.1038/s41586-023-06415-8>) and its public
`rfdiffusion/potentials/{potentials,manager}.py` source. The contact prior is a
generation aid, not a new definition of backbone success.

## Independent-pose configuration correction (2026-08-21)

Commit `3b41f95` implemented one independently seeded pre-diffusion assembly
pose per requested design whenever a public design declares a variable
`sampling.initial_pose`. The first 12-output packing campaign did not exercise
that path: both frozen C3 packing templates omitted `initial_pose`, so every
output reused the already-positioned input motif and changed only its diffusion
seed. This is recorded by the old run manifests and must not be described as
an assembly-pose diversity experiment.

The locked and guided C3 templates now declare the same explicit pose envelope:

- radial distance uniformly sampled from 20 to 30 A;
- fixed zero axial displacement (a common axial translation does not change a
  C3 neighbour relation);
- Haar-uniform SO(3) motif orientation;
- one diffusion trajectory per pose.

For C3, adjacent motif-centre separation is
`2 r sin(pi / 3) = sqrt(3) r`, hence the declared radial interval spans about
34.6--52.0 A. The selected G2 motif has a heavy-atom radial extent of about
9.37 A, so this interval does not begin with copy overlap. RFD3 prevalidation
still rejects any atomically invalid sampled orientation. These checks prove
geometric feasibility, not interface quality; the latter remains the purpose
of the GPU campaign.

The campaign launcher derives the pose seed from each campaign seed and gives
the corresponding locked/guided jobs the same ordered pose population. Thus
`locked[i]` and `guided[i]` are a controlled pair, while design indices and
different campaign seeds cannot silently reuse one pre-RFD3 pose. Locked
freezes its selected pose for all diffusion timesteps; guided starts from the
matched pose and may apply only its declared bounded rigid-orbit corrections.

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

Do not tune weights again from four outputs. First run the revised explicit
pose-distribution configuration as 6 locked and 6 guided outputs on LRZ, plus
2 locked and 2 guided outputs on the RTX 3070 development server. This
distinguishes a low-yield stochastic method from a systematically broken
controller and tests both bounded mobility and per-design pose changes.

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

The current-revision campaign was submitted on 2026-08-21 with seeds 73000,
73002 and 73004. Locked jobs are `5756760`, `5756761` and `5756762`; their
paired guided jobs are `5756763`, `5756764` and `5756765`. Each job requests
two independently instantiated designs. The frozen manifest is
`_campaigns/packing-replicates/20260821T105510Z/campaign_manifest.json` below
the LRZ run root. All six inputs passed geometry and RFD3 feature prevalidation
before submission; output-quality evidence remains pending.

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

## AI-cluster A100 follow-up

The first two current-campaign locked jobs also completed RFD3 and produced
two structures each.  Their original worker status is `failed` because they
were launched before the per-design audit-input isolation fix; post-hoc audit
from `4df4fcc` recovered the scientific results without rerunning inference.

| job | output | minimum edge distance (A) | heavy-atom residue pairs | chain breaks | constraint orbit | interface result |
|---|---:|---:|---:|---:|---|---|
| `5755477` (`s63000`) | 0 | 11.457 | 0 | 3 | PASS | FAIL, 0/3 edges |
| `5755477` (`s63000`) | 1 | 3.503 | 5 | 0 | PASS | FAIL, 0/3 edges |
| `5755478` (`s63002`) | 0 | 15.030 | 0 | 6 | PASS | FAIL, 0/3 edges |
| `5755478` (`s63002`) | 1 | 15.970 | 0 | 0 | PASS | FAIL, 0/3 edges |

Thus the completed A100 locked subset has scientific yield 0/4.  Only one
output approaches contact, but it still lacks a complete heavy-atom interface;
the other three remain too far apart and two also contain chain breaks.  The
motif constraint itself remains exact in all four outputs.

The remaining A100 jobs subsequently completed inference.  Their original
workers stopped during result auditing because the early multi-design run
layout passed the merged engine input to a one-result constraint audit.  The
post-hoc audit path added in `4df4fcc` reconstructed each exact one-example
input and recovered all twelve results without rerunning RFD3.  Every fixed or
joint-rigid constraint orbit then passed; every guided result also passed its
mobility audit.  The full scientific result is nevertheless 0/12:

| mode | job | output | minimum edge distance (A) | minimum per-side coverage | minimum contiguous patch | shape loss | interface result |
|---|---|---:|---:|---:|---:|---:|---|
| locked | `5755477` | 0 | 11.457 | 0 | 0 | 1.821 | FAIL, 0/3 edges |
| locked | `5755477` | 1 | 3.503 | 2 | 2 | 0.470 | FAIL, 0/3 edges |
| locked | `5755478` | 0 | 15.030 | 0 | 0 | 3.505 | FAIL, 0/3 edges |
| locked | `5755478` | 1 | 15.970 | 0 | 0 | 4.162 | FAIL, 0/3 edges |
| locked | `5755479` | 0 | 5.328 | 2 | 2 | 0.282 | FAIL, 0/3 edges |
| locked | `5755479` | 1 | 11.869 | 0 | 0 | 2.160 | FAIL, 0/3 edges |
| guided | `5755482` | 0 | 11.745 | 0 | 0 | 2.039 | FAIL, 0/3 edges |
| guided | `5755482` | 1 | 3.505 | 2 | 2 | 0.477 | FAIL, 0/3 edges |
| guided | `5755483` | 0 | 15.956 | 0 | 0 | 3.836 | FAIL, 0/3 edges |
| guided | `5755483` | 1 | 17.329 | 0 | 0 | 4.831 | FAIL, 0/3 edges |
| guided | `5755484` | 0 | 5.806 | 2 | 1 | 0.244 | FAIL, 0/3 edges |
| guided | `5755484` | 1 | 12.639 | 0 | 0 | 2.419 | FAIL, 0/3 edges |

The audit repair therefore closes an engineering defect but does not alter the
scientific verdict.  Locked and guided inputs preserve their required motif
geometry, guided SE(3) updates execute, and most final scaffolds are continuous
and clash-free, yet no output forms even one accepted physical interface
edge.  Including the earlier H100 0/4 and RTX 3070 0/4 campaigns, the retained
generated-interface evidence is now 0/20.  The next change must improve actual
capture/continuity/shape formation; increasing sample count or weakening the
audits is not justified by these results.

### Geometric interpretation of the 0/12 A100 cohort

Visual inspection is consistent with the numerical failure: several outputs
place helix ends or narrow tips toward one another across the C3 neighbours.
That is not a broad protein interface.  A valid generated interface need not
be a coiled coil, but it must expose an extended side-by-side or otherwise
complementary surface containing several mutually supported residues on both
sides.  A small minimum distance at one tip cannot substitute for coverage,
continuity and shape complementarity; this is exactly why the closest A100
outputs still have only one or two contiguous residues and zero accepted
physical edges.

The current canaries do not provide a global rescue mechanism for this
geometry:

- the locked input declares no stochastic `sampling.initial_pose`; its
  supplied C3 arrangement therefore remains the single compiled arrangement,
  and only generated atoms may change during diffusion;
- the guided input also starts from that single compiled arrangement.  It
  preserves the selected motif internally and can move its complete master
  orbit only through the local `radial_axial_rotation` controller, currently
  bounded to 4 A and 10 degrees for `create_symmetric_interface`;
- exact C3 copies are always regenerated from the same master action, so they
  never move independently;
- local line search and rollback are intended to refine a nearby feasible
  interface, not convert a globally tip-facing arrangement into a new
  side-facing assembly.

Consequently, the A100 campaign proves that the runtime controller and audits
execute, but its pose is not a scientifically adequate generated-interface
canary.  The correct next gate is not a weaker audit or an arbitrarily larger
local drag.  It is a pre-diffusion population of independent, strictly
replayable C3 poses, filtered for an interface-facing corridor before RFD3,
followed by bounded runtime refinement.  Locked semantics remain useful only
when the user-supplied arrangement already passes that geometric feasibility
check; `optimize_components` is the route when the complete internally rigid
motif orbit is allowed to change radius and orientation.
