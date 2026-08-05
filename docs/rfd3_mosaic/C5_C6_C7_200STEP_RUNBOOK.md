# LHD101 C5/C6/C7 200-step capability runbook

## 8.3 low-tilt 200-step campaign

The `8.3` campaign freezes exactly one interface-seed pose for each of C5,
C6, and C7, then varies only the RFD3 diffusion seed. A low tilt means that
the seed principal axis is close to parallel to the declared cyclic axis:
the accepted interval is 0--15 degrees.

Selection is performed from each complete `pose_ensemble.json`, not from the
quality-diversity shortlist. A candidate must satisfy all of the following
before it can be ranked:

- `accepted = true`;
- `hard_clashes = 0`;
- `interface_ok = true` and `linker_ok = true`;
- `required_objective_failures = 0`;
- `0 <= maximum_principal_axis_tilt_deg <= 15`.

Among eligible candidates, the script minimizes objective penalty, then
scaffold endpoint span and mean span, preferring larger inter-group clearance
and lower tilt as later tie-breakers. It records the selected manifest and its
SHA256 in `selected_seed_interfaces.tsv`; subsequent invocations reuse this
frozen selection rather than silently choosing a different pose.

After synchronizing the scripts to LRZ, inspect the selected C5/C6/C7 poses
without submitting jobs:

```bash
RFD3_8_3_SELECT_ONLY=true \
bash scripts/rfd3_mosaic/submit_lhd101_cn_low_tilt_8_3.sh

column -t -s $'\t' \
  /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/8.3/selected_seed_interfaces.tsv
```

The production invocation plans 100 diffusion seeds per pose, or 300
independent 200-step jobs. It submits at most 36 jobs per invocation to avoid
the per-user Slurm submission limit; rerun the same command as queue capacity
becomes available:

```bash
bash scripts/rfd3_mosaic/submit_lhd101_cn_low_tilt_8_3.sh
```

The default partition pool includes the LRZ H100, A100-80GB, V100 and P100
partitions. Confirm the live names before the first submission, and override
the pool if any partition is unavailable to the account:

```bash
sinfo -h -o '%P | %G | %a' | grep -Ei 'h100|a100|v100|p100'

RFD3_8_3_PARTITIONS='lrz-hgx-h100-94x4,lrz-dgx-a100-80x8' \
bash scripts/rfd3_mosaic/submit_lhd101_cn_low_tilt_8_3.sh
```

All campaign state is isolated under:

```text
$RUN_BASE/8.3/selected_seed_interfaces.tsv
$RUN_BASE/8.3/submissions.tsv
$RUN_BASE/8.3/native_c5_full/<job-id>/
$RUN_BASE/8.3/native_c6_full/<job-id>/
$RUN_BASE/8.3/native_c7_full/<job-id>/
```

The submission table is resumable. Active, completed, and already audited
tasks are not duplicated. A failed task with no pair of audit reports is
treated as infrastructure-incomplete and can be retried once by default.
Scientific failures that produced both reports are retained and are not
automatically resubmitted. Production uses the validated
`explicit_all_copy` backend and low-memory mode.

An inspected pose can replace the automatic quality winner for one order,
but it must still pass every hard gate and the configured tilt interval. For
example, to freeze C5 pose seed 3070 while leaving C6/C7 automatic, remove an
old selection only before any campaign jobs have been submitted, then run:

```bash
rm /dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic/8.3/selected_seed_interfaces.tsv

RFD3_8_3_C5_POSE_SEED=3070 \
RFD3_8_3_SELECT_ONLY=true \
bash scripts/rfd3_mosaic/submit_lhd101_cn_low_tilt_8_3.sh
```

Once the replacement row is frozen, later submission invocations reuse it
without requiring the override variable again.

## C5 inter-chain-attention validation matrix

After the `>3`-chain sparse-attention row-index fix passes the complete unit
suite, run one controlled 12-job matrix before changing mobile motif behavior:

```text
3 previously screened C5 poses: 3063, 3458, 3145
x 50 and 200 diffusion steps
x P100 and H100
= 12 jobs, with diffusion seed fixed at 44
```

The three 50-step P100 jobs are paired with pre-fix jobs `5722375`, `5722380`
and `5722385`. The submission script assigns stable logical IDs `A01`--`A12`
and records every Slurm ID and condition in a timestamped TSV:

```bash
bash scripts/rfd3_mosaic/submit_c5_attention_validation_matrix.sh
```

This matrix tests attention-fix reproducibility across pose, diffusion depth
and accelerator. It does not test mobile motif updates, which remain disabled.

## Scope

This suite tests whether the validated exact cyclic sampler path generalizes
from C3 to C5, C6, and C7. It does not assume that any order is biologically or
structurally preferable.

Each order uses:

- the same fixed two-fragment LHD101 interface seed;
- native RFD3 `C<n>` symmetry;
- Haar-uniform SO(3) orientation sampling;
- joint Latin-hypercube pose sampling;
- the same static hard gates and quality-diversity selection;
- 200 diffusion steps in float32 exact-orbit, low-memory mode on P100;
- seed-integrity and transform-aware scaffold audits.

The placement radius is scaled as

```text
R_n = R_3 * sin(pi / 3) / sin(pi / n)
```

This preserves the C3 adjacent-copy chord range instead of forcing C5--C7
into the original C3 radius.

The absolute central-axis clearance objective window and its scale are
multiplied by the same order-dependent factor. QD morphology bins use
`minimum_axis_clearance / sampled_radius`, not raw Angstrom clearance, so C3,
C5, C6, and C7 are compared in a common dimensionless coordinate.

## Stage 1: CPU pose ensembles

Run all three orders from the repository root:

```bash
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic

for ORDER in 5 6 7; do
  RFD3_POSE_SAMPLES=512 \
  RFD3_POSE_SEED_START=3000 \
  RFD3_QD_MAX_SELECTED=16 \
  RFD3_QD_MIN_SO3_SEPARATION=25 \
  bash scripts/rfd3_mosaic/lhd101_cn_pose_qd.sh \
    "$ORDER" "$RUN_BASE/lhd101_c${ORDER}_qd_v2"
done
```

The shortlist for each order is:

```text
$RUN_BASE/lhd101_c5_qd_v2/pose_qd_shortlist.json
$RUN_BASE/lhd101_c6_qd_v2/pose_qd_shortlist.json
$RUN_BASE/lhd101_c7_qd_v2/pose_qd_shortlist.json
```

## Stage 2: choose one manifest per order

The following shell function prints the top QD candidate manifest without
requiring `jq`:

```bash
top_manifest () {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(Path(data["shortlist"][0]["directory"]) / "manifest.json")
PY
}
```

Resolve and inspect all three selections:

```bash
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic

for ORDER in 5 6 7; do
  SHORTLIST="$RUN_BASE/lhd101_c${ORDER}_qd_v2/pose_qd_shortlist.json"
  MANIFEST=$(top_manifest "$SHORTLIST")
  echo "C${ORDER}: $MANIFEST"
  python -m json.tool "$MANIFEST" >/dev/null
done
```

Top-QD selection is a reproducible compute-priority rule, not proof that the
pose will diffuse successfully.

## Stage 3: submit three independent 200-step P100 jobs

```bash
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic

for ORDER in 5 6 7; do
  SHORTLIST="$RUN_BASE/lhd101_c${ORDER}_qd_v2/pose_qd_shortlist.json"
  MANIFEST=$(top_manifest "$SHORTLIST")
  sbatch \
    --job-name="rfd3-c${ORDER}-200" \
    --export="ALL,RFD3_CYCLIC_ORDER=${ORDER},RFD3_NUM_TIMESTEPS=200,RFD3_POSE_CANDIDATE_MANIFEST=${MANIFEST}" \
    scripts/rfd3_mosaic/lhd101_cn_full_p100.sbatch
done
```

Record the three returned job IDs. Each run is written under:

```text
$RUN_BASE/native_c5_full/<job-id>
$RUN_BASE/native_c6_full/<job-id>
$RUN_BASE/native_c7_full/<job-id>
```

## Stage 4: required result checks

For each `ORDER` and `JOB_ID`:

```bash
RUN_BASE=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic
RUN_DIR="$RUN_BASE/native_c${ORDER}_full/$JOB_ID"

sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
grep -E \
  'RFD3 input construction:|RFD3_MOSAIC_FABRIC_PRECISION|Interface-Seed audit:|Scaffold audit:|Required result audits:|completed' \
  "$RUN_DIR"/slurm-*.out "$RUN_DIR"/slurm-*.err

python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
for name in ("seed_integrity_audit.json", "scaffold_validity_audit.json"):
    path = run / name
    data = json.loads(path.read_text())
    print(name, "passed=", data.get("passed"))
    print("summary=", data.get("summary"))
PY
```

A run counts as successful only if both audit reports have `passed: true`.
Visual inspection alone is not sufficient.

C5--C7 contain more tokens than the validated C3 baseline. Low-memory mode is
enabled, but P100 memory feasibility has not yet been established; an
out-of-memory result is a backend-capacity failure, not a geometry result.

## Stage 5: batch screen extracted structures

After copying result structures into each order's `extracted_cif` directory,
run all three screens and print a combined summary from the repository root:

```bash
bash scripts/rfd3_mosaic/screen_extracted_cn_structures.sh
```

`RFD3_SCREEN_ORDERS="5 7"` can restrict the run to selected orders. The
underlying single-directory command remains
`python -m rfd3_mosaic.rfd3_batch_screen`.

For a file named `<job-id>__<result>.cif`, the screen resolves the sibling
`<job-id>` run directory and reuses its adapter transform registry and
seed/scaffold reports. It writes one JSON report and one flat CSV per order:

```text
native_c<n>_full/extracted_cif/c<n>_batch_screen.json
native_c<n>_full/extracted_cif/c<n>_batch_screen.csv
```

The strict gate still requires the original seed audit plus recomputed
continuity, compactness, zero CA clashes, and declared-transform Cn symmetry.
Additional ring and packing fields are diagnostics for ranking:

- neighbouring-chain CA contacts below 8 A and contacts per residue;
- minimum inter-chain CA distance and non-neighbour contacts;
- fitted chain-COM ring radius and radial coefficient of variation;
- axial COM RMS and cyclic angular-gap error;
- minimum CA clearance from the fitted cyclic axis.

Higher contact count is not an independent success criterion: a collapsed or
clashing assembly can also have many contacts. Rank only after the hard gates,
and send shortlisted structures to sequence design and multimer prediction
before any biological claim.

The first completed screen on 2026-07-31 produced:

```text
C5: 12/15 strict passes
C6: 13/18 strict passes
C7: 12/15 strict passes
total: 37/48 strict passes
```

All 48 structures passed the seed audit and had their declared symmetry
available. The principal failure mode was CA clash; two structures also
failed continuity. For C6, do not automatically promote contact-ranked jobs
`5722400` and `5722401`: their minimum inter-chain CA distances are only
3.006 A and 3.229 A. Jobs `5722398` and `5722341` provide less borderline
packing controls at 3.940 A and 4.223 A.

## Separate C5 low-tilt mobility comparison

The formal C5/C6/C7 path above keeps the interface seed static. The
scaffold-aware mobility pilot is a separate opt-in experiment and must not be
mixed into the static capability claim.

For the retained experimental C5 pose seed `3419`, submit the paired
proposal/applied 50/200-step experiment on P100 with:

```bash
bash scripts/rfd3_mosaic/submit_lhd101_c5_mobile_pair_p100.sh
```

Each 200-step job has an `afterok` dependency on the 50-step job with the same
mobility mode. The exact rationale and interpretation are in
`docs/rfd3_mosaic/SCAFFOLD_AWARE_MOTIF_MOBILITY_PILOT.md`.

## Higher-order boundary

The generic schema, Cn/Dn registry, and instance compiler are parameterized and
can represent C12/C20. The local branch now removes both artificial
10-transform guards in the Mosaic adapter and Foundry motif-frame recovery,
with C12/D6 CPU regressions added. This is not yet equivalent to validated
high-order GPU inference. Full token-pair state grows quadratically with assembly
size; C12 is high risk and C20 is expected to be impractical on a 16-GB P100
with the current explicit all-copy representation. These are code- and
memory-scaling assessments, not completed C12/C20 GPU measurements. High-order
work therefore begins with LRZ CPU construction and a bounded C12 GPU probe;
C20 and larger groups require a separate local-neighborhood strategy,
non-factorial audits, and dedicated chain-encoding validation.

## H100 robustness screen

After the first convergence jobs, submit a controlled large-scale screen:

```text
C5/C6/C7 x top 3 QD poses x diffusion seeds 42--46 x 50 steps
= 45 independent jobs
```

Run:

```bash
bash scripts/rfd3_mosaic/submit_lhd101_cn_h100_screen.sh
```

The script writes a timestamped TSV under the run base with the order, pose
rank, pose seed, diffusion seed, step count, job ID, and exact manifest.
Screen success is defined by both required audit reports passing. Only
pose/order combinations with reproducible success should be promoted to
additional 200-step runs.

If LRZ stops submission at `QOSMaxSubmitJobPerUserLimit`, already submitted
jobs remain valid. Resume the same matrix after queue capacity is released:

```bash
RFD3_SCREEN_JOB_FILE=/absolute/path/to/cn_h100_screen_TIMESTAMP.tsv \
bash scripts/rfd3_mosaic/submit_lhd101_cn_h100_screen.sh
```

The resume path validates the TSV and skips every combination already
recorded, so it does not duplicate jobs.
