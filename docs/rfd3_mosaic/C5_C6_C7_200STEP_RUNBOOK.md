# LHD101 C5/C6/C7 200-step capability runbook

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
can represent C12/C20. That is not equivalent to native RFD3 inference
support. The current native path has two independent hard guards:

- Mosaic rejects symmetric-motif multiplicity greater than 10.
- Official Foundry RFD3 also defines `MAX_TRANSFORMS = 10` during motif-frame
  recovery.

Consequently C12/C20 are deliberately excluded from this runbook. Even after
removing both guards, full token-pair state grows quadratically with assembly
size; C12 is high risk and C20 is expected to be impractical on a 16-GB P100
with the current explicit all-copy representation. These are code- and
memory-scaling assessments, not completed C12/C20 GPU measurements. High-order
work requires a separate ASU/local-neighborhood strategy, non-factorial audits,
and dedicated chain-encoding validation rather than a larger value in this
submission script.

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
