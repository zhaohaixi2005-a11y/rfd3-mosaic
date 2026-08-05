#!/bin/bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE=${RFD3_RUN_BASE:-/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/runs/rfd3-mosaic}
CAMPAIGN=${RFD3_8_3_CAMPAIGN:-8.3}
CAMPAIGN_ROOT="$RUN_BASE/$CAMPAIGN"
ENSEMBLE_TAG=${RFD3_8_3_ENSEMBLE_TAG:-qd_v2}
ORDERS=${RFD3_8_3_ORDERS:-"5 6 7"}
TILT_MIN=${RFD3_8_3_TILT_MIN:-0}
TILT_MAX=${RFD3_8_3_TILT_MAX:-15}
C5_POSE_SEED=${RFD3_8_3_C5_POSE_SEED:-auto}
C6_POSE_SEED=${RFD3_8_3_C6_POSE_SEED:-auto}
C7_POSE_SEED=${RFD3_8_3_C7_POSE_SEED:-auto}
NUM_TIMESTEPS=${RFD3_8_3_NUM_TIMESTEPS:-200}
DIFFUSION_SEED_START=${RFD3_8_3_DIFFUSION_SEED_START:-8300}
DIFFUSION_SEED_COUNT=${RFD3_8_3_DIFFUSION_SEED_COUNT:-100}
MAX_SUBMISSIONS=${RFD3_8_3_MAX_SUBMISSIONS_PER_RUN:-36}
MAX_ATTEMPTS=${RFD3_8_3_MAX_ATTEMPTS:-2}
PARTITIONS=${RFD3_8_3_PARTITIONS:-lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-dgx-1-v100x8,lrz-v100x2,lrz-dgx-1-p100x8,lrz-hpe-p100x4}
WALLTIME=${RFD3_8_3_WALLTIME:-24:00:00}
BACKEND=${RFD3_8_3_SYMMETRY_BACKEND:-explicit_all_copy}
NEIGHBOUR_RADIUS=${RFD3_8_3_NEIGHBOUR_RADIUS:-1}
SELECT_ONLY=${RFD3_8_3_SELECT_ONLY:-false}
SELECTION_FILE="$CAMPAIGN_ROOT/selected_seed_interfaces.tsv"
JOB_FILE=${RFD3_8_3_JOB_FILE:-"$CAMPAIGN_ROOT/submissions.tsv"}
SBATCH_SCRIPT="$PROJECT_DIR/scripts/rfd3_mosaic/lhd101_cn_full_p100.sbatch"

if [[ ! "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || [[ "$CAMPAIGN" == "." || "$CAMPAIGN" == ".." ]]; then
    echo "ERROR: RFD3_8_3_CAMPAIGN must be one safe directory name"
    exit 2
fi
if [[ ! "$NUM_TIMESTEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( NUM_TIMESTEPS < 2 || NUM_TIMESTEPS > 200 )); then
    echo "ERROR: RFD3_8_3_NUM_TIMESTEPS must be from 2 to 200"
    exit 2
fi
for value_name in \
    DIFFUSION_SEED_START DIFFUSION_SEED_COUNT MAX_SUBMISSIONS MAX_ATTEMPTS; do
    value=${!value_name}
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $value_name must be a non-negative integer"
        exit 2
    fi
done
if (( DIFFUSION_SEED_COUNT < 1 )); then
    echo "ERROR: RFD3_8_3_DIFFUSION_SEED_COUNT must be positive"
    exit 2
fi
if (( MAX_ATTEMPTS < 1 )); then
    echo "ERROR: RFD3_8_3_MAX_ATTEMPTS must be positive"
    exit 2
fi
if [[ "$SELECT_ONLY" != "true" && "$SELECT_ONLY" != "false" ]]; then
    echo "ERROR: RFD3_8_3_SELECT_ONLY must be true or false"
    exit 2
fi
if [[ "$BACKEND" != "explicit_all_copy" \
    && "$BACKEND" != "local_neighbourhood" ]]; then
    echo "ERROR: unsupported symmetry backend: $BACKEND"
    exit 2
fi
if [[ ! "$NEIGHBOUR_RADIUS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RFD3_8_3_NEIGHBOUR_RADIUS must be non-negative"
    exit 2
fi
for value_name in C5_POSE_SEED C6_POSE_SEED C7_POSE_SEED; do
    value=${!value_name}
    if [[ "$value" != "auto" && ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $value_name must be auto or a non-negative pose seed"
        exit 2
    fi
done
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: batch script is missing: $SBATCH_SCRIPT"
    exit 2
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/models/rfd3/src:${PYTHONPATH:-}"
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "ERROR: neither python nor python3 is available"
    exit 2
fi
mkdir -p "$CAMPAIGN_ROOT"

EXPECTED_SELECTION_HEADER=$'order\tpose_rank\tpose_seed\ttilt_deg\tobjective_penalty\tmaximum_scaffold_span\tminimum_inter_group_distance\tmanifest_sha256\tmanifest'
if [[ -f "$SELECTION_FILE" ]]; then
    if [[ "$(head -n 1 "$SELECTION_FILE")" != "$EXPECTED_SELECTION_HEADER" ]]; then
        echo "ERROR: incompatible selection file: $SELECTION_FILE"
        exit 2
    fi
    echo "Reusing frozen low-tilt selection: $SELECTION_FILE"
else
    SELECTION_TMP=$(mktemp "$CAMPAIGN_ROOT/.selected_seed_interfaces.XXXXXX")
    trap 'rm -f "$SELECTION_TMP"' EXIT
    if ! "$PYTHON_BIN" - \
        "$RUN_BASE" \
        "$ENSEMBLE_TAG" \
        "$TILT_MIN" \
        "$TILT_MAX" \
        "$C5_POSE_SEED" \
        "$C6_POSE_SEED" \
        "$C7_POSE_SEED" \
        $ORDERS >"$SELECTION_TMP" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path


def optional_float(value, *, missing):
    return missing if value is None else float(value)


run_base = Path(sys.argv[1])
ensemble_tag = sys.argv[2]
tilt_min = float(sys.argv[3])
tilt_max = float(sys.argv[4])
pose_overrides = {
    order: None if value == "auto" else int(value)
    for order, value in zip((5, 6, 7), sys.argv[5:8])
}
orders = [int(value) for value in sys.argv[8:]]
if not orders or any(order not in (5, 6, 7) for order in orders):
    raise SystemExit("ERROR: orders must be selected from 5, 6, and 7")
if not 0.0 <= tilt_min <= tilt_max <= 90.0:
    raise SystemExit("ERROR: tilt interval must lie inside [0, 90] degrees")

print(
    "order",
    "pose_rank",
    "pose_seed",
    "tilt_deg",
    "objective_penalty",
    "maximum_scaffold_span",
    "minimum_inter_group_distance",
    "manifest_sha256",
    "manifest",
    sep="\t",
)
for order in orders:
    ensemble_path = (
        run_base
        / f"lhd101_c{order}_{ensemble_tag}"
        / "pose_ensemble.json"
    )
    if not ensemble_path.is_file():
        raise SystemExit(f"ERROR: missing C{order} ensemble: {ensemble_path}")
    data = json.loads(ensemble_path.read_text())
    eligible = []
    for rank, candidate in enumerate(data["ranking"], start=1):
        tilt = candidate.get("maximum_principal_axis_tilt_deg")
        requested_pose_seed = pose_overrides[order]
        if (
            not candidate.get("accepted", False)
            or (
                requested_pose_seed is not None
                and int(candidate["pose_seed"]) != requested_pose_seed
            )
            or tilt is None
            or not tilt_min <= float(tilt) <= tilt_max
            or int(candidate.get("hard_clashes", -1)) != 0
            or not candidate.get("interface_ok", False)
            or not candidate.get("linker_ok", False)
            or int(candidate.get("required_objective_failures", -1)) != 0
        ):
            continue
        eligible.append((rank, candidate))
    if not eligible:
        override_note = (
            ""
            if pose_overrides[order] is None
            else f" for requested pose seed {pose_overrides[order]}"
        )
        raise SystemExit(
            f"ERROR: C{order} has no accepted {tilt_min:g}-{tilt_max:g} "
            f"degree candidate{override_note} in {ensemble_path}; "
            "expand the LHS ensemble or choose another pose seed "
            "instead of relaxing hard geometry gates"
        )
    rank, candidate = min(
        eligible,
        key=lambda item: (
            float(item[1]["objective_penalty"]),
            optional_float(
                item[1].get("maximum_linker_endpoint_distance"),
                missing=math.inf,
            ),
            optional_float(
                item[1].get("mean_linker_endpoint_distance"),
                missing=math.inf,
            ),
            -optional_float(
                item[1].get("minimum_inter_group_distance"),
                missing=-math.inf,
            ),
            float(item[1]["maximum_principal_axis_tilt_deg"]),
            int(item[1]["pose_seed"]),
        ),
    )
    manifest = Path(candidate["directory"]) / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"ERROR: missing C{order} manifest: {manifest}")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    print(
        order,
        rank,
        candidate["pose_seed"],
        f'{float(candidate["maximum_principal_axis_tilt_deg"]):.6f}',
        f'{float(candidate["objective_penalty"]):.8f}',
        f'{optional_float(candidate.get("maximum_linker_endpoint_distance"), missing=math.nan):.6f}',
        f'{optional_float(candidate.get("minimum_inter_group_distance"), missing=math.nan):.6f}',
        manifest_sha,
        manifest,
        sep="\t",
    )
PY
    then
        echo "ERROR: low-tilt selection failed; no frozen selection was installed"
        exit 1
    fi
    mv "$SELECTION_TMP" "$SELECTION_FILE"
    trap - EXIT
fi

echo "Selected seed interfaces:"
column -t -s $'\t' "$SELECTION_FILE" 2>/dev/null || cat "$SELECTION_FILE"

PLANNED_COUNT=$(( DIFFUSION_SEED_COUNT * $(awk 'END { print NR - 1 }' "$SELECTION_FILE") ))
echo "Campaign root: $CAMPAIGN_ROOT"
echo "Planned 200-step designs: $PLANNED_COUNT"
echo "Partition pool: $PARTITIONS"
if [[ "$SELECT_ONLY" == "true" ]]; then
    echo "Selection-only mode: no jobs were submitted."
    exit 0
fi

EXPECTED_JOB_HEADER=$'campaign\torder\tpose_rank\tpose_seed\ttilt_deg\tmanifest_sha256\tdiffusion_seed\tsteps\tpartition_pool\tbackend\tattempt\tjob_id\tmanifest'
if [[ -f "$JOB_FILE" ]]; then
    if [[ "$(head -n 1 "$JOB_FILE")" != "$EXPECTED_JOB_HEADER" ]]; then
        echo "ERROR: incompatible job file: $JOB_FILE"
        exit 2
    fi
    echo "Resuming submissions from: $JOB_FILE"
else
    printf '%s\n' "$EXPECTED_JOB_HEADER" >"$JOB_FILE"
fi

SUBMITTED=0
for (( seed_offset=0; seed_offset<DIFFUSION_SEED_COUNT; seed_offset++ )); do
    DIFFUSION_SEED=$(( DIFFUSION_SEED_START + seed_offset ))
    while IFS=$'\t' read -r \
        ORDER POSE_RANK POSE_SEED TILT PENALTY MAX_SPAN MIN_DISTANCE \
        MANIFEST_SHA MANIFEST; do
        if [[ ! -f "$MANIFEST" ]]; then
            echo "ERROR: selected manifest no longer exists: $MANIFEST"
            exit 2
        fi
        OBSERVED_MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
        if [[ "$OBSERVED_MANIFEST_SHA" != "$MANIFEST_SHA" ]]; then
            echo "ERROR: selected manifest changed after freezing: $MANIFEST"
            echo "expected=$MANIFEST_SHA observed=$OBSERVED_MANIFEST_SHA"
            exit 2
        fi

        PREVIOUS_RECORD=$(awk -F '\t' \
            -v campaign="$CAMPAIGN" \
            -v order="$ORDER" \
            -v manifest_sha="$MANIFEST_SHA" \
            -v diffusion_seed="$DIFFUSION_SEED" \
            -v steps="$NUM_TIMESTEPS" \
            'NR > 1 && $1 == campaign && $2 == order &&
             $6 == manifest_sha && $7 == diffusion_seed && $8 == steps {
                 print $11 "\t" $12
             }
            ' "$JOB_FILE" | tail -n 1)
        ATTEMPT=1
        if [[ -n "$PREVIOUS_RECORD" ]]; then
            IFS=$'\t' read -r PREVIOUS_ATTEMPT PREVIOUS_JOB_ID \
                <<<"$PREVIOUS_RECORD"
            PREVIOUS_RUN_DIR="$CAMPAIGN_ROOT/native_c${ORDER}_full/$PREVIOUS_JOB_ID"
            if [[ -f "$PREVIOUS_RUN_DIR/seed_integrity_audit.json" \
                && -f "$PREVIOUS_RUN_DIR/scaffold_validity_audit.json" ]]; then
                echo "Skipping audited task C${ORDER}/seed=${DIFFUSION_SEED}: job $PREVIOUS_JOB_ID"
                continue
            fi

            PREVIOUS_STATE=$(sacct -X -j "$PREVIOUS_JOB_ID" \
                --noheader --parsable2 --format=State 2>/dev/null \
                | awk -F '|' 'NF { print $1; exit }' || true)
            PREVIOUS_STATE=${PREVIOUS_STATE%% *}
            PREVIOUS_STATE=${PREVIOUS_STATE%%+}
            case "$PREVIOUS_STATE" in
                PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING|COMPLETED)
                    echo "Skipping ${PREVIOUS_STATE} task C${ORDER}/seed=${DIFFUSION_SEED}: job $PREVIOUS_JOB_ID"
                    continue
                    ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED)
                    if (( PREVIOUS_ATTEMPT >= MAX_ATTEMPTS )); then
                        echo "Retry limit reached for C${ORDER}/seed=${DIFFUSION_SEED}: job $PREVIOUS_JOB_ID state=$PREVIOUS_STATE"
                        continue
                    fi
                    ATTEMPT=$(( PREVIOUS_ATTEMPT + 1 ))
                    echo "Retrying infrastructure-incomplete task C${ORDER}/seed=${DIFFUSION_SEED}: previous job $PREVIOUS_JOB_ID state=$PREVIOUS_STATE attempt=$ATTEMPT"
                    ;;
                "")
                    echo "Skipping task with unavailable accounting state C${ORDER}/seed=${DIFFUSION_SEED}: job $PREVIOUS_JOB_ID"
                    continue
                    ;;
                *)
                    echo "Skipping task with unclassified state C${ORDER}/seed=${DIFFUSION_SEED}: job $PREVIOUS_JOB_ID state=$PREVIOUS_STATE"
                    continue
                    ;;
            esac
        fi
        if (( MAX_SUBMISSIONS > 0 && SUBMITTED >= MAX_SUBMISSIONS )); then
            echo "Reached per-run submission limit: $MAX_SUBMISSIONS"
            echo "Resume later with the same command; completed keys are recorded in $JOB_FILE"
            exit 0
        fi
        JOB_NAME="83-c${ORDER}-p${POSE_SEED}-d${DIFFUSION_SEED}"
        if ! JOB_ID=$(sbatch --parsable \
                --partition="$PARTITIONS" \
                --job-name="$JOB_NAME" \
                --time="$WALLTIME" \
                --cpus-per-task=8 \
                --export="ALL,RFD3_RUN_BASE=${RUN_BASE},RFD3_RUN_CAMPAIGN=${CAMPAIGN},RFD3_ACCELERATOR_LABEL=AUTO,RFD3_CYCLIC_ORDER=${ORDER},RFD3_NUM_TIMESTEPS=${NUM_TIMESTEPS},RFD3_SEED=${DIFFUSION_SEED},RFD3_POSE_CANDIDATE_MANIFEST=${MANIFEST},RFD3_SYMMETRY_EXECUTION_BACKEND=${BACKEND},RFD3_SYMMETRY_NEIGHBOUR_RADIUS=${NEIGHBOUR_RADIUS}" \
                "$SBATCH_SCRIPT"); then
            echo "Submission stopped, usually because of the Slurm/QOS limit."
            echo "Resume later with the same command; successful submissions are recorded in $JOB_FILE"
            exit 75
        fi
        JOB_ID=${JOB_ID%%;*}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$CAMPAIGN" "$ORDER" "$POSE_RANK" "$POSE_SEED" "$TILT" \
            "$MANIFEST_SHA" "$DIFFUSION_SEED" "$NUM_TIMESTEPS" \
            "$PARTITIONS" "$BACKEND" "$ATTEMPT" "$JOB_ID" "$MANIFEST" \
            | tee -a "$JOB_FILE"
        SUBMITTED=$(( SUBMITTED + 1 ))
    done < <(tail -n +2 "$SELECTION_FILE")
done

echo "All planned jobs have been submitted and recorded in: $JOB_FILE"
