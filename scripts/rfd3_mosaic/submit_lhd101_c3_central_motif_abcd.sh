#!/bin/bash

set -euo pipefail

PROJECT_DIR=/dss/dssfs02/lwp-dss-0001/pn57ki/pn57ki-dss-0000/haixi/projects/rfd3-mosaic
SBATCH_SCRIPT="$PROJECT_DIR/scripts/rfd3_mosaic/lhd101_c3_central_motif_probe_p100.sbatch"
TEMPLATE_INPUT="${RFD3_CENTRAL_TEMPLATE_INPUT:?Set RFD3_CENTRAL_TEMPLATE_INPUT to an existing C3 adapter rfd3_input.json}"
RUN_TAG="${RFD3_CENTRAL_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
JOB_FILE="$PROJECT_DIR/central_motif_c3_abcd_${RUN_TAG}.tsv"
ARMS="${RFD3_CENTRAL_ARMS:-A B C D}"

if [[ ! -f "$TEMPLATE_INPUT" ]]; then
    echo "ERROR: template input does not exist: $TEMPLATE_INPUT"
    exit 2
fi

printf 'arm\tjob_id\tdiagnosis\ttemplate_input\n' >"$JOB_FILE"
for arm in $ARMS; do
    case "$arm" in
        A) diagnosis=official_original_realign ;;
        B) diagnosis=exact_mosaic ;;
        C) diagnosis=official_original_no_realign ;;
        D) diagnosis=legacy_complete_orbit_restore ;;
        *)
            echo "ERROR: RFD3_CENTRAL_ARMS may contain only A, B, C, or D"
            exit 2
            ;;
    esac
    submit_output=$(
        sbatch \
            --job-name="c3-cent-${arm,,}" \
            --export="ALL,RFD3_CENTRAL_PROBE_ARM=$arm" \
            "$SBATCH_SCRIPT"
    )
    job_id=${submit_output##* }
    printf '%s\t%s\t%s\t%s\n' \
        "$arm" "$job_id" "$diagnosis" "$TEMPLATE_INPUT" | tee -a "$JOB_FILE"
done

echo "Job manifest: $JOB_FILE"
column -t -s $'\t' "$JOB_FILE" 2>/dev/null || cat "$JOB_FILE"
