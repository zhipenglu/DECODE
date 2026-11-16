#!/bin/bash
#SBATCH --job-name=decode_map
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/decode_map_%A_%a.out
#SBATCH --error=logs/decode_map_%A_%a.err

# =============================================================================
# DECODE Mappability Step 3: Parallel (K, E)-Mappability Sweep
# =============================================================================
# Runs GenMap across 13 read lengths (K = 20..1000 nt) in both 4-base (A/C/G/T)
# and 2-base (G/T) alphabets at mismatch tolerances E = 0 and E = 1 (48 total
# tracks; E = 1 is omitted for ultra-short K = 20, 30 nt).
#
# Raw BedGraph tracks are written to node-local scratch, summarized across the
# five biotype BED strata into `stats/`, and cleaned up immediately to keep
# disk usage minimal.
#
# Usage:
#   sbatch mappability/03_run_genmap_array.slurm
# =============================================================================

set -euo pipefail

REFS_DIR="${REFS_DIR:-refs_dual_strand}"
IDX_DIR="${IDX_DIR:-genmap_indices}"
STATS_DIR="${STATS_DIR:-stats}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Locate Step 4 summarization script whether submitted from repo root or subdir
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/mappability/04_summarize_and_plot_mappability.py" ]]; then
    SUM_SCRIPT="${SUBMIT_DIR}/mappability/04_summarize_and_plot_mappability.py"
elif [[ -f "${SUBMIT_DIR}/04_summarize_and_plot_mappability.py" ]]; then
    SUM_SCRIPT="${SUBMIT_DIR}/04_summarize_and_plot_mappability.py"
else
    echo "ERROR: Cannot locate 04_summarize_and_plot_mappability.py" >&2
    exit 1
fi

K_LIST=(20 30 40 50 75 100 150 200 250 300 500 750 1000)
K="${K_LIST[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p "${STATS_DIR}" logs
WORK_TMP="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/genmap_K${K}_XXXXXX")"
trap 'rm -rf "${WORK_TMP}"' EXIT

run_track() {
    local alphabet="$1"
    local idx_path="$2"
    local errors="$3"
    local prefix="${WORK_TMP}/${alphabet}_K${K}_E${errors}"
    local out_tsv="${STATS_DIR}/stats_${alphabet}_K${K}_E${errors}.tsv"

    echo "[$(date)] Running GenMap: alphabet=${alphabet}, K=${K}, E=${errors}"
    genmap map \
        -K "${K}" \
        -E "${errors}" \
        -I "${idx_path}" \
        -O "${prefix}" \
        -T "${THREADS}" \
        -bg

    python3 "${SUM_SCRIPT}" summarize \
        --bedgraph "${prefix}.bedgraph" \
        --beds_dir "${REFS_DIR}/beds" \
        --alphabet "${alphabet}" \
        --k "${K}" \
        --e "${errors}" \
        --out "${out_tsv}"

    rm -f "${prefix}.bedgraph"
}

# E = 0 exact mappability for all K
run_track "4base" "${IDX_DIR}/idx_4base" 0
run_track "2base_GT" "${IDX_DIR}/idx_2base_GT" 0

# E = 1 mismatch-tolerant mappability for K >= 40 nt
if [[ "${K}" -ge 40 ]]; then
    run_track "4base" "${IDX_DIR}/idx_4base" 1
    run_track "2base_GT" "${IDX_DIR}/idx_2base_GT" 1
fi

echo "[$(date)] Completed all tracks for K=${K}."
