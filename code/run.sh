#!/usr/bin/env bash
# Reproduce the analysis.
#
#   bash code/run.sh                        quick verification, ~5 minutes
#   REPRODUCTION_MODE=full bash code/run.sh  full reproduction, ~10-20 h on CPU
#
# Paths default to ./data and ./results relative to the repository root, so a
# fresh clone runs with no configuration. Override with KD_DATA_DIR and
# KD_RESULT_DIR if you keep the data elsewhere.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
export KD_DATA_DIR="${KD_DATA_DIR:-$ROOT/data}"
export KD_RESULT_DIR="${KD_RESULT_DIR:-$ROOT/results}"
export MPLBACKEND=Agg
MODE="${REPRODUCTION_MODE:-smoke}"

mkdir -p "$KD_RESULT_DIR"
cd "$HERE"

echo "mode=$MODE  data=$KD_DATA_DIR  results=$KD_RESULT_DIR"

if [ "$MODE" = "full" ]; then
    python run_experiments.py --seed 0 --n-ensemble-seeds 3 \
        --device "${KD_DEVICE:-cpu}" --output-dir "$KD_RESULT_DIR"

    for S in 10 20; do
        python run_experiments.py --seed "$S" --n-ensemble-seeds 3 \
            --device "${KD_DEVICE:-cpu}" --output-dir "$KD_RESULT_DIR/replicate_seed$S"
    done

    python seed_robustness_check.py --device "${KD_DEVICE:-cpu}"
    python glmm_logit_clusterrobust.py
    python glmm_sensitivity_check.py || \
        echo "[note] variational-Bayes GLMM did not converge (expected; see paper)"
    python participant_balanced_eer.py
    python generate_report.py
    python generate_paper_figures.py
else
    # One fold, one seed, reduced training budget. Exercises every stage of the
    # pipeline. Absolute error rates will NOT match the paper.
    python run_experiments.py \
        --folds 0 --k-values 5 --clip-conditions clipped \
        --max-impostors 3 --epochs 2 --steps-per-epoch 10 \
        --n-ensemble-seeds 1 --device "${KD_DEVICE:-cpu}" \
        --output-dir "$KD_RESULT_DIR"
fi

echo
echo "Done. Outputs in $KD_RESULT_DIR"
