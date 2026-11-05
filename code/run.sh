#!/usr/bin/env bash
# Reproduce the analysis.
#
#   bash code/run.sh                        quick verification, ~10 minutes
#   REPRODUCTION_MODE=full bash code/run.sh  full reproduction, ~20-30 h on CPU
#
# Paths default to ./data and ./results relative to the repository root, so a
# fresh clone runs with no configuration. Override with KD_DATA_DIR and
# KD_RESULT_DIR if you keep the data elsewhere, and KD_FIG_DIR to send figures
# somewhere other than results/figures.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
export KD_DATA_DIR="${KD_DATA_DIR:-$ROOT/data}"
export KD_RESULT_DIR="${KD_RESULT_DIR:-$ROOT/results}"
export MPLBACKEND=Agg
MODE="${REPRODUCTION_MODE:-smoke}"
DEV="${KD_DEVICE:-cpu}"

mkdir -p "$KD_RESULT_DIR"
cd "$HERE"

echo "mode=$MODE  device=$DEV  data=$KD_DATA_DIR  results=$KD_RESULT_DIR"

if [ "$MODE" = "full" ]; then

    # ---- primary run and the two independent replicates -------------------
    python run_experiments.py --seed 0 --n-ensemble-seeds 3 \
        --device "$DEV" --output-dir "$KD_RESULT_DIR"

    for S in 10 20; do
        python run_experiments.py --seed "$S" --n-ensemble-seeds 3 \
            --device "$DEV" --output-dir "$KD_RESULT_DIR/replicate_seed$S"
    done

    # ---- published robustness checks --------------------------------------
    python seed_robustness_check.py --device "$DEV"
    python glmm_logit_clusterrobust.py
    python glmm_sensitivity_check.py || \
        echo "[note] variational-Bayes GLMM did not converge (expected; see paper)"
    python participant_balanced_eer.py

    # ---- M5, the ITAD instance-based metric  (Sections 3.6, 4.2-4.4) ------
    # exp_itad.py benchmarks ITAD on its own. exp_itad_integrate.py then
    # scores it across the full grid, so M5 enters the same tables, figures
    # and mixed-effects model as M1 to M4 rather than sitting apart.
    python exp_itad.py
    python exp_itad_integrate.py
    python itad_appendix_rows.py
    python refit_model_with_itad.py

    # ---- how much of the signal is dwell, and where dwell comes from ------
    # Establishes the feature weighting used for M5, and answers the
    # reviewer question on the dwell channel.
    python exp_feature_ablation.py
    python exp_dwell_provenance.py

    # ---- mixed-effects model extensions  (Section 4.4) --------------------
    # Password length by gap, and enrolment week, each held to the same bar
    # as the published terms: three replicates and a change of link function.
    for R in primary seed10 seed20; do
        python exp_model_extensions.py --replicate "$R"
    done
    python exp_model_robustness.py
    python summarise_model_extensions.py
    python exp_length_by_mechanism.py

    # ---- learning effects against template ageing  (Section 4.6) ---------
    python exp_familiarisation.py

    # ---- a threshold fixed at enrolment  (Section 4.7) -------------------
    python exp_fixed_threshold.py
    python plot_fixed_threshold.py

    # ---- unseen passwords  (Section 4.8) ---------------------------------
    # Both halves of the password split are needed: the swapped run is what
    # separates password-set difficulty from memorisation.
    python exp_password_disjoint.py --device "$DEV"
    python exp_password_disjoint.py --device "$DEV" --swap --tag swapped
    python analyse_password_disjoint.py
    python estimate_memorisation.py

    # ---- why M3 ages more slowly  (Section 5.1) --------------------------
    python exp_why_m3_ages_slowly.py
    python exp_m3_slope_prediction.py
    python exp_floor_effect.py

    # ---- report and figures ----------------------------------------------
    python generate_report.py
    python generate_paper_figures.py
    python regenerate_bw_figures.py
    python make_single_column_figures.py

else
    # One fold, one seed, reduced training budget. Exercises every stage of
    # the pipeline, including the M5 matcher. Absolute error rates will NOT
    # match the paper.
    python run_experiments.py \
        --folds 0 --k-values 5 --clip-conditions clipped \
        --max-impostors 3 --epochs 2 --steps-per-epoch 10 \
        --n-ensemble-seeds 1 --device "$DEV" \
        --output-dir "$KD_RESULT_DIR"

    python exp_itad.py --folds 0 --fusions equal
    python exp_feature_ablation.py --folds 0
fi

echo
echo "Done. Outputs in $KD_RESULT_DIR"
