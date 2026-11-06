"""
Section 3.6: EER, DET curves, FNMR at fixed FMR operating points (1%, 0.1%),
and bootstrap confidence intervals from resampling test participants within
a fold. Scores follow the "higher = more genuine" convention used throughout
this codebase.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

FMR_OPERATING_POINTS = [0.01, 0.001]


def compute_eer(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels)) < 2:
        return np.nan, np.nan
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_fnmr_at_fmr(labels, scores, target_fmr):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    order = np.argsort(fpr)
    return float(np.interp(target_fmr, fpr[order], fnr[order]))


def det_curve_points(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    return fpr, fnr, thresholds


def bootstrap_eer_ci(df_cell, n_boot=500, seed=0):
    """df_cell: rows for one (mechanism, delta[, fold]) group, with columns
    participant_id, label, score. Resamples test participants with
    replacement and recomputes EER each time."""
    rng = np.random.RandomState(seed)
    participants = df_cell['participant_id'].unique()
    if len(participants) < 2:
        return np.nan, np.nan

    by_participant = {pid: g for pid, g in df_cell.groupby('participant_id')}
    boot_eers = []
    for _ in range(n_boot):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        parts = [by_participant[p] for p in sampled]
        merged = pd.concat(parts, ignore_index=True)
        eer, _ = compute_eer(merged['label'], merged['score'])
        if not np.isnan(eer):
            boot_eers.append(eer)
    if not boot_eers:
        return np.nan, np.nan
    lo, hi = np.percentile(boot_eers, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_by_mechanism_delta(trials_df, n_boot=500, seed=0):
    """One row per (mechanism, clipped, k_enrollment, delta): EER (mean +/- sd
    across folds) and a bootstrap CI computed by pooling all folds' trials for
    that cell, plus FNMR at the two ISO/IEC 19795-1 operating points."""
    rows = []
    group_cols = ['mechanism', 'clipped', 'k_enrollment', 'delta']
    for key, g in trials_df.groupby(group_cols):
        mechanism, clipped, k_enrollment, delta = key
        per_fold_eer = []
        for fold, gf in g.groupby('fold'):
            eer, _ = compute_eer(gf['label'], gf['score'])
            if not np.isnan(eer):
                per_fold_eer.append(eer)
        pooled_eer, _ = compute_eer(g['label'], g['score'])
        ci_lo, ci_hi = bootstrap_eer_ci(g, n_boot=n_boot, seed=seed)
        row = {
            'mechanism': mechanism, 'clipped': clipped, 'k_enrollment': k_enrollment, 'delta': delta,
            'n_trials': len(g), 'n_genuine': int((g['label'] == 1).sum()), 'n_impostor': int((g['label'] == 0).sum()),
            'eer_pooled': pooled_eer,
            'eer_mean_across_folds': float(np.mean(per_fold_eer)) if per_fold_eer else np.nan,
            'eer_sd_across_folds': float(np.std(per_fold_eer)) if per_fold_eer else np.nan,
            'eer_ci_lo': ci_lo, 'eer_ci_hi': ci_hi,
        }
        for target in FMR_OPERATING_POINTS:
            row[f'fnmr_at_fmr_{target}'] = compute_fnmr_at_fmr(g['label'], g['score'], target)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(['mechanism', 'clipped', 'k_enrollment', 'delta'])
    return out


def plot_aging_curves(summary_df, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for (mechanism, clipped), g in summary_df.groupby(['mechanism', 'clipped']):
        g = g.sort_values('delta')
        style = '-' if clipped == 'clipped' else '--'
        ax.plot(g['delta'], g['eer_pooled'] * 100, style, marker='o', label=f'{mechanism} ({clipped})')
        ax.fill_between(g['delta'], g['eer_ci_lo'] * 100, g['eer_ci_hi'] * 100, alpha=0.1)
    ax.set_xlabel('Temporal gap Delta (weeks)')
    ax.set_ylabel('EER (%)')
    ax.set_title('Template-aging curve: EER vs. enrolment-to-query gap')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_det_curves(trials_df, out_path, k_enrollment, clipped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = trials_df[(trials_df['k_enrollment'] == k_enrollment) & (trials_df['clipped'] == clipped)]
    fig, ax = plt.subplots(figsize=(6, 6))
    for mechanism, g in subset.groupby('mechanism'):
        fpr, fnr, _ = det_curve_points(g['label'], g['score'])
        ax.plot(fpr * 100, fnr * 100, label=mechanism)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('False Match Rate (%)')
    ax.set_ylabel('False Non-Match Rate (%)')
    ax.set_title(f'DET curves (k={k_enrollment}, {clipped})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
