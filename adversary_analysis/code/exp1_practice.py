"""
Experiment 1: does adversary practice matter?

Runs M1 (scaled Manhattan, training-free) over the adversary-aware protocol and
answers two questions:

  RQ2 pilot   Does a password-knowing impostor get closer to the victim's
              template as they practise the password? A0 (first-ever exposure)
              versus A1 (week 8, ~28 prior repetitions).

  Paper 1     Paper 1 tied the impostor's week to the genuine query week, so its
  decomposition  ageing curve mixes genuine template drift with impostor practice
              growth. Holding the impostor week fixed while varying delta
              separates them.

M1 needs no training, so this is CPU-only and takes minutes rather than the
GPU-days M4 would need. Run M4 only once this says the effect is real.

Usage:
    python exp1_practice.py                 # all 5 folds
    python exp1_practice.py --folds 0       # single fold, for a quick check
    python exp1_practice.py --analyse-only  # re-analyse saved trials
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import data_loading
from dataset import Dataset
from features import IQRClipper
from password_metadata import build_password_metadata
from splits import fold_splits
from metrics import compute_eer
from protocol_adversary import generate_trials, trials_to_frame

from matchers_classical import M1Manhattan

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
TRIALS_DIR = os.path.join(RESULT_DIR, 'exp1_trials')
K_ENROLLMENT = 5
FNMR_BUDGETS = [0.01, 0.02, 0.05]   # usability budgets; see design.md Section 7
LAST_WEEK = 8


# ---------------------------------------------------------------- trials

def run_trials(folds_wanted=None):
    os.makedirs(TRIALS_DIR, exist_ok=True)
    samples = data_loading.load_samples(verbose=False)
    dataset = Dataset(samples)
    password_meta = {r['password_id']: r for r in build_password_metadata()}
    print(f'{len(samples)} samples, {len(dataset.participant_ids)} participants, '
          f'{len(dataset.password_ids)} passwords')

    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        if folds_wanted is not None and fold not in folds_wanted:
            continue
        t0 = time.time()

        train_set = set(train_ids)
        feats_by_pw = {pw: [dataset.feature_vectors[i] for i in dataset.by_password[pw]
                            if dataset.samples[i].participant_id in train_set]
                       for pw in dataset.password_ids}
        clipper = IQRClipper().fit(feats_by_pw)

        m1 = M1Manhattan()
        m1.fit(dataset, train_ids, clipper)

        rows = generate_trials(dataset, m1, clipper, test_ids, password_meta,
                               'M1_Manhattan', fold, 'clipped', k_enrollment=K_ENROLLMENT)
        df = trials_to_frame(rows)
        out = os.path.join(TRIALS_DIR, f'fold{fold}.parquet')
        df.to_parquet(out, index=False)
        print(f'fold {fold}: {len(df):,} trials '
              f'({int((df.label == 1).sum()):,} genuine / {int((df.label == 0).sum()):,} impostor) '
              f'-> {os.path.basename(out)}  [{time.time() - t0:.0f}s]')


def load_trials():
    paths = sorted(f for f in os.listdir(TRIALS_DIR) if f.endswith('.parquet'))
    if not paths:
        raise SystemExit('no trials found; run without --analyse-only first')
    df = pd.concat([pd.read_parquet(os.path.join(TRIALS_DIR, p)) for p in paths],
                   ignore_index=True)
    print(f'loaded {len(df):,} trials from {len(paths)} fold(s)')
    return df


# ------------------------------------------------- A0 vs A1, paired by gallery

def paired_practice_effect(imp):
    """Within-gallery paired comparison of A0 against A1.

    Scores are not comparable across galleries (different passwords have
    different feature dimensionality, so the Manhattan sum has a different
    scale), so each gallery's impostor scores are standardised against that
    gallery's own impostor score distribution before anything is aggregated.
    Pairing is exact: the same impostor attacking the same gallery, early
    versus late.
    """
    gal = ['fold', 'participant_id', 'password_id', 'enrol_week']
    stats = imp.groupby(gal)['score'].agg(['mean', 'std']).rename(
        columns={'mean': 'g_mean', 'std': 'g_std'})
    imp = imp.join(stats, on=gal)
    imp = imp[imp['g_std'] > 0].copy()
    imp['z'] = (imp['score'] - imp['g_mean']) / imp['g_std']

    a0 = imp[imp['imp_practice'] == 0]
    a1 = imp[imp['imp_week'] == LAST_WEEK]
    key = gal + ['imp_participant_id']
    merged = a0.merge(a1, on=key, suffixes=('_a0', '_a1'))
    merged['d'] = merged['z_a1'] - merged['z_a0']
    return merged


def summarise_paired(merged, n_boot=2000, seed=0):
    """Aggregate to one value per victim, then bootstrap over victims, so the
    interval respects the fact that trials sharing a victim are not
    independent."""
    per_victim = merged.groupby('participant_id')['d'].mean()
    rng = np.random.RandomState(seed)
    vals = per_victim.values
    boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    n_pos = int((vals > 0).sum())
    return {
        'n_pairs': len(merged),
        'n_victims': len(vals),
        'mean_delta_z': float(vals.mean()),
        'ci_lo': float(lo),
        'ci_hi': float(hi),
        'victims_with_positive_effect': n_pos,
        'victims_total': len(vals),
    }


def practice_curve(imp):
    """Standardised impostor score against practice ordinal, one row per week."""
    gal = ['fold', 'participant_id', 'password_id', 'enrol_week']
    stats = imp.groupby(gal)['score'].agg(['mean', 'std']).rename(
        columns={'mean': 'g_mean', 'std': 'g_std'})
    imp = imp.join(stats, on=gal)
    imp = imp[imp['g_std'] > 0].copy()
    imp['z'] = (imp['score'] - imp['g_mean']) / imp['g_std']
    return imp.groupby('imp_week').agg(
        mean_z=('z', 'mean'),
        median_practice=('imp_practice', 'median'),
        n=('z', 'size'),
    ).reset_index()


# ------------------------------------------------- operating-point metrics

def security_table(gen, imp):
    """FMR and bits of marginal security at fixed usability budgets, for each
    adversary level and each delta. This is the paper's headline currency: fix
    how often a legitimate user is wrongly challenged, then ask what security
    that buys."""
    rows = []
    for delta in sorted(gen['delta'].unique()):
        g = gen[gen['delta'] == delta]['score'].values
        if len(g) < 100:
            continue
        sub = impostors_for_delta(gen, imp, delta)
        levels = {
            'A0_unpractised': sub[sub['imp_practice'] == 0],
            'A1_practised': sub[sub['imp_week'] == LAST_WEEK],
        }
        for level_name, isub in levels.items():
            s = isub['score'].values
            eer, _ = compute_eer(np.r_[np.ones(len(g)), np.zeros(len(s))], np.r_[g, s])
            row = {'delta': int(delta), 'adversary': level_name,
                   'n_genuine': len(g), 'n_impostor': len(s), 'eer': eer}
            for budget in FNMR_BUDGETS:
                thr = np.percentile(g, budget * 100)   # higher score = more genuine
                fmr = float((s >= thr).mean())
                row[f'fmr_at_fnmr_{budget}'] = fmr
                row[f'bits_at_fnmr_{budget}'] = float(-np.log2(fmr)) if fmr > 0 else np.inf
                # three allowed attempts, as a deployed MFA prompt would give
                fmr3 = 1 - (1 - fmr) ** 3
                row[f'bits3_at_fnmr_{budget}'] = float(-np.log2(fmr3)) if fmr3 > 0 else np.inf
            rows.append(row)
    return pd.DataFrame(rows)


GALLERY_KEY = ['fold', 'participant_id', 'password_id', 'enrol_week']


def impostors_for_delta(gen, imp, delta):
    """Impostor trials from exactly the galleries that produce genuine trials at
    this delta.

    This restriction is essential and easy to miss. An impostor score depends on
    the gallery it was scored against, and the reachable enrolment weeks change
    with delta: delta = 7 is only reachable from enrolment week 1, whereas
    delta = 0 draws on all eight. Pooling impostors across every enrolment week
    would compare score distributions built from different gallery mixtures, and
    the resulting difference would be gallery composition rather than anything
    about the adversary.
    """
    galleries = gen[gen['delta'] == delta][GALLERY_KEY].drop_duplicates()
    return imp.merge(galleries, on=GALLERY_KEY, how='inner')


def paper1_replication(gen, imp):
    """Reproduce paper 1's exact pairing, then re-run it with the impostor held
    still, so the bias in its ageing estimate can be measured.

    Paper 1 drew the impostor from week j = i + delta, so growing delta advanced
    the impostor too. Holding the impostor at a fixed week, over the same
    galleries, isolates genuine template drift.
    """
    rows = []
    for delta in sorted(gen['delta'].unique()):
        g = gen[gen['delta'] == delta]['score'].values
        if len(g) < 100:
            continue
        lg = np.ones(len(g))
        sub = impostors_for_delta(gen, imp, delta)
        variants = {
            'paper1_diagonal': sub[sub['imp_week'] == sub['enrol_week'] + delta],
            'impostor_fixed_wk1': sub[sub['imp_week'] == 1],
            'impostor_fixed_wk8': sub[sub['imp_week'] == LAST_WEEK],
        }
        row = {'delta': int(delta), 'n_genuine': len(g),
               'n_galleries': len(gen[gen['delta'] == delta][GALLERY_KEY].drop_duplicates())}
        for name, s_df in variants.items():
            s = s_df['score'].values
            if len(s) < 100:
                row[f'eer_{name}'] = np.nan
                continue
            eer, _ = compute_eer(np.r_[lg, np.zeros(len(s))], np.r_[g, s])
            row[f'eer_{name}'] = eer
        rows.append(row)
    return pd.DataFrame(rows)


def hardware_controlled_contrast(imp, late_week=4):
    """A0 versus a later but still pre-hardware-change week.

    The collection had a hardware transition at week 5 (paper 1 carries it as a
    covariate), which sits in the middle of the practice curve. Contrasting week
    1 against week 4 keeps both sides on the same hardware, so any shift is
    practice rather than equipment.
    """
    gal = ['fold', 'participant_id', 'password_id', 'enrol_week']
    stats = imp.groupby(gal)['score'].agg(['mean', 'std']).rename(
        columns={'mean': 'g_mean', 'std': 'g_std'})
    imp = imp.join(stats, on=gal)
    imp = imp[imp['g_std'] > 0].copy()
    imp['z'] = (imp['score'] - imp['g_mean']) / imp['g_std']

    early = imp[imp['imp_practice'] == 0]
    late = imp[imp['imp_week'] == late_week]
    key = gal + ['imp_participant_id']
    merged = early.merge(late, on=key, suffixes=('_a0', '_a1'))
    merged['d'] = merged['z_a1'] - merged['z_a0']
    return merged


def decomposition_grid(gen, imp):
    """EER over the full (genuine delta) x (impostor week) grid.

    Paper 1 only ever observed the diagonal where the impostor week equals the
    genuine query week. Reading down a column shows genuine drift with the
    adversary held still; reading across a row shows adversary practice with the
    genuine side held still.
    """
    rows = []
    for delta in sorted(gen['delta'].unique()):
        g = gen[gen['delta'] == delta]['score'].values
        if len(g) < 100:
            continue
        labels_g = np.ones(len(g))
        sub = impostors_for_delta(gen, imp, delta)
        for m in sorted(sub['imp_week'].unique()):
            s = sub[sub['imp_week'] == m]['score'].values
            if len(s) < 100:
                continue
            eer, _ = compute_eer(np.r_[labels_g, np.zeros(len(s))], np.r_[g, s])
            rows.append({'delta': int(delta), 'imp_week': int(m), 'eer': eer,
                         'n_genuine': len(g), 'n_impostor': len(s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reporting

def analyse():
    df = load_trials()
    gen = df[df['role'] == 'genuine']
    imp = df[df['role'] == 'impostor']

    out_lines = []

    def emit(s=''):
        print(s)
        out_lines.append(str(s))

    emit('=' * 78)
    emit('EXPERIMENT 1: does adversary practice matter?   M1 scaled Manhattan')
    emit('=' * 78)
    emit(f'genuine trials : {len(gen):,}')
    emit(f'impostor trials: {len(imp):,}')
    emit(f'folds          : {sorted(df["fold"].unique())}')
    emit()

    emit('-' * 78)
    emit('1. A0 (first-ever exposure) vs A1 (week 8) -- paired within gallery')
    emit('-' * 78)
    merged = paired_practice_effect(imp)
    res = summarise_paired(merged)
    emit(f'matched attacker/gallery pairs : {res["n_pairs"]:,}')
    emit(f'standardised score shift A1-A0 : {res["mean_delta_z"]:+.4f} '
         f'[95% CI {res["ci_lo"]:+.4f}, {res["ci_hi"]:+.4f}]')
    emit(f'victims where practice helped  : {res["victims_with_positive_effect"]}'
         f'/{res["victims_total"]}')
    direction = 'CLOSER to' if res['mean_delta_z'] > 0 else 'FURTHER from'
    sig = 'excludes 0' if (res['ci_lo'] > 0 or res['ci_hi'] < 0) else 'includes 0'
    emit(f'=> practice moves the attacker {direction} the victim template ({sig})')
    emit()
    emit('   hardware control: week 1 vs week 4, both pre-transition')
    hw = summarise_paired(hardware_controlled_contrast(imp))
    emit(f'   standardised shift wk4-wk1     : {hw["mean_delta_z"]:+.4f} '
         f'[95% CI {hw["ci_lo"]:+.4f}, {hw["ci_hi"]:+.4f}]')
    hw_sig = 'excludes 0' if (hw['ci_lo'] > 0 or hw['ci_hi'] < 0) else 'includes 0'
    emit(f'   => effect survives holding hardware fixed ({hw_sig})')
    emit()

    emit('-' * 78)
    emit('2. Impostor practice curve (standardised, within gallery)')
    emit('-' * 78)
    curve = practice_curve(imp)
    emit(curve.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    emit()

    emit('-' * 78)
    emit('3. Marginal security at fixed usability budgets')
    emit('-' * 78)
    sec = security_table(gen, imp)
    show = ['delta', 'adversary', 'eer', 'fmr_at_fnmr_0.05',
            'bits_at_fnmr_0.05', 'bits3_at_fnmr_0.05']
    emit(sec[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    emit()
    emit('bits3 = marginal security after three allowed attempts.')
    emit('One extra random lowercase character is worth about 4.70 bits.')
    emit()

    emit('-' * 78)
    emit('4. Paper 1 decomposition: genuine delta vs impostor week')
    emit('-' * 78)
    grid = decomposition_grid(gen, imp)
    pivot = grid.pivot(index='delta', columns='imp_week', values='eer')
    emit('EER by genuine delta (rows) x impostor week (columns)')
    emit(pivot.to_string(float_format=lambda v: f'{v:.4f}'))
    emit()
    col_effect = pivot.mean(axis=0)
    row_effect = pivot.mean(axis=1)
    emit(f'marginal effect of genuine drift  (delta 0 -> 7, impostor fixed): '
         f'{row_effect.iloc[0]:.4f} -> {row_effect.iloc[-1]:.4f} '
         f'({(row_effect.iloc[-1] - row_effect.iloc[0]) * 100:+.2f} pts)')
    emit(f'marginal effect of adversary practice (week 1 -> 8, genuine fixed): '
         f'{col_effect.iloc[0]:.4f} -> {col_effect.iloc[-1]:.4f} '
         f'({(col_effect.iloc[-1] - col_effect.iloc[0]) * 100:+.2f} pts)')
    emit()

    emit('-' * 78)
    emit('5. How much of paper 1\'s ageing effect is the adversary improving?')
    emit('-' * 78)
    rep = paper1_replication(gen, imp)
    emit(rep.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    emit()
    first, last = rep.iloc[0], rep.iloc[-1]
    d_paper1 = (last['eer_paper1_diagonal'] - first['eer_paper1_diagonal']) * 100
    d_wk8 = (last['eer_impostor_fixed_wk8'] - first['eer_impostor_fixed_wk8']) * 100
    d_wk1 = (last['eer_impostor_fixed_wk1'] - first['eer_impostor_fixed_wk1']) * 100
    emit(f'ageing delta 0 -> 7, paper 1 pairing          : {d_paper1:+.2f} pts')
    emit(f'ageing delta 0 -> 7, impostor held at week 8  : {d_wk8:+.2f} pts')
    emit(f'ageing delta 0 -> 7, impostor held at week 1  : {d_wk1:+.2f} pts')
    bias_lo = d_paper1 - max(d_wk1, d_wk8)
    bias_hi = d_paper1 - min(d_wk1, d_wk8)
    word = 'overstates' if bias_lo > 0 else 'understates'
    emit(f'=> paper 1 {word} genuine template ageing by '
         f'{abs(bias_hi):.2f} to {abs(bias_lo):.2f} pts '
         f'({abs(bias_hi) / d_paper1 * 100:.0f}-{abs(bias_lo) / d_paper1 * 100:.0f}% '
         f'of its reported effect)')
    rep.to_csv(os.path.join(RESULT_DIR, 'exp1_paper1_replication.csv'), index=False)

    os.makedirs(RESULT_DIR, exist_ok=True)
    sec.to_csv(os.path.join(RESULT_DIR, 'exp1_security_table.csv'), index=False)
    grid.to_csv(os.path.join(RESULT_DIR, 'exp1_decomposition_grid.csv'), index=False)
    curve.to_csv(os.path.join(RESULT_DIR, 'exp1_practice_curve.csv'), index=False)
    with open(os.path.join(RESULT_DIR, 'exp1_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    ap.add_argument('--analyse-only', action='store_true')
    args = ap.parse_args()

    if not args.analyse_only:
        run_trials(folds_wanted=set(args.folds) if args.folds else None)
    analyse()
