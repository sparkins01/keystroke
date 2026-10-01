"""
Experiment 8: recompute multi-attempt attack success empirically.

Experiment 6 reported cumulative success as 1 - (1 - p)^n, applying the formula
to the pooled single-attempt false match rate. That assumes every attempt is an
independent Bernoulli draw with the same probability. It is not. Victims differ:
some are vulnerable to almost any forgery, others resist nearly all of them.

Because 1 - (1 - p)^n is concave in p, Jensen's inequality says the pooled
calculation OVERSTATES cumulative success whenever victims are heterogeneous.
The size of the error depends on the spread, which has to be measured rather
than assumed.

This script computes the same quantity from the actual attempts:

  synthetic attacks   five independent forgeries exist per gallery, so the
                      probability that at least one of three drawn without
                      replacement succeeds is exact:
                      P = 1 - C(5-k, 3) / C(5, 3), for k successes out of five

  human attacks       an attacker is one impostor. Their three attempts are
                      their samples from weeks 6, 7 and 8, which keeps all
                      three at a practised level.

Per-gallery (or per-gallery-per-attacker) probabilities are then averaged, which
is the unbiased estimator. The pooled figure is reported alongside so the size
of the bias is visible.
"""
import os
import re
from math import comb

import numpy as np
import pandas as pd

from metrics import compute_eer  # noqa: F401  (kept for parity with other scripts)

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FNMR_BUDGET = 0.05
N_RETRY = 3
GALLERY = ['fold', 'participant_id', 'password_id', 'enrol_week']
HUMAN_RETRY_WEEKS = (6, 7, 8)
SYNTH = 'A4_keyed_sampled_full'


def read_dir(path):
    parts = sorted(f for f in os.listdir(path) if f.endswith('.parquet'))
    return pd.concat([pd.read_parquet(os.path.join(path, p)) for p in parts],
                     ignore_index=True)


def mechanisms():
    out = {}
    trials = read_dir(os.path.join(RESULT_DIR, 'exp1_trials'))
    forg = pd.read_parquet(os.path.join(RESULT_DIR, 'exp2_forgery_scores.parquet'))
    out['M1_Manhattan'] = (trials, forg)
    for name in sorted(os.listdir(RESULT_DIR)):
        m = re.fullmatch(r'exp3_trials_(.+)', name)
        if not m:
            continue
        mech = m.group(1)
        out[mech] = (read_dir(os.path.join(RESULT_DIR, name)),
                     read_dir(os.path.join(RESULT_DIR, f'exp3_forgeries_{mech}')))
    return out


def thresholds(gen, fold, delta):
    """Leave-one-participant-out 5% FNMR thresholds for one fold and gap."""
    g = gen[(gen['fold'] == fold) & (gen['delta'] == delta)]
    by_v = {v: sub['score'].values for v, sub in g.groupby('participant_id')}
    out = {}
    for v in by_v:
        others = [s for w, s in by_v.items() if w != v]
        if not others:
            continue
        pool = np.concatenate(others)
        if len(pool) >= 200:
            out[v] = float(np.percentile(pool, FNMR_BUDGET * 100))
    return out


def exact_best_of_n(k, total, n):
    """P(at least one success) when n of `total` attempts are drawn without
    replacement and k of them succeed."""
    if k <= 0:
        return 0.0
    if total - k < n:
        return 1.0
    return 1.0 - comb(total - k, n) / comb(total, n)


def evaluate(mech, trials, forg, delta):
    gen = trials[trials['role'] == 'genuine']
    imp = trials[trials['role'] == 'impostor']
    galleries = gen[gen['delta'] == delta][GALLERY].drop_duplicates()
    if not len(galleries):
        return None

    imp_d = imp.merge(galleries, on=GALLERY, how='inner')
    forg_d = forg.merge(galleries, on=GALLERY, how='inner')
    forg_d = forg_d[forg_d['attack'] == SYNTH]

    rows = []
    for fold in sorted(galleries['fold'].unique()):
        thr = thresholds(gen, fold, delta)
        if not thr:
            continue

        # ---- synthetic: five forgeries per gallery, exact best-of-three
        f = forg_d[forg_d['fold'] == fold].copy()
        if len(f):
            f['hit'] = [s >= thr.get(p, np.inf)
                        for s, p in zip(f['score'], f['participant_id'])]
            agg = f.groupby(GALLERY)['hit'].agg(['sum', 'size']).reset_index()
            agg = agg[agg['size'] >= N_RETRY]
            for _, r in agg.iterrows():
                rows.append({'adversary': 'A4 synthetic, keyed',
                             'participant_id': r['participant_id'],
                             'single': r['sum'] / r['size'],
                             'multi': exact_best_of_n(int(r['sum']), int(r['size']), N_RETRY)})

        # ---- human: one attacker is one impostor, attempts at weeks 6, 7, 8
        h = imp_d[(imp_d['fold'] == fold) & (imp_d['imp_week'].isin(HUMAN_RETRY_WEEKS))].copy()
        if len(h):
            h['hit'] = [s >= thr.get(p, np.inf)
                        for s, p in zip(h['score'], h['participant_id'])]
            key = GALLERY + ['imp_participant_id']
            agg = h.groupby(key)['hit'].agg(['sum', 'size']).reset_index()
            agg = agg[agg['size'] == len(HUMAN_RETRY_WEEKS)]
            for _, r in agg.iterrows():
                rows.append({'adversary': 'A1 practised human',
                             'participant_id': r['participant_id'],
                             'single': r['sum'] / r['size'],
                             'multi': 1.0 if r['sum'] > 0 else 0.0})

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['mechanism'] = mech
    df['delta'] = delta
    return df


def main():
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 80)
    emit('EXPERIMENT 8: multi-attempt success, computed from the attempts')
    emit('=' * 80)
    emit(f'usability budget {FNMR_BUDGET:.0%}, {N_RETRY} permitted attempts')
    emit('pooled  = 1-(1-p)^n applied to the pooled rate, as in experiment 6')
    emit('empirical = per-gallery probability, then averaged')
    emit()

    all_rows = []
    for mech, (trials, forg) in mechanisms().items():
        for delta in (0, 7):
            d = evaluate(mech, trials, forg, delta)
            if d is not None:
                all_rows.append(d)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_parquet(os.path.join(RESULT_DIR, 'exp8_retry_units.parquet'), index=False)

    out = []
    for (mech, delta, adv), g in df.groupby(['mechanism', 'delta', 'adversary']):
        p_bar = g['single'].mean()
        pooled = 1 - (1 - p_bar) ** N_RETRY
        empirical = g['multi'].mean()
        out.append({'mechanism': mech, 'delta': delta, 'adversary': adv,
                    'n_units': len(g), 'single_attempt': p_bar,
                    'pooled_formula': pooled, 'empirical': empirical,
                    'overstatement': pooled - empirical})
    tab = pd.DataFrame(out).sort_values(['adversary', 'delta', 'mechanism'])
    tab.to_csv(os.path.join(RESULT_DIR, 'exp8_retry.csv'), index=False)

    for adv in sorted(tab['adversary'].unique()):
        for delta in (0, 7):
            sub = tab[(tab['adversary'] == adv) & (tab['delta'] == delta)]
            if not len(sub):
                continue
            emit('-' * 80)
            emit(f'{adv},  delta = {delta}')
            emit('-' * 80)
            emit(sub[['mechanism', 'n_units', 'single_attempt', 'pooled_formula',
                      'empirical', 'overstatement']]
                 .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
            emit()

    emit('-' * 80)
    emit('Heterogeneity across victims (single-attempt success, A4 keyed, delta=0)')
    emit('-' * 80)
    sub = df[(df['adversary'] == 'A4 synthetic, keyed') & (df['delta'] == 0)]
    for mech, g in sub.groupby('mechanism'):
        per_victim = g.groupby('participant_id')['single'].mean()
        emit(f'{mech}:  median {per_victim.median():.2f}  '
             f'IQR {per_victim.quantile(.25):.2f}-{per_victim.quantile(.75):.2f}  '
             f'always vulnerable (>0.95): {int((per_victim > 0.95).sum())}/{len(per_victim)}  '
             f'resistant (<0.5): {int((per_victim < 0.5).sum())}/{len(per_victim)}')
    emit()
    worst = tab['overstatement'].max()
    emit(f'largest overstatement by the pooled formula: {worst:.3f}')

    with open(os.path.join(RESULT_DIR, 'exp8_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
