"""
Experiment 9: confidence intervals, and the 1 and 2 per cent usability budgets.

Two gaps in the submitted draft. Attack success and marginal security were given
as bare point estimates, and only the 5 per cent budget was reported although
three were computed.

Intervals come from a participant-level bootstrap, because trials sharing a
victim are not independent. Victims are resampled with replacement; the
threshold is recomputed from the resampled genuine scores at each draw, then the
false match rate is measured on the resampled attack scores. Resampling at the
trial level would ignore the clustering and give intervals that are far too
narrow.

The interval is computed on the false match rate and only then transformed to
bits. Transforming an interval through -log2 is valid because the map is
monotonic; averaging bits across bootstrap draws would not be, since -log2 is
convex.
"""
import os
import re

import numpy as np
import pandas as pd

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
BUDGETS = [0.01, 0.02, 0.05]
N_BOOT = 1000
SEED = 0
GALLERY = ['fold', 'participant_id', 'password_id', 'enrol_week']

LEVELS = {
    'A0 unpractised human': ('imp', lambda d: d['imp_practice'] == 0),
    'A1 practised human': ('imp', lambda d: d['imp_week'] == 8),
    'A2 population statistics': ('forg', 'A2_sampled'),
    'A3 victim speed': ('forg', 'A4_global_sampled_full'),
    'A4 victim timings, 2-week leak': ('forg', 'A4_keyed_sampled_early'),
    'A4 victim timings, full leak': ('forg', 'A4_keyed_sampled_full'),
}


def read_dir(path):
    parts = sorted(f for f in os.listdir(path) if f.endswith('.parquet'))
    return pd.concat([pd.read_parquet(os.path.join(path, p)) for p in parts],
                     ignore_index=True)


def mechanisms():
    out = {'M1_Manhattan': (read_dir(os.path.join(RESULT_DIR, 'exp1_trials')),
                            pd.read_parquet(os.path.join(RESULT_DIR,
                                                         'exp2_forgery_scores.parquet')))}
    for name in sorted(os.listdir(RESULT_DIR)):
        m = re.fullmatch(r'exp3_trials_(.+)', name)
        if m:
            mech = m.group(1)
            out[mech] = (read_dir(os.path.join(RESULT_DIR, name)),
                         read_dir(os.path.join(RESULT_DIR, f'exp3_forgeries_{mech}')))
    return out


def bits(fmr):
    return float(-np.log2(fmr)) if fmr > 0 else np.inf


def evaluate(trials, forg, delta, budget, rng):
    gen = trials[(trials['role'] == 'genuine') & (trials['delta'] == delta)]
    if not len(gen):
        return []
    galleries = gen[GALLERY].drop_duplicates()
    imp = trials[trials['role'] == 'impostor'].merge(galleries, on=GALLERY, how='inner')
    fg = forg.merge(galleries, on=GALLERY, how='inner')

    gen_by_v = {v: s['score'].values for v, s in gen.groupby('participant_id')}
    victims = np.array(sorted(gen_by_v))

    rows = []
    for label, (src, sel) in LEVELS.items():
        if src == 'imp':
            sub = imp[sel(imp)]
        else:
            sub = fg[fg['attack'] == sel]
        if len(sub) < 200:
            continue
        atk_by_v = {v: s['score'].values for v, s in sub.groupby('participant_id')}
        common = np.array([v for v in victims if v in atk_by_v])
        if len(common) < 10:
            continue

        thr = np.percentile(np.concatenate([gen_by_v[v] for v in common]), budget * 100)
        point = float(np.mean(np.concatenate([atk_by_v[v] for v in common]) >= thr))

        boot = []
        for _ in range(N_BOOT):
            pick = rng.choice(common, len(common), replace=True)
            t = np.percentile(np.concatenate([gen_by_v[v] for v in pick]), budget * 100)
            a = np.concatenate([atk_by_v[v] for v in pick])
            boot.append(float(np.mean(a >= t)))
        lo, hi = np.percentile(boot, [2.5, 97.5])

        rows.append({
            'delta': delta, 'budget': budget, 'adversary': label,
            'fmr': point, 'fmr_lo': lo, 'fmr_hi': hi,
            'bits': bits(point), 'bits_lo': bits(hi), 'bits_hi': bits(lo),
            'n_victims': len(common),
        })
    return rows


def main():
    rng = np.random.RandomState(SEED)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 86)
    emit('EXPERIMENT 9: bootstrap intervals and alternative usability budgets')
    emit('=' * 86)
    emit(f'{N_BOOT} participant-level bootstrap draws, 95 per cent percentile intervals')
    emit('bits intervals are the transformed FMR interval, not an average of bits')
    emit()

    all_rows = []
    for mech, (trials, forg) in mechanisms().items():
        for delta in (0, 7):
            for b in BUDGETS:
                for r in evaluate(trials, forg, delta, b, rng):
                    r['mechanism'] = mech
                    all_rows.append(r)
        print('  done', mech, flush=True)
    tab = pd.DataFrame(all_rows)
    tab.to_csv(os.path.join(RESULT_DIR, 'exp9_intervals.csv'), index=False)

    emit('-' * 86)
    emit('Marginal security in bits with 95% CI, fresh template (delta = 0)')
    emit('-' * 86)
    for b in BUDGETS:
        emit(f'usability budget = {b:.0%} FNMR')
        sub = tab[(tab['delta'] == 0) & (tab['budget'] == b)]
        piv = sub.pivot(index='adversary', columns='mechanism', values='bits')
        lo = sub.pivot(index='adversary', columns='mechanism', values='bits_lo')
        hi = sub.pivot(index='adversary', columns='mechanism', values='bits_hi')
        for adv in piv.index:
            cells = ['%-14s' % adv[:14]]
            for m in piv.columns:
                cells.append('%5.2f [%4.2f,%5.2f]' % (piv.loc[adv, m], lo.loc[adv, m], hi.loc[adv, m]))
            emit('  ' + '  '.join(cells))
        emit('    columns: ' + ', '.join(piv.columns))
        emit()

    emit('-' * 86)
    emit('Is the conclusion stable across budgets?')
    emit('-' * 86)
    key = tab[(tab['delta'] == 0) & (tab['adversary'] == 'A4 victim timings, full leak')]
    for b in BUDGETS:
        s = key[key['budget'] == b]
        emit(f'  budget {b:.0%}: leaked-profile security ranges '
             f'{s["bits"].min():.3f} to {s["bits"].max():.3f} bits '
             f'across the four mechanisms')
    emit()
    emit('One extra random lowercase character is worth 4.70 bits at every budget.')

    with open(os.path.join(RESULT_DIR, 'exp9_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
