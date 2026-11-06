"""
Experiment 10: test the digraph-coverage mechanism below the category level,
and check the composition result against password-level clustering.

The submitted draft supported the coverage mechanism with correlations computed
across five substitution-category means. Five aggregate points is thin, and the
categories are each represented by only eight specific password strings, so the
effect could be a property of those strings rather than of composition.

This refits the relationship at the victim-password level, with coverage as a
continuous predictor, controlling for length, shift use and repeated characters.
Two questions follow:

  1. Does coverage predict attack success once composition is controlled?
  2. Does composition still predict it once coverage is controlled? If coverage
     fully explains the composition effect, the category terms should collapse.

Standard errors are clustered two ways, on participant and on password, by the
Cameron, Gelbach and Miller estimator: V = V_participant + V_password - V_both.
Clustering on participant alone, as experiment 6 did, treats the 40 password
strings as if they were independent draws, which is the reviewer's concern.
"""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from dataset_cache import load_dataset
from exp2_transfer import canonical_keys

warnings.filterwarnings('ignore')

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FNMR_BUDGET = 0.05
ATTACK = 'A4_keyed_sampled_full'
GALLERY = ['fold', 'participant_id', 'password_id', 'enrol_week']
SHIFTED = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ!"£$%^&*()_+{}:@~<>?|')


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


def password_features():
    ds = load_dataset()
    rows = []
    for pw in ds.password_ids:
        keys = canonical_keys(ds, pw)
        digraphs = list(zip(keys[:-1], keys[1:]))
        rows.append({
            'password_id': pw,
            'n_shifted': sum(1 for k in keys if k in SHIFTED),
            'n_repeated': len(keys) - len(set(keys)),
            'n_unique_digraphs': len(set(digraphs)),
        })
    return pd.DataFrame(rows)


def outcome_table():
    """Attack success per (mechanism, victim, password) at delta = 0."""
    rows = []
    for mech, (trials, forg) in mechanisms().items():
        gen = trials[(trials['role'] == 'genuine') & (trials['delta'] == 0)]
        galleries = gen[GALLERY].drop_duplicates()
        fg = forg.merge(galleries, on=GALLERY, how='inner')
        fg = fg[fg['attack'] == ATTACK]

        for fold in sorted(gen['fold'].unique()):
            g = gen[gen['fold'] == fold]
            by_v = {v: s['score'].values for v, s in g.groupby('participant_id')}
            thr = {}
            for v in by_v:
                others = [s for w, s in by_v.items() if w != v]
                if others:
                    pool = np.concatenate(others)
                    if len(pool) >= 200:
                        thr[v] = float(np.percentile(pool, FNMR_BUDGET * 100))
            f = fg[fg['fold'] == fold].copy()
            if not len(f):
                continue
            f['hit'] = [s >= thr.get(p, np.inf)
                        for s, p in zip(f['score'], f['participant_id'])]
            agg = (f.groupby(['participant_id', 'password_id'])['hit']
                   .agg(n_success='sum', n_trials='size').reset_index())
            agg['mechanism'] = mech
            agg['fold'] = fold
            rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def two_way_cluster(model, df, formula, g1, g2):
    """Cameron-Gelbach-Miller: V = V(g1) + V(g2) - V(g1 x g2)."""
    both = df[g1].astype(str) + '|' + df[g2].astype(str)
    fits = {}
    for name, groups in (('a', df[g1]), ('b', df[g2]), ('ab', both)):
        fits[name] = smf.glm(formula, data=df, family=sm.families.Binomial(),
                             freq_weights=df['n_trials']).fit(
            cov_type='cluster', cov_kwds={'groups': groups})
    V = fits['a'].cov_params() + fits['b'].cov_params() - fits['ab'].cov_params()
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    params = fits['a'].params
    z = params / se
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return pd.DataFrame({'coef': params, 'se_2way': se, 'z': z, 'p': p},
                        index=params.index), fits['a']


def main():
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    cov = pd.read_csv(os.path.join(RESULT_DIR, 'exp5b_coverage.csv'))
    feats = password_features()
    out = outcome_table()

    df = (out.merge(cov, on=['participant_id', 'password_id'], how='inner')
             .merge(feats, on='password_id', how='left'))
    df = df[df['n_trials'] > 0].copy()
    df['rate'] = df['n_success'] / df['n_trials']
    for c in ('digraph_coverage', 'char_coverage', 'length', 'n_shifted', 'n_repeated'):
        df[c + '_c'] = df[c] - df[c].mean()

    emit('=' * 84)
    emit('EXPERIMENT 10: coverage mechanism at victim-password level')
    emit('=' * 84)
    emit(f'{len(df):,} victim-password-mechanism units, '
         f'{df["participant_id"].nunique()} participants, '
         f'{df["password_id"].nunique()} passwords')
    emit(f'outcome: {ATTACK} success at a {FNMR_BUDGET:.0%} budget, delta = 0')
    emit('standard errors clustered two ways (participant and password)')
    emit()

    models = {
        'composition only': 'rate ~ C(substitution_category) + length_c + C(mechanism)',
        'coverage only': 'rate ~ digraph_coverage_c + length_c + C(mechanism)',
        'both': ('rate ~ digraph_coverage_c + C(substitution_category) + length_c '
                 '+ n_shifted_c + n_repeated_c + C(mechanism)'),
    }
    for name, formula in models.items():
        emit('-' * 84)
        emit(name.upper())
        emit('-' * 84)
        tab, fit = two_way_cluster(None, df, formula, 'participant_id', 'password_id')
        keep = [i for i in tab.index if 'mechanism' not in i]
        emit(tab.loc[keep].to_string(float_format=lambda v: f'{v:.4f}'))
        emit(f'  pseudo R2 (deviance): {1 - fit.deviance / fit.null_deviance:.4f}')
        emit()

    emit('-' * 84)
    emit('Reading')
    emit('-' * 84)
    tab_both, _ = two_way_cluster(None, df, models['both'], 'participant_id', 'password_id')
    cov_p = tab_both.loc['digraph_coverage_c', 'p']
    sub_terms = [i for i in tab_both.index if 'substitution' in i]
    sub_sig = [i for i in sub_terms if tab_both.loc[i, 'p'] < 0.05]
    emit(f'coverage, controlling for composition: coef '
         f'{tab_both.loc["digraph_coverage_c", "coef"]:+.3f}, p = {cov_p:.4g}')
    emit(f'composition terms still significant with coverage in the model: '
         f'{len(sub_sig)} of {len(sub_terms)}')
    if cov_p < 0.05 and not sub_sig:
        emit('=> coverage accounts for the composition effect')
    elif cov_p < 0.05 and sub_sig:
        emit('=> coverage predicts attack success, but composition retains an '
             'effect beyond it, so coverage is part of the story and not all of it')
    else:
        emit('=> coverage does not survive; the category-level correlation was '
             'not evidence of this mechanism')

    df.to_csv(os.path.join(RESULT_DIR, 'exp10_units.csv'), index=False)
    tab_both.to_csv(os.path.join(RESULT_DIR, 'exp10_model.csv'))
    with open(os.path.join(RESULT_DIR, 'exp10_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
