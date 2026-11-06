"""
Experiment 16 (R1 revision): robustness checks requested by reviewer 1.

Part A, marginal adversary effects (reviewer 1, comment 5).
  The odds ratios R0 reported for A2, A3 and A4 (0.99, 0.54, 2.37) are the
  main-effect terms of a model with adversary x mechanism and adversary x
  substitution interactions, so each applies only at the reference cell: M1,
  combination passwords, mean gap. A3's 0.54 is not a statement about A3 in
  general, which is why Table 2 appears to contradict it. Two marginal
  summaries are reported: a main-effects model with no adversary interactions,
  and model-standardised success (every cell's prediction under each adversary
  level, averaged over the observed cells).

Part B, dependence (reviewer 1, comment 4).
  1. The main-effects and interaction models refitted with standard errors
     clustered two ways, on participant and on password (Cameron, Gelbach and
     Miller).
  2. The composition effect tested at the level of the 40 password strings, by
     permuting the substitution labels across passwords (eight per category).
     Any dependence among victims, impostors, forgeries or gaps sits inside a
     password's summary, so the test relies only on exchangeability of the 40
     strings.
  3. A linear mixed model on the empirical logit of victim-password success,
     with crossed random intercepts for participant and password.

Part C, threshold calibration (reviewer 1, comment 3).
  1. Split-half: within each fold the test participants are divided in two;
     thresholds are estimated on one half and applied to the other, so no
     score from an evaluated participant enters their own threshold.
  2. Fixed at enrolment: the split-half threshold estimated at delta = 0 is
     held for every later gap, as a deployment that calibrates once would.
     The cost of template ageing then appears as false non-matches.
"""
import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from exp10_coverage_model import two_way_cluster

warnings.filterwarnings('ignore')

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FNMR_BUDGET = 0.05
REFERENCE = 'A1_human_practised'
GALLERY = ['fold', 'participant_id', 'password_id', 'enrol_week']
N_PERM = 10000
ATTACKS = {'A2_sampled': 'A2 population', 'A4_keyed_sampled_early': 'A4 2-week leak',
           'A4_keyed_sampled_full': 'A4 full leak'}


def emit_factory(lines):
    def emit(s=''):
        print(s)
        lines.append(str(s))
    return emit


def prep(cells):
    df = cells[cells['n_trials'] > 0].copy()
    df['adversary'] = pd.Categorical(
        df['adversary'], categories=[REFERENCE] + sorted(set(df['adversary']) - {REFERENCE}))
    df['delta_c'] = df['delta'] - df['delta'].mean()
    df['length_c'] = df['length'] - df['length'].mean()
    df['success_rate'] = df['n_success'] / df['n_trials']
    return df


# ===================================================================== part A

MAIN = ('success_rate ~ adversary + delta_c + C(mechanism) + C(substitution) '
        '+ length_c + C(fold)')
FULL = ('success_rate ~ adversary * delta_c + adversary * C(mechanism) '
        '+ adversary * C(substitution) + length_c + C(fold)')


def or_table(fit, se=None):
    keep = [p for p in fit.params.index if p.startswith('adversary[') and ':' not in p]
    b = fit.params[keep]
    s = fit.bse[keep] if se is None else se[keep]
    return pd.DataFrame({'odds_ratio': np.exp(b), 'ci_lo': np.exp(b - 1.96 * s),
                         'ci_hi': np.exp(b + 1.96 * s),
                         'p': 2 * (1 - stats.norm.cdf(np.abs(b / s)))})


def standardised(fit, df):
    """Average predicted success over every observed cell, with each cell's
    adversary set to each level in turn."""
    out = {}
    w = df['n_trials'].values
    for lvl in df['adversary'].cat.categories:
        d = df.copy()
        d['adversary'] = pd.Categorical([lvl] * len(d), categories=df['adversary'].cat.categories)
        out[lvl] = float(np.average(fit.predict(d), weights=w))
    return pd.Series(out)


def part_a(df, label, emit):
    emit('-' * 84)
    emit(f'A. MARGINAL ADVERSARY EFFECTS ({label})')
    emit('-' * 84)
    main = smf.glm(MAIN, data=df, family=sm.families.Binomial(),
                   freq_weights=df['n_trials']).fit(
        cov_type='cluster', cov_kwds={'groups': df['participant_id']})
    emit('Main-effects model (no adversary interactions), participant-clustered:')
    emit(or_table(main).to_string(float_format=lambda v: f'{v:.4f}'))
    full = smf.glm(FULL, data=df, family=sm.families.Binomial(),
                   freq_weights=df['n_trials']).fit(
        cov_type='cluster', cov_kwds={'groups': df['participant_id']})
    emit()
    emit('Model-standardised success from the full interaction model '
         '(averaged over all observed cells):')
    st = standardised(full, df)
    emit(st.to_string(float_format=lambda v: f'{v:.4f}'))
    emit()
    emit('Why the reference-cell odds ratio for A3 is below one: '
         'predicted success on M1, delta = 0, by category')
    grid = []
    for sub in sorted(df['substitution'].unique()):
        row = {'substitution': sub}
        for lvl in df['adversary'].cat.categories:
            d = pd.DataFrame({'adversary': pd.Categorical([lvl], categories=df['adversary'].cat.categories),
                              'delta_c': [0 - df['delta'].mean()], 'mechanism': ['M1_Manhattan'],
                              'substitution': [sub], 'length_c': [0.0], 'fold': [0]})
            row[lvl] = float(full.predict(d)[0])
        grid.append(row)
    emit(pd.DataFrame(grid).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    emit()
    raw = (df.groupby(['mechanism', 'adversary'], observed=True)[['n_success', 'n_trials']].sum())
    raw = (raw['n_success'] / raw['n_trials']).unstack('mechanism')
    emit('Raw pooled success over all gaps (for comparison with Table 2, which is delta = 0):')
    emit(raw.to_string(float_format=lambda v: f'{v:.3f}'))
    return main, full


# ===================================================================== part B

def part_b(df, main, label, emit):
    emit('-' * 84)
    emit(f'B. DEPENDENCE ({label})')
    emit('-' * 84)

    for name, formula in (('main effects', MAIN), ('full interaction', FULL)):
        tab, _ = two_way_cluster(None, df, formula, 'participant_id', 'password_id')
        keep = [i for i in tab.index if i.startswith('adversary[')
                and (':' not in i or 'substitution' in i or 'delta' in i)]
        keep += ['delta_c', 'length_c']
        t = tab.loc[keep].copy()
        t['odds_ratio'] = np.exp(t['coef'])
        emit(f'{name}, two-way clustered (participant x password):')
        emit(t[['odds_ratio', 'se_2way', 'z', 'p']].to_string(float_format=lambda v: f'{v:.4f}'))
        emit()

    # ---- password-level permutation test of the composition effect
    d0 = df[df['delta'] == 0]
    adv_a4 = 'A4_synth_keyed'
    pw = (d0[d0['adversary'].isin([REFERENCE, adv_a4])]
          .groupby(['password_id', 'substitution', 'adversary'], observed=True)
          [['n_success', 'n_trials']].sum().reset_index())
    pw['logit'] = np.log((pw['n_success'] + 0.5) / (pw['n_trials'] - pw['n_success'] + 0.5))
    wide = pw.pivot_table(index=['password_id', 'substitution'], columns='adversary',
                          values='logit', observed=True).reset_index()
    wide['effect'] = wide[adv_a4] - wide[REFERENCE]
    eff = wide['effect'].values
    lab = wide['substitution'].values
    is_comb = lab == 'combination'

    def stat(mask):
        return eff[~mask].mean() - eff[mask].mean()

    obs = stat(is_comb)
    rng = np.random.RandomState(0)
    null = np.array([stat(rng.permutation(is_comb)) for _ in range(N_PERM)])
    p_perm = (1 + np.sum(null >= obs)) / (1 + N_PERM)
    # omnibus: variance between category means of the per-password effect
    def f_stat(labels):
        groups = [eff[labels == c] for c in np.unique(labels)]
        return stats.f_oneway(*groups).statistic
    f_obs = f_stat(lab)
    f_null = np.array([f_stat(rng.permutation(lab)) for _ in range(N_PERM)])
    p_f = (1 + np.sum(f_null >= f_obs)) / (1 + N_PERM)
    emit('Password-level permutation test, delta = 0, all mechanisms pooled.')
    emit('Per password: logit(A4 success) - logit(A1 success). '
         f'{len(wide)} passwords, {N_PERM:,} permutations of the category labels.')
    emit(wide.groupby('substitution')['effect'].agg(['mean', 'std', 'size'])
         .to_string(float_format=lambda v: f'{v:.3f}'))
    emit(f'  other categories minus combination: {obs:+.3f} logits, '
         f'one-sided permutation p = {p_perm:.4f}')
    emit(f'  any difference between categories (F = {f_obs:.2f}): permutation p = {p_f:.4f}')
    emit()

    # ---- crossed random effects
    vp = (d0[d0['adversary'].isin([REFERENCE, adv_a4])]
          .groupby(['mechanism', 'participant_id', 'password_id', 'substitution',
                    'length_c', 'adversary'], observed=True)
          [['n_success', 'n_trials']].sum().reset_index())
    vp['elogit'] = np.log((vp['n_success'] + 0.5) / (vp['n_trials'] - vp['n_success'] + 0.5))
    vp['a4'] = (vp['adversary'] == adv_a4).astype(int)
    vp['one'] = 1
    vp['substitution'] = pd.Categorical(
        vp['substitution'], categories=['combination', 'none', 'numeric', 'symbol', 'uppercase'])
    lmm = smf.mixedlm('elogit ~ a4 * C(substitution) + length_c + C(mechanism)', vp,
                      groups=vp['one'],
                      vc_formula={'participant': '0 + C(participant_id)',
                                  'password': '0 + C(password_id)'}).fit(reml=True)
    keep = [i for i in lmm.params.index if i.startswith('a4')]
    emit('Linear mixed model on the empirical logit, delta = 0, A1 and A4 units,')
    emit(f'crossed random intercepts for participant and password ({len(vp):,} units):')
    emit(pd.DataFrame({'coef': lmm.params[keep], 'se': lmm.bse[keep],
                       'p': lmm.pvalues[keep]}).to_string(float_format=lambda v: f'{v:.4f}'))
    emit(f'  variance components: {dict(zip(lmm.model.exog_vc.names, np.round(lmm.vcomp, 4)))}')
    emit()


# ===================================================================== part C

def read_dir(path):
    parts = sorted(f for f in os.listdir(path) if f.endswith('.parquet'))
    return pd.concat([pd.read_parquet(os.path.join(path, p)) for p in parts], ignore_index=True)


def mechanisms():
    out = {'M1_Manhattan': (read_dir(os.path.join(RESULT_DIR, 'exp1_trials')),
                            pd.read_parquet(os.path.join(RESULT_DIR, 'exp2_forgery_scores.parquet')))}
    for name in sorted(os.listdir(RESULT_DIR)):
        m = re.fullmatch(r'exp3_trials_(.+)', name)
        if m:
            mech = m.group(1)
            out[mech] = (read_dir(os.path.join(RESULT_DIR, name)),
                         read_dir(os.path.join(RESULT_DIR, f'exp3_forgeries_{mech}')))
    return out


def lopo_thresholds(gen_fd):
    by_v = {v: s['score'].values for v, s in gen_fd.groupby('participant_id')}
    out = {}
    for v in by_v:
        others = [s for w, s in by_v.items() if w != v]
        if others:
            pool = np.concatenate(others)
            if len(pool) >= 200:
                out[v] = float(np.percentile(pool, FNMR_BUDGET * 100))
    return out


def split_thresholds(gen_fd, halves):
    out = {}
    for h in (0, 1):
        cal = gen_fd[gen_fd['participant_id'].isin(halves[1 - h])]['score'].values
        if len(cal) >= 200:
            t = float(np.percentile(cal, FNMR_BUDGET * 100))
            for v in halves[h]:
                out[v] = t
    return out


def rates(gen_d, att, thr):
    """Pooled FNMR over genuine trials and pooled FMR per attack, given
    per-victim thresholds."""
    t_g = gen_d['participant_id'].map(thr)
    ok = t_g.notna()
    fnmr = float((gen_d.loc[ok, 'score'] < t_g[ok]).mean())
    out = {'FNMR': fnmr}
    for name, sub in att.items():
        t = sub['participant_id'].map(thr)
        ok = t.notna()
        out[name] = float((sub.loc[ok, 'score'] >= t[ok]).mean())
    return out


def part_c(emit):
    emit('-' * 84)
    emit('C. THRESHOLD CALIBRATION')
    emit('-' * 84)
    rows = []
    for mech, (trials, forg) in mechanisms().items():
        gen = trials[trials['role'] == 'genuine']
        imp = trials[trials['role'] == 'impostor']
        for fold in sorted(gen['fold'].unique()):
            g_f = gen[gen['fold'] == fold]
            vics = np.array(sorted(g_f['participant_id'].unique()))
            perm = np.random.RandomState(1000 + fold).permutation(vics)
            halves = (set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:]))
            fixed = split_thresholds(g_f[g_f['delta'] == 0], halves)
            for delta in sorted(g_f['delta'].unique()):
                g_d = g_f[g_f['delta'] == delta]
                gal = g_d[GALLERY].drop_duplicates()
                i_d = imp[imp['fold'] == fold].merge(gal, on=GALLERY)
                f_d = forg[forg['fold'] == fold].merge(gal, on=GALLERY)
                att = {'A0 unpractised': i_d[i_d['imp_practice'] == 0],
                       'A1 practised': i_d[i_d['imp_week'] == 8]}
                for raw, name in ATTACKS.items():
                    att[name] = f_d[f_d['attack'] == raw]
                for scheme, thr in (('LOPO per gap (R0)', lopo_thresholds(g_d)),
                                    ('split-half per gap', split_thresholds(g_d, halves)),
                                    ('split-half fixed at delta=0', fixed)):
                    r = rates(g_d, att, thr)
                    # counts so folds can be pooled properly
                    n = {'FNMR': len(g_d), **{k: len(v) for k, v in att.items()}}
                    for k, v in r.items():
                        rows.append({'mechanism': mech, 'fold': fold, 'delta': delta,
                                     'scheme': scheme, 'quantity': k, 'rate': v, 'n': n[k]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, 'exp16_thresholds.csv'), index=False)
    pooled = (df.assign(w=df['rate'] * df['n'])
                .groupby(['scheme', 'mechanism', 'delta', 'quantity'])[['w', 'n']].sum())
    pooled = (pooled['w'] / pooled['n']).rename('rate').reset_index()

    for delta in (0, 7):
        emit(f'delta = {delta}: FNMR achieved and attack success (FMR), by calibration scheme')
        sub = pooled[pooled['delta'] == delta]
        emit(sub.pivot_table(index=['quantity', 'scheme'], columns='mechanism', values='rate')
             .to_string(float_format=lambda v: f'{v:.3f}'))
        emit()
    emit('Marginal security in bits for A4 2-week leak, delta = 0:')
    s = pooled[(pooled['delta'] == 0) & (pooled['quantity'] == 'A4 2-week leak')].copy()
    s['bits'] = -np.log2(s['rate'])
    emit(s.pivot_table(index='scheme', columns='mechanism', values='bits')
         .to_string(float_format=lambda v: f'{v:.3f}'))
    emit()
    emit('Fixed-at-enrolment threshold: FNMR and A1 / A4 2-week success by gap')
    s = pooled[(pooled['scheme'] == 'split-half fixed at delta=0') &
               (pooled['quantity'].isin(['FNMR', 'A1 practised', 'A4 2-week leak']))]
    emit(s.pivot_table(index=['quantity', 'delta'], columns='mechanism', values='rate')
         .to_string(float_format=lambda v: f'{v:.3f}'))
    return pooled


def main():
    lines = []
    emit = emit_factory(lines)
    emit('=' * 84)
    emit('EXPERIMENT 16: robustness checks for reviewer 1 (comments 3, 4, 5)')
    emit('=' * 84)
    for label, path in (('full leak, as R0', 'exp6_cells.parquet'),
                        ('two-week leak', 'exp15_cells_early.parquet')):
        df = prep(pd.read_parquet(os.path.join(RESULT_DIR, path)))
        main_fit, _ = part_a(df, label, emit)
        part_b(df, main_fit, label, emit)
    part_c(emit)
    with open(os.path.join(RESULT_DIR, 'exp16_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f'\nwritten to {RESULT_DIR}')


if __name__ == '__main__':
    main()
