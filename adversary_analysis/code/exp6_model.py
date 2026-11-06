"""
Experiment 6: the inferential model, and the retry-policy analysis.

Everything so far has been descriptive. This fits the model promised in
`design.md` S8 and answers RQ1 to RQ4 with proper standard errors.

Two decisions carried over from paper 1, both learned the hard way there:

  1. **Logistic is primary this time.** Paper 1 used a linear probability model
     as primary and found, late, that a logistic specification overturned two of
     its three mechanism-specific conclusions. Here the logistic fit leads and
     the linear probability model is the sensitivity check.

  2. **Clustered robust standard errors, not a variational GLMM.** Paper 1's
     `BinomialBayesMixedGLM` did not converge, and a fixed-effects logistic
     regression with participant-clustered robust errors agreed with it to three
     decimals anyway. Clustering on participant is what the non-independence
     actually requires, since trials sharing a victim are correlated.

Threshold calibration is leave-one-participant-out: the operating point applied
to a victim's trials is estimated from the genuine scores of the *other* test
participants in the same fold, at the same gap. Scores are not comparable across
folds for the embedding mechanisms, since each fold trains its own encoders, so
calibration never crosses a fold boundary.

Outcome is attack success: did this impostor or forgery score at or above the
threshold that holds the legitimate user's false-reject rate at the usability
budget.
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from password_metadata import build_password_metadata
from exp3_analyse import discover, GALLERY_KEY, FNMR_BUDGET

warnings.filterwarnings('ignore')

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
N_RETRY = 3

# the matched-noise ladder: every rung is scored the same way and differs only
# in what the adversary knows. A1 is the reference, being the realistic human.
LADDER = {
    'A0_unpractised': 'A0_human_naive',
    'A1_practised': 'A1_human_practised',
    'A2_sampled': 'A2_synth_population',
    'A4_global_sampled_full': 'A3_synth_speed',
    'A4_keyed_sampled_full': 'A4_synth_keyed',
}
REFERENCE = 'A1_human_practised'


def log(msg, lines):
    print(msg)
    lines.append(str(msg))


# ------------------------------------------------------------ assembly

def attack_rows(mech, gen, imp, forg, meta):
    """Aggregate attack outcomes to counts per
    (fold, participant, password, delta, adversary)."""
    out = []
    for fold, g_f in gen.groupby('fold'):
        victims = g_f['participant_id'].unique()
        if len(victims) < 2:
            continue

        # leave-one-participant-out thresholds, per victim per gap
        thr = {}
        for d, g_d in g_f.groupby('delta'):
            by_v = {v: sub['score'].values for v, sub in g_d.groupby('participant_id')}
            for v in victims:
                others = [s for w, s in by_v.items() if w != v]
                if not others:
                    continue
                pool = np.concatenate(others)
                if len(pool) >= 200:
                    thr[(v, d)] = float(np.percentile(pool, FNMR_BUDGET * 100))

        # which gaps each gallery actually supports
        gal_deltas = (g_f.groupby(GALLERY_KEY)['delta']
                      .apply(lambda s: sorted(set(s))).to_dict())

        pieces = []
        i_f = imp[imp['fold'] == fold]
        a0 = i_f[i_f['imp_practice'] == 0].assign(adversary=LADDER['A0_unpractised'])
        a1 = i_f[i_f['imp_week'] == 8].assign(adversary=LADDER['A1_practised'])
        pieces += [a0, a1]
        if forg is not None:
            f_f = forg[forg['fold'] == fold]
            for raw, name in LADDER.items():
                if raw.startswith('A0') or raw.startswith('A1'):
                    continue
                sub = f_f[f_f['attack'] == raw]
                if len(sub):
                    pieces.append(sub.assign(adversary=name))
        if not pieces:
            continue
        att = pd.concat(pieces, ignore_index=True)

        for key, grp in att.groupby(GALLERY_KEY):
            deltas = gal_deltas.get(key)
            if not deltas:
                continue
            _, pid, pw, _ = key
            for adv, sub in grp.groupby('adversary'):
                s = sub['score'].values
                for d in deltas:
                    t = thr.get((pid, d))
                    if t is None:
                        continue
                    out.append((mech, fold, pid, pw, d, adv,
                                int((s >= t).sum()), len(s)))

    df = pd.DataFrame(out, columns=['mechanism', 'fold', 'participant_id', 'password_id',
                                    'delta', 'adversary', 'n_success', 'n_trials'])
    df['length'] = df['password_id'].map(lambda p: meta[p]['length'])
    df['substitution'] = df['password_id'].map(lambda p: meta[p]['substitution_category'])
    return df


# ------------------------------------------------------------ models

def fit_models(df, lines):
    df = df[df['n_trials'] > 0].copy()
    df['adversary'] = pd.Categorical(
        df['adversary'], categories=[REFERENCE] + sorted(set(df['adversary']) - {REFERENCE}))
    df['delta_c'] = df['delta'] - df['delta'].mean()
    df['length_c'] = df['length'] - df['length'].mean()
    df['success_rate'] = df['n_success'] / df['n_trials']

    formula = ('success_rate ~ adversary * delta_c + adversary * C(mechanism) '
               '+ adversary * C(substitution) + length_c + C(fold)')

    log('', lines)
    log('-' * 78, lines)
    log('PRIMARY: binomial logistic, participant-clustered robust SEs', lines)
    log('-' * 78, lines)
    logit = smf.glm(formula, data=df, family=sm.families.Binomial(),
                    freq_weights=df['n_trials']).fit(
        cov_type='cluster', cov_kwds={'groups': df['participant_id']})
    log(f'observations (cells): {int(logit.nobs):,}   '
        f'underlying trials: {int(df["n_trials"].sum()):,}', lines)
    log(f'clusters (participants): {df["participant_id"].nunique()}', lines)
    log('', lines)

    keep = [p for p in logit.params.index
            if 'adversary' in p or p in ('delta_c', 'length_c', 'Intercept')]
    tab = pd.DataFrame({
        'coef': logit.params[keep],
        'odds_ratio': np.exp(logit.params[keep]),
        'se': logit.bse[keep],
        'z': logit.tvalues[keep],
        'p': logit.pvalues[keep],
    })
    log(tab.to_string(float_format=lambda v: f'{v:.4f}'), lines)

    log('', lines)
    log('-' * 78, lines)
    log('SENSITIVITY: linear probability model, same clustering', lines)
    log('-' * 78, lines)
    lin = smf.wls(formula, data=df, weights=df['n_trials']).fit(
        cov_type='cluster', cov_kwds={'groups': df['participant_id']})
    lin_keep = [p for p in lin.params.index if p in keep]
    comp = pd.DataFrame({
        'logit_p': logit.pvalues[lin_keep],
        'linear_p': lin.pvalues[lin_keep],
        'logit_sign': np.sign(logit.params[lin_keep]),
        'linear_sign': np.sign(lin.params[lin_keep]),
    })
    comp['agree'] = ((comp['logit_sign'] == comp['linear_sign']) &
                     ((comp['logit_p'] < 0.05) == (comp['linear_p'] < 0.05)))
    log(comp.to_string(float_format=lambda v: f'{v:.4f}'), lines)
    n_dis = int((~comp['agree']).sum())
    log('', lines)
    log(f'terms where the two specifications disagree on sign or significance: {n_dis}', lines)
    log('=> ' + ('conclusions are specification-robust' if n_dis == 0 else
                 'CHECK: some conclusions depend on the link function, as in paper 1'), lines)
    return logit, lin, df


# ------------------------------------------------------------ retry policy

def retry_table(df, lines):
    """Attack success once the adversary is allowed several attempts.

    A human gets distinct attempts by trying again. A synthetic attacker draws a
    fresh forgery each time, so its per-attempt successes are close to
    independent and 1-(1-p)^n applies. A scripted attacker is not really limited
    to three, which is the point of the last column.
    """
    g = (df.groupby(['mechanism', 'adversary', 'delta'])[['n_success', 'n_trials']]
         .sum().reset_index())
    g['p'] = g['n_success'] / g['n_trials']
    g[f'p_{N_RETRY}'] = 1 - (1 - g['p']) ** N_RETRY
    g['bits_1'] = -np.log2(g['p'].clip(lower=1e-9))
    g[f'bits_{N_RETRY}'] = -np.log2(g[f'p_{N_RETRY}'].clip(lower=1e-9))
    g['attempts_for_50pc'] = np.ceil(np.log(0.5) / np.log((1 - g['p']).clip(1e-9, 1 - 1e-9)))

    log('', lines)
    log('-' * 78, lines)
    log(f'RETRY POLICY: attack success with 1 and {N_RETRY} allowed attempts', lines)
    log('-' * 78, lines)
    for delta in (0, 7):
        sub = g[g['delta'] == delta]
        if not len(sub):
            continue
        log(f'delta = {delta}', lines)
        piv = sub.pivot(index='adversary', columns='mechanism', values=f'p_{N_RETRY}')
        log(piv.to_string(float_format=lambda v: f'{v:.3f}'), lines)
        log('', lines)
    log('Median attempts for an attacker to reach 50% cumulative success:', lines)
    med = (g[g['delta'] == 0].pivot(index='adversary', columns='mechanism',
                                    values='attempts_for_50pc'))
    log(med.to_string(float_format=lambda v: f'{v:.0f}'), lines)
    return g


def main():
    lines = []
    meta = {r['password_id']: r for r in build_password_metadata()}
    mechs = discover()

    frames = []
    for mech, (gen, imp, forg) in mechs.items():
        frames.append(attack_rows(mech, gen, imp, forg, meta))
        print(f'{mech}: {len(frames[-1]):,} cells')
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(os.path.join(RESULT_DIR, 'exp6_cells.parquet'), index=False)

    log('=' * 78, lines)
    log('EXPERIMENT 6: inferential model and retry policy', lines)
    log('=' * 78, lines)
    log(f'{len(df):,} aggregated cells, {int(df["n_trials"].sum()):,} attack trials', lines)
    log(f'adversary rungs: {sorted(df["adversary"].unique())}', lines)
    log(f'reference level: {REFERENCE}', lines)

    fit_models(df, lines)
    g = retry_table(df, lines)
    g.to_csv(os.path.join(RESULT_DIR, 'exp6_retry.csv'), index=False)

    with open(os.path.join(RESULT_DIR, 'exp6_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwritten to {RESULT_DIR}')


if __name__ == '__main__':
    main()
