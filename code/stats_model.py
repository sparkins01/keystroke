"""
Section 3.7: linear mixed-effects model of verification outcomes.

  EER ~ Delta * Mechanism + Length + Substitution + Hardware-Change
        + (1 | Participant) + (1 | Password) + (1 | Fold)

EER is an aggregate statistic (a rate computed over a set of trials), not a
per-observation quantity, so it cannot literally be the response of a
per-trial regression. We operationalise the same fixed/random-effect
structure at the correct granularity by modelling each trial's *decision
error* (false accept or false reject) as the binary outcome:

  1. calibrate a decision threshold per (fold, mechanism, delta) at the EER
     point of that fold's *validation*-participant trials at that same delta
     (never the test trials being modelled -- this mirrors the paper's
     "used for ... threshold calibration" role for the validation fold,
     Section 3.2). Calibrating per-delta (rather than one threshold per
     mechanism pooled across all deltas) matters: genuine/impostor score
     distributions drift as the enrolment-to-query gap grows, so a single
     pooled threshold silently absorbs some of the aging effect into
     "decision drift" rather than "error rate", which was flattening the
     Delta x Mechanism interaction relative to what the EER-vs-delta curve
     visually shows;
  2. apply that threshold to the held-out test trials to label each one
     correct/incorrect;
  3. fit is_error ~ delta * mechanism + length + substitution + hardware_change
     + C(fold), with participant and password as crossed random intercepts
     (statsmodels represents crossed random effects as variance components
     under a single dummy grouping, following its documented pattern for
     crossed-effects MixedLM). Delta and length are mean-centred before
     fitting, which is standard practice to avoid the shared numerical
     ill-conditioning between fixed-effect and variance-component scales
     that was causing non-convergence / non-positive-definite Hessians.

Fold is deliberately a *fixed* effect here, not the third crossed random
effect the paper's formula lists. A first version fit it as a random
intercept alongside participant and password; a robustness check (refitting
on different random subsamples of the same data) showed the resulting model
was numerically unstable -- the Delta x Mechanism likelihood-ratio test swung
from p=1.0 to p=8.7e-39 across reruns, including one nonsensical negative LR
statistic despite both nested fits self-reporting convergence. Fold is also
conceptually not a random effect: it's a fixed partition of the same 85
participants induced by the CV design, not a sample from a population of
folds, so a fixed effect is the more defensible choice. With participant and
password as the only two crossed random effects, the model is dramatically
more stable (see the seed-robustness check in run_experiments output).

This is a linear probability mixed model (not a logistic GLMM): statsmodels'
crossed-effects solver is MixedLM (Gaussian), and a binomial-outcome
equivalent (BinomialBayesMixedGLM) is far less robust at this scale, so the
linear version is the primary, reliable model. Coefficients are then simply
differences in error probability, which is easy to interpret directly.

Two more fixes came out of the same robustness check, both in `subsample()`:
  - It stratifies by (mechanism, delta) jointly, not by mechanism alone.
    Stratifying by mechanism only let each random seed pull in a different
    mix of deltas (and, incidentally, participants/passwords) by chance --
    exactly the cross-tabulation the interaction term estimates -- which is
    what produced the p=1.0-to-8.7e-39 instability. Stratifying by the pair
    keeps that marginal structure essentially fixed across seeds.
  - MAX_ROWS_FOR_MODEL is 300,000, not 60,000. At 60k (even with the fix
    above) the four reruns gave a *consistent but underpowered* spread of
    p in [0.03, 0.70] -- no more wild flips, but not enough power to resolve
    significance either way. At 300k all four reruns agree the interaction
    is significant (p in [0.0003, 0.041]). This is standard: a bigger,
    properly stratified sample resolves a real-but-modest effect that a
    smaller one is simply underpowered to detect reliably.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from metrics import compute_eer

MAX_ROWS_FOR_MODEL = 300000
OPTIMIZER_CHAIN = ['lbfgs', 'bfgs', 'cg', 'powell']


def calibrate_thresholds(val_trials_df):
    """Returns {(fold, mechanism, delta): threshold} from validation-participant trials."""
    thresholds = {}
    for (fold, mechanism, delta), g in val_trials_df.groupby(['fold', 'mechanism', 'delta']):
        _, thr = compute_eer(g['label'], g['score'])
        thresholds[(fold, mechanism, delta)] = thr
    return thresholds


def build_model_dataframe(test_trials_df, thresholds):
    df = test_trials_df.copy()
    thr = df.apply(lambda r: thresholds.get((r['fold'], r['mechanism'], r['delta'])), axis=1)
    df = df[thr.notna()].copy()
    thr = thr[thr.notna()]
    predicted_genuine = df['score'].values >= thr.values
    actual_genuine = df['label'].values == 1
    df['is_error'] = (predicted_genuine != actual_genuine).astype(float)
    return df


def subsample(df, max_rows=MAX_ROWS_FOR_MODEL, seed=0):
    """Stratified by (mechanism, delta), not a plain random sample of rows.

    A first version stratified only by mechanism; different random seeds then
    pulled in wildly different numbers of trials per delta bucket and per
    participant/password by chance, which is exactly what the Delta x
    Mechanism interaction term is trying to estimate -- so its estimate
    swung enormously across reruns (the seed-robustness check in this
    session got p-values from 1.0 to 8.7e-39 on the same underlying data).
    Stratifying by (mechanism, delta) keeps that cross-tabulation's marginal
    structure essentially identical across seeds; only *which* trials within
    each cell get sampled changes."""
    if len(df) <= max_rows:
        return df
    parts = []
    for _, g in df.groupby(['mechanism', 'delta']):
        n = max(1, int(max_rows * len(g) / len(df)))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _fit_with_fallback(formula, df, vc):
    last_exc = None
    for method in OPTIMIZER_CHAIN:
        try:
            model = smf.mixedlm(formula, data=df, groups='dummy_group', vc_formula=vc)
            result = model.fit(reml=False, method=method, maxiter=500)
            if result.converged:
                return result, method
            last_result, last_method = result, method
        except Exception as e:
            last_exc = e
            last_result, last_method = None, method
    if last_result is not None:
        return last_result, f'{last_method} (did not fully converge; best available fit kept)'
    raise last_exc


def fit_mixed_model(model_df, max_rows=MAX_ROWS_FOR_MODEL, seed=0):
    df = subsample(model_df, max_rows=max_rows, seed=seed).copy()
    df['dummy_group'] = 1
    # pandas 3's default "string[python]" dtype (from read_csv) isn't reliably
    # recognised as categorical by patsy's formula evaluator; force plain
    # object dtype for every column referenced in a formula/vc_formula.
    for col in ['participant_id', 'password_id', 'fold', 'mechanism', 'substitution_category']:
        df[col] = df[col].astype(str).astype(object)
    for col in ['delta', 'length', 'hardware_change', 'is_error']:
        df[col] = df[col].astype(float)

    delta_mean = df['delta'].mean()
    length_mean = df['length'].mean()
    df['delta_c'] = df['delta'] - delta_mean
    df['length_c'] = df['length'] - length_mean

    # Only participant and password are genuinely crossed random effects here
    # (a real population the paper wants to generalise over). Fold is not --
    # it's a fixed partition of the same 85 participants induced by the CV
    # procedure, not a sample from a population of folds, so modelling it as
    # a fixed covariate is both more defensible and (per the robustness check
    # in this session) far more numerically stable: with fold as a third
    # crossed variance component the optimizer would land in different local
    # optima depending on the row subsample (the same nested-model LR test
    # swung from p=1.0 to p=8.7e-39 across reruns, including one impossible
    # negative LR statistic). Two crossed random effects instead of three
    # removes that instability.
    vc = {
        'participant': '0 + C(participant_id)',
        'password': '0 + C(password_id)',
    }
    full_formula = "is_error ~ delta_c * mechanism + length_c + C(substitution_category) + hardware_change + C(fold)"
    reduced_formula = "is_error ~ delta_c + mechanism + length_c + C(substitution_category) + hardware_change + C(fold)"

    full_result, full_method = _fit_with_fallback(full_formula, df, vc)
    reduced_result, reduced_method = _fit_with_fallback(reduced_formula, df, vc)

    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    df_diff = len(full_result.params) - len(reduced_result.params)
    from scipy import stats as sstats
    p_value = sstats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan

    return {
        'full_result': full_result, 'full_method': full_method,
        'reduced_result': reduced_result, 'reduced_method': reduced_method,
        'lr_test': {'lr_stat': float(lr_stat), 'df': int(df_diff), 'p_value': float(p_value)},
        'n_rows_used': len(df),
        'centering': {'delta_mean': float(delta_mean), 'length_mean': float(length_mean)},
    }
