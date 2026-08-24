import os
RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
"""
Second, more numerically robust binary-outcome sensitivity check: a standard
logistic regression (no random effects) with cluster-robust standard errors
clustered by participant. This sidesteps the crossed-random-effects
variational-Bayes convergence problems seen with BinomialBayesMixedGLM,
while still directly testing "does a logit link change the interaction
conclusion" and still accounting for within-participant correlation (the
dominant source of non-independence) via clustering, per Bolle et al. 2004's
argument that naive SEs understate uncertainty when scores from the same
subject are correlated.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_model import subsample, MAX_ROWS_FOR_MODEL

def prep(path, seed):
    df = pd.read_parquet(path)
    df = subsample(df, max_rows=MAX_ROWS_FOR_MODEL, seed=seed).copy()
    for col in ['participant_id', 'password_id', 'fold', 'mechanism', 'substitution_category']:
        df[col] = df[col].astype(str).astype(object)
    for col in ['delta', 'length', 'hardware_change', 'is_error']:
        df[col] = df[col].astype(float)
    df['delta_c'] = df['delta'] - df['delta'].mean()
    df['length_c'] = df['length'] - df['length'].mean()
    return df

runs = [
    ('primary (seed 0)', os.path.join(RESULT_DIR, 'mixed_model_input.parquet')),
    ('replicate seed10', os.path.join(RESULT_DIR, 'replicate_seed10', 'mixed_model_input.parquet')),
    ('replicate seed20', os.path.join(RESULT_DIR, 'replicate_seed20', 'mixed_model_input.parquet')),
]

formula = "is_error ~ delta_c * mechanism + length_c + C(substitution_category) + hardware_change + C(fold) + C(participant_id) + C(password_id)"
# Including participant/password as fixed effects (rather than random) is
# infeasible here (85 + 40 levels is fine, but let's first try WITHOUT them
# as fixed effects, matching the population-averaged GEE-style approach, and
# rely on cluster-robust SEs to absorb the correlation instead).
formula_no_re = "is_error ~ delta_c * mechanism + length_c + C(substitution_category) + hardware_change + C(fold)"

for label, path in runs:
    print(f"\n=== {label} ===")
    df = prep(path, seed=0)
    model = smf.logit(formula_no_re, data=df)
    result = model.fit(disp=0, cov_type='cluster', cov_kwds={'groups': df['participant_id']})
    interaction_terms = [n for n in result.params.index if 'delta_c:mechanism' in n]
    summary_df = pd.DataFrame({
        'term': interaction_terms,
        'coef': [result.params[t] for t in interaction_terms],
        'se': [result.bse[t] for t in interaction_terms],
        'z': [result.tvalues[t] for t in interaction_terms],
        'p': [result.pvalues[t] for t in interaction_terms],
    })
    print(summary_df.to_string(index=False))
    safe = label.replace(' ', '_').replace('(', '').replace(')', '')
    summary_df.to_csv(os.path.join(RESULT_DIR, 'stats_model', f'logit_clusterrobust_{safe}.csv'), index=False)

print("\nDone.")
