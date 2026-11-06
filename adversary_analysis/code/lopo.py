"""
Leave-one-participant-out thresholds, shared by every analysis (R1 revision).

The paper states that the operating point applied to a victim is estimated
from the genuine scores of the other test participants in the same fold and at
the same gap. R0 followed this in the model, retry and coverage analyses, but
the ladder, ageing, budget and composition tables pooled one threshold across
all folds. Pooling across folds is invalid for M3 and M4, whose scores are not
comparable between folds. Every analysis now calls this module, so there is a
single procedure.

`gen` is always the genuine trials for one gap (and, where relevant, one
password group). Thresholds are keyed by (fold, participant_id).
"""
import numpy as np

MIN_POOL = 200


def lopo_thresholds(gen, budgets, min_pool=MIN_POOL):
    """{budget: {(fold, participant): threshold}}, each from the genuine scores
    of the other participants in that fold."""
    budgets = list(budgets)
    out = {b: {} for b in budgets}
    q = [b * 100 for b in budgets]
    for fold, g in gen.groupby('fold'):
        by_v = {v: s['score'].values for v, s in g.groupby('participant_id')}
        for v in by_v:
            others = [s for w, s in by_v.items() if w != v]
            if not others:
                continue
            pool = np.concatenate(others)
            if len(pool) < min_pool:
                continue
            for b, t in zip(budgets, np.percentile(pool, q)):
                out[b][(fold, v)] = float(t)
    return out


def hits(df, thr):
    """Boolean hit per row of `df` (score >= its victim's threshold), and the
    mask of rows whose victim has a threshold. Rows without one are dropped."""
    t = np.array([thr.get(k, np.nan) for k in zip(df['fold'], df['participant_id'])])
    ok = ~np.isnan(t)
    return df['score'].values[ok] >= t[ok], ok


def fmr(df, thr):
    h, _ = hits(df, thr)
    return float(h.mean()) if len(h) else np.nan
