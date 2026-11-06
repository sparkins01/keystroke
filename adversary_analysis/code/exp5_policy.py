"""
Experiment 5, RQ4: does password composition change the *marginal
security* a keystroke second factor provides, and does that ranking match the
ranking by raw accuracy?

This is where the study connects back to Parkinson et al. 2021 and 2023, which
recommended password policies chosen to maximise keystroke matching accuracy.
Those recommendations were derived against zero-effort impostors. If a policy
that looks good against a random stranger looks bad against an attacker holding
the victim's timings, the earlier recommendation is not merely incomplete, it
points the wrong way, and saying so is a concrete contribution rather than a
restatement.

No new experiments. Length and substitution category are already on every trial
row, and the forgery rows carry password_id, which joins to the same metadata.
"""
import os

import numpy as np
import pandas as pd

from password_metadata import build_password_metadata
from exp3_analyse import discover, GALLERY_KEY, FNMR_BUDGET
from lopo import lopo_thresholds, fmr as lopo_fmr

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
HUMAN_RUNG = 'A1_practised'              # best a human attacker manages
LEAKED_RUNG = 'A4_keyed_sampled_full'    # attacker holds the victim's timings


def bits(fmr):
    # nan when no victim in the group has a LOPO threshold (length 7 is one
    # password, too few genuine scores per fold)
    if np.isnan(fmr):
        return np.nan
    return float(-np.log2(fmr)) if fmr > 0 else np.inf


def policy_table(gen, imp, forg, meta, delta=0, group='substitution_category'):
    """One row per policy group: how much security the factor buys against a
    human attacker and against a leaked-profile attacker.

    The usability budget is applied *within* each group, because the genuine
    score distribution differs by password composition. Using one global
    threshold would confound composition with calibration.
    """
    gen = gen[gen['delta'] == delta].copy()
    if not len(gen):
        return pd.DataFrame()

    gen[group] = gen['password_id'].map(lambda p: meta[p][group])
    galleries = gen[GALLERY_KEY].drop_duplicates()
    imp_d = imp.merge(galleries, on=GALLERY_KEY, how='inner').copy()
    imp_d[group] = imp_d['password_id'].map(lambda p: meta[p][group])

    forg_d = None
    if forg is not None and len(forg):
        forg_d = forg.merge(galleries, on=GALLERY_KEY, how='inner').copy()
        forg_d[group] = forg_d['password_id'].map(lambda p: meta[p][group])

    rows = []
    for key in sorted(gen[group].unique()):
        gk = gen[gen[group] == key]
        if len(gk) < 200:
            continue
        # leave-one-participant-out within fold, on this group's genuine scores
        thr = lopo_thresholds(gk, [FNMR_BUDGET])[FNMR_BUDGET]

        human = imp_d[(imp_d[group] == key) & (imp_d['imp_week'] == 8)]
        if len(human) < 100:
            continue
        row = {group: key, 'n_genuine': len(gk), 'human_fmr': lopo_fmr(human, thr)}
        row['human_bits'] = bits(row['human_fmr'])

        if forg_d is not None:
            leaked = forg_d[(forg_d[group] == key) & (forg_d['attack'] == LEAKED_RUNG)]
            if len(leaked) >= 50:
                row['leaked_fmr'] = lopo_fmr(leaked, thr)
                row['leaked_bits'] = bits(row['leaked_fmr'])
        rows.append(row)
    return pd.DataFrame(rows)


def rank_agreement(tab, group):
    """Do the two threat models rank policies the same way?"""
    if 'leaked_bits' not in tab.columns or tab['leaked_bits'].isna().any():
        return None
    by_human = list(tab.sort_values('human_bits', ascending=False)[group])
    by_leaked = list(tab.sort_values('leaked_bits', ascending=False)[group])
    rho = tab['human_bits'].corr(tab['leaked_bits'], method='spearman')
    return by_human, by_leaked, rho


def main():
    meta = {r['password_id']: r for r in build_password_metadata()}
    mechs = discover()
    if not mechs:
        raise SystemExit('no results found')

    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 86)
    emit('EXPERIMENT 5 (RQ4): does password composition change marginal security?')
    emit('=' * 86)
    emit(f'Fresh template (delta=0), {FNMR_BUDGET:.0%} usability budget applied within')
    emit('each group. "human" = practised human impostor; "leaked" = attacker holding')
    emit("the victim's cross-password timings.")
    emit()

    all_rows = []
    for group in ('substitution_category', 'length'):
        emit('#' * 86)
        emit(f'Grouped by {group}')
        emit('#' * 86)
        for mech, (gen, imp, forg) in mechs.items():
            tab = policy_table(gen, imp, forg, meta, delta=0, group=group)
            if tab.empty:
                continue
            emit(f'-- {mech}')
            cols = [group, 'n_genuine', 'human_fmr', 'human_bits']
            if 'leaked_bits' in tab.columns:
                cols += ['leaked_fmr', 'leaked_bits']
            emit(tab[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

            ra = rank_agreement(tab, group)
            if ra:
                by_human, by_leaked, rho = ra
                emit(f'   best-to-worst vs human : {by_human}')
                emit(f'   best-to-worst vs leak  : {by_leaked}')
                emit(f'   Spearman rho = {rho:+.3f}'
                     f'   {"ORDERING PRESERVED" if by_human == by_leaked else "ORDERING CHANGES"}')
            emit()
            all_rows.append(tab.assign(mechanism=mech, grouping=group))

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(os.path.join(RESULT_DIR, 'exp5_policy.csv'), index=False)

    emit('-' * 86)
    emit('Reading')
    emit('-' * 86)
    emit('If the ordering is preserved, the 2021/2023 policy advice still holds and')
    emit('this is a robustness result. If it changes, a policy chosen to maximise')
    emit('accuracy against strangers is not the policy that survives a real attacker,')
    emit('which is a direct correction to that advice.')

    with open(os.path.join(RESULT_DIR, 'exp5_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
