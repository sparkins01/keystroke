"""
Experiment 17 (R1 revision): analyse the scores produced by experiment 14.

1. Human retries within one session (reviewer 2, comment 3). One attacker is
   one impostor attacking one gallery; their attempts are the first three
   repetitions of a single week-8 session, taken in the order typed. Reported
   beside R0's design (one sample from each of weeks 6, 7 and 8).

2. A leak that precedes the attack (reviewer 1, comment 2). Attack success
   and marginal security at delta = 0 for the trailing two-week window and
   the cumulative window, beside R0's two horizons, and broken down by
   enrolment week so the effect of a short observation period at week 1 is
   visible.

Thresholds are R0's: leave-one-participant-out within fold, per gap, at a 5%
usability budget, so every number here is directly comparable with R0's.
"""
import os
import re

import numpy as np
import pandas as pd

from exp8_retry_empirical import exact_best_of_n, thresholds, read_dir

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
SCORE_DIR = os.path.join(RESULT_DIR, 'exp14_rescore')
GALLERY = ['fold', 'participant_id', 'password_id', 'enrol_week']
MECHS = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer']
N_RETRY = 3


def genuine(mech):
    path = 'exp1_trials' if mech == 'M1_Manhattan' else f'exp3_trials_{mech}'
    t = read_dir(os.path.join(RESULT_DIR, path))
    return t[t['role'] == 'genuine']


def r0_forgeries(mech):
    if mech == 'M1_Manhattan':
        return pd.read_parquet(os.path.join(RESULT_DIR, 'exp2_forgery_scores.parquet'))
    return read_dir(os.path.join(RESULT_DIR, f'exp3_forgeries_{mech}'))


def load(kind, mech):
    parts = sorted(f for f in os.listdir(SCORE_DIR)
                   if re.fullmatch(fr'{kind}_{mech}_fold\d\.parquet', f))
    return pd.concat([pd.read_parquet(os.path.join(SCORE_DIR, p)) for p in parts],
                     ignore_index=True) if parts else None


def hits(df, thr_by_fold):
    t = [thr_by_fold[f].get(p, np.nan) for f, p in zip(df['fold'], df['participant_id'])]
    t = np.array(t)
    ok = ~np.isnan(t)
    out = df[ok].copy()
    out['hit'] = out['score'].values >= t[ok]
    return out


def main():
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 84)
    emit('EXPERIMENT 17: within-session retries and a leak that precedes the attack')
    emit('=' * 84)

    retry_rows, causal_rows, by_week_rows = [], [], []
    for mech in MECHS:
        sess = load('session', mech)
        caus = load('causal', mech)
        if sess is None or caus is None:
            emit(f'{mech}: scores not found, skipped')
            continue
        gen = genuine(mech)
        forg0 = r0_forgeries(mech)
        forg0 = forg0[forg0['attack'].isin(['A4_keyed_sampled_early', 'A4_keyed_sampled_full'])]
        allf = pd.concat([caus.drop(columns='mechanism'), forg0], ignore_index=True)

        for delta in (0, 7):
            gal = gen[gen['delta'] == delta][GALLERY].drop_duplicates()
            thr = {f: thresholds(gen, f, delta) for f in sorted(gal['fold'].unique())}

            # ---- within-session human retries
            s = hits(sess.merge(gal, on=GALLERY), thr)
            per = (s.sort_values('attempt')
                    .groupby(GALLERY + ['imp_participant_id'])['hit']
                    .agg(list).reset_index())
            per = per[per['hit'].map(len) == N_RETRY]
            first_n = {n: per['hit'].map(lambda h: any(h[:n])).mean() for n in (1, 2, 3)}
            retry_rows.append({'mechanism': mech, 'delta': delta, 'units': len(per),
                               **{f'within-session, {n} attempts': v for n, v in first_n.items()}})

            # ---- forgeries: every horizon, same thresholds
            f = hits(allf.merge(gal, on=GALLERY), thr)
            for att, g in f.groupby('attack'):
                agg = g.groupby(GALLERY)['hit'].agg(['sum', 'size']).reset_index()
                agg = agg[agg['size'] >= N_RETRY]
                causal_rows.append({
                    'mechanism': mech, 'delta': delta, 'attack': att,
                    'fmr': g['hit'].mean(), 'bits': -np.log2(max(g['hit'].mean(), 1e-9)),
                    'three_attempts': np.mean([exact_best_of_n(int(k), int(n), N_RETRY)
                                               for k, n in zip(agg['sum'], agg['size'])])})
                if delta == 0:
                    for wk, gw in g.groupby('enrol_week'):
                        by_week_rows.append({'mechanism': mech, 'attack': att,
                                             'enrol_week': wk, 'fmr': gw['hit'].mean()})

    if not retry_rows:
        raise SystemExit('no scores yet')

    r = pd.DataFrame(retry_rows)
    r0 = pd.read_csv(os.path.join(RESULT_DIR, 'exp8_retry.csv'))
    r0 = r0[r0['adversary'] == 'A1 practised human'][['mechanism', 'delta', 'single_attempt',
                                                      'empirical']]
    r0 = r0.rename(columns={'single_attempt': 'R0 weeks 6-8, per attempt',
                            'empirical': 'R0 weeks 6-8, 3 attempts'})
    r = r.merge(r0, on=['mechanism', 'delta'], how='left')
    r.to_csv(os.path.join(RESULT_DIR, 'exp17_session_retry.csv'), index=False)
    emit()
    emit('-' * 84)
    emit('1. HUMAN RETRIES WITHIN ONE SESSION (week 8, first three repetitions)')
    emit('-' * 84)
    for delta in (0, 7):
        emit(f'delta = {delta}')
        emit(r[r['delta'] == delta].drop(columns='delta').set_index('mechanism').T
             .to_string(float_format=lambda v: f'{v:.3f}'))
        emit()

    c = pd.DataFrame(causal_rows)
    c.to_csv(os.path.join(RESULT_DIR, 'exp17_causal.csv'), index=False)
    names = {'A4_keyed_sampled_trailing': 'two weeks ending at enrolment (causal)',
             'A4_keyed_sampled_upto': 'all weeks up to enrolment (causal)',
             'A4_keyed_sampled_prior': 'two weeks before enrolment week (strictly prior)',
             'A4_keyed_sampled_early': 'weeks 1-2 (R0 two-week)',
             'A4_keyed_sampled_full': 'weeks 1-8 (R0 full)'}
    c['horizon'] = c['attack'].map(names)
    emit('-' * 84)
    emit('2. LEAK HORIZON: A4 keyed sampled forgeries')
    emit('-' * 84)
    for delta in (0, 7):
        for q in ('fmr', 'bits', 'three_attempts'):
            emit(f'delta = {delta}, {q}')
            emit(c[c['delta'] == delta].pivot_table(index='horizon', columns='mechanism', values=q)
                 .to_string(float_format=lambda v: f'{v:.3f}'))
            emit()
    w = pd.DataFrame(by_week_rows)
    w.to_csv(os.path.join(RESULT_DIR, 'exp17_causal_by_week.csv'), index=False)
    emit('delta = 0, attack success by enrolment week')
    for att in ('A4_keyed_sampled_trailing', 'A4_keyed_sampled_prior', 'A4_keyed_sampled_early'):
        emit(f'-- {names[att]}')
        emit(w[w['attack'] == att].pivot_table(index='enrol_week', columns='mechanism', values='fmr')
             .to_string(float_format=lambda v: f'{v:.3f}'))

    with open(os.path.join(RESULT_DIR, 'exp17_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f'\nwritten to {RESULT_DIR}')


if __name__ == '__main__':
    main()
