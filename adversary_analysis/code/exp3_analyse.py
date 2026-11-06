"""
Adversary-ladder analysis across every mechanism that has results on disk.

The question it exists to answer: does marginal security lost to a leaked victim
profile scale with the matcher's baseline accuracy? Experiment 3 found M4 lost
more than M1, which would mean the features that make a fixed-text matcher
accurate are the same ones that transfer across passwords and therefore leak.
Two points is a coincidence; four is a finding.

Mechanisms are discovered from the results directory, so it reports on whatever
has finished. M1 is restricted to the folds the other mechanisms have, keeping
every comparison matched.
"""
import os
import re

import numpy as np
import pandas as pd

from metrics import compute_eer
from lopo import lopo_thresholds, fmr as lopo_fmr

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FNMR_BUDGET = 0.05
GALLERY_KEY = ['fold', 'participant_id', 'password_id', 'enrol_week']

# the two rungs the leakability claim is measured between: identical forgery
# construction, differing only in whether the attacker knows the victim
BASELINE_RUNG = 'A2_sampled'
LEAKED_RUNG = 'A4_keyed_sampled_full'


def _read_dir(path):
    if not os.path.isdir(path):
        return None
    parts = sorted(f for f in os.listdir(path) if f.endswith('.parquet'))
    if not parts:
        return None
    return pd.concat([pd.read_parquet(os.path.join(path, p)) for p in parts],
                     ignore_index=True)


def discover():
    """{mechanism: (genuine, impostor, forgeries)} for everything on disk."""
    out = {}

    trials = _read_dir(os.path.join(RESULT_DIR, 'exp1_trials'))
    fpath = os.path.join(RESULT_DIR, 'exp2_forgery_scores.parquet')
    if trials is not None:
        forg = pd.read_parquet(fpath) if os.path.exists(fpath) else None
        out['M1_Manhattan'] = (trials[trials['role'] == 'genuine'],
                               trials[trials['role'] == 'impostor'], forg)

    for name in sorted(os.listdir(RESULT_DIR)):
        m = re.fullmatch(r'exp3_trials_(.+)', name)
        if not m:
            continue
        mech = m.group(1)
        trials = _read_dir(os.path.join(RESULT_DIR, name))
        if trials is None:
            continue
        forg = _read_dir(os.path.join(RESULT_DIR, f'exp3_forgeries_{mech}'))
        out[mech] = (trials[trials['role'] == 'genuine'],
                     trials[trials['role'] == 'impostor'], forg)
    return out


def attack_table(gen, imp, forg, folds=None):
    if folds is not None:
        gen = gen[gen['fold'].isin(folds)]
        imp = imp[imp['fold'].isin(folds)]
        if forg is not None:
            forg = forg[forg['fold'].isin(folds)]

    rows = []
    for delta in sorted(gen['delta'].unique()):
        gd = gen[gen['delta'] == delta]
        g = gd['score'].values
        if len(g) < 100:
            continue
        galleries = gd[GALLERY_KEY].drop_duplicates()
        imp_d = imp.merge(galleries, on=GALLERY_KEY, how='inner')

        levels = {
            'A0_unpractised': imp_d[imp_d['imp_practice'] == 0],
            'A1_practised': imp_d[imp_d['imp_week'] == 8],
        }
        if forg is not None and len(forg):
            forg_d = forg.merge(galleries, on=GALLERY_KEY, how='inner')
            for label in sorted(forg_d['attack'].unique()):
                levels[label] = forg_d[forg_d['attack'] == label]

        thr = lopo_thresholds(gd, [FNMR_BUDGET])[FNMR_BUDGET]
        for name, sub in levels.items():
            s = sub['score'].values
            if len(s) < 50:
                continue
            # the EER is threshold-free and stays a pooled summary
            eer, _ = compute_eer(np.r_[np.ones(len(g)), np.zeros(len(s))], np.r_[g, s])
            fmr = lopo_fmr(sub, thr)
            rows.append({
                'delta': int(delta), 'attack': name, 'n': len(s), 'eer': eer,
                'fmr': fmr, 'bits': float(-np.log2(fmr)) if fmr > 0 else np.inf,
            })
    return pd.DataFrame(rows)


def analyse():
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    mechs = discover()
    if not mechs:
        raise SystemExit('no results found')

    # match folds across mechanisms so nothing is compared on different subjects
    fold_sets = [set(g['fold'].unique()) for g, _, _ in mechs.values()]
    common = sorted(set.intersection(*fold_sets))

    emit('=' * 88)
    emit('ADVERSARY LADDER ACROSS MECHANISMS')
    emit('=' * 88)
    emit(f'mechanisms : {", ".join(mechs)}')
    emit(f'folds      : {common} (intersection, so all mechanisms are matched)')
    emit(f'metric     : FMR at a {FNMR_BUDGET:.0%} usability budget')
    emit()

    tables = {m: attack_table(*v, folds=common) for m, v in mechs.items()}

    for delta in (0, 7):
        emit('-' * 88)
        emit(f'Delta = {delta}')
        emit('-' * 88)
        merged = None
        for mech, tab in tables.items():
            sub = tab[tab['delta'] == delta][['attack', 'fmr']]
            sub = sub.rename(columns={'fmr': mech})
            merged = sub if merged is None else merged.merge(sub, on='attack', how='outer')
        if merged is not None:
            emit(merged.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        emit()

    emit('-' * 88)
    emit('Does a more accurate matcher lose more security to a leaked profile?')
    emit('-' * 88)
    emit(f'Both rungs use identical forgery construction. {BASELINE_RUNG} knows nothing')
    emit(f'about the victim; {LEAKED_RUNG} holds their cross-password timings.')
    emit()
    rows = []
    for mech, tab in tables.items():
        t0 = tab[tab['delta'] == 0].set_index('attack')
        if not {BASELINE_RUNG, LEAKED_RUNG, 'A1_practised'} <= set(t0.index):
            continue
        base, leaked = t0.loc[BASELINE_RUNG, 'bits'], t0.loc[LEAKED_RUNG, 'bits']
        rows.append({
            'mechanism': mech,
            'baseline_accuracy_bits': t0.loc['A1_practised', 'bits'],
            'uninformed_attack_bits': base,
            'leaked_profile_bits': leaked,
            'security_lost_bits': base - leaked,
        })
    lead = pd.DataFrame(rows).sort_values('baseline_accuracy_bits')
    emit(lead.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    emit()
    if len(lead) >= 3:
        r = np.corrcoef(lead['baseline_accuracy_bits'], lead['security_lost_bits'])[0, 1]
        mono = lead['security_lost_bits'].is_monotonic_increasing
        emit(f'correlation(baseline accuracy, security lost) = {r:+.3f} '
             f'over {len(lead)} mechanisms')
        emit(f'monotone in baseline accuracy: {mono}')
        emit('=> ' + ('the more accurate the matcher, the more it loses to the leak'
                      if mono and r > 0 else
                      'no clean monotone relationship; treat the M1/M4 contrast as '
                      'a two-point observation rather than a trend'))
    else:
        emit('(needs at least 3 mechanisms to say anything about a trend)')
    emit()

    emit('-' * 88)
    emit('Artefact check: median forgery vs the same attack with realistic spread')
    emit('-' * 88)
    pairs = [('A2_population', 'A2_sampled'),
             ('A4_global_median_full', 'A4_global_sampled_full'),
             ('A4_keyed_median_full', 'A4_keyed_sampled_full')]
    for mech, tab in tables.items():
        t0 = tab[tab['delta'] == 0].set_index('attack')
        parts = []
        for med, samp in pairs:
            if med in t0.index and samp in t0.index:
                parts.append(f'{med.replace("_full", ""):<22} {t0.loc[med, "fmr"] - t0.loc[samp, "fmr"]:+.3f}')
        if parts:
            emit(f'{mech}:')
            for p in parts:
                emit(f'   gap  {p}')
    emit()

    out = pd.concat([t.assign(mechanism=m) for m, t in tables.items()], ignore_index=True)
    out.to_csv(os.path.join(RESULT_DIR, 'exp3_comparison.csv'), index=False)
    lead.to_csv(os.path.join(RESULT_DIR, 'exp4_leakability.csv'), index=False)
    with open(os.path.join(RESULT_DIR, 'exp3_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    analyse()
