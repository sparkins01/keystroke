"""
Experiment 5b: why does password composition change how well the transfer
attack works?

Experiment 5 found the ranking of substitution policies by marginal security
nearly reverses between a human attacker and one holding the victim's timings.
A correlation like that needs a mechanism before it can be trusted.

The obvious candidate: the transfer attack builds its forgery from the victim's
per-character dwells and per-digraph intervals, learned from the passwords they
typed *other than* the target. Whenever a character or digraph of the target has
never been seen, the attack falls back to a population estimate scaled by the
victim's overall speed, which experiment 3 showed is much weaker. A password
whose characters and digraphs are rare elsewhere in the corpus therefore gets
attacked with a mostly generic forgery.

If coverage predicts attack success across composition categories, the
composition result has a mechanism rather than being a curiosity.
"""
import os

import numpy as np
import pandas as pd

from dataset_cache import load_dataset
from password_metadata import build_password_metadata
from exp2_transfer import canonical_keys, per_password_tables, victim_profile

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


def coverage_table():
    dataset = load_dataset()
    meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}
    tables = per_password_tables(dataset, dataset.participant_ids, weeks=None)

    rows = []
    for pid, tab in tables.items():
        for pw_id in dataset.password_ids:
            if pw_id not in meta:
                continue
            keys = keys_by_pw[pw_id]
            if len(keys) < 3:
                continue
            vp = victim_profile(tab, exclude_pw=pw_id)
            assert pw_id not in vp['used_passwords']
            if not vp['char']:
                continue

            digraphs = list(zip(keys[:-1], keys[1:]))
            rows.append({
                'participant_id': pid,
                'password_id': pw_id,
                'substitution_category': meta[pw_id]['substitution_category'],
                'length': meta[pw_id]['length'],
                'char_coverage': float(np.mean([k in vp['char'] for k in keys])),
                'digraph_coverage': float(np.mean([d in vp['digraph'] for d in digraphs])),
            })
    return pd.DataFrame(rows)


def main():
    cov = coverage_table()
    cov.to_csv(os.path.join(RESULT_DIR, 'exp5b_coverage.csv'), index=False)

    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 80)
    emit('EXPERIMENT 5b: how much of a target password can the attacker cover?')
    emit('=' * 80)
    emit('Fraction of the target password\'s characters and digraphs that appear')
    emit('in the victim\'s profile built from every OTHER password they typed.')
    emit('Uncovered positions fall back to a population estimate, which is the')
    emit('weak form of the attack.')
    emit()

    for group in ('substitution_category', 'length'):
        g = cov.groupby(group).agg(
            char_coverage=('char_coverage', 'mean'),
            digraph_coverage=('digraph_coverage', 'mean'),
            n=('char_coverage', 'size'),
        ).reset_index().sort_values('digraph_coverage', ascending=False)
        emit(f'By {group}, best-covered first:')
        emit(g.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        emit()

    # does coverage predict attack success across composition categories?
    pol = os.path.join(RESULT_DIR, 'exp5_policy.csv')
    if os.path.exists(pol):
        p = pd.read_csv(pol)
        p = p[(p['grouping'] == 'substitution_category') & p['leaked_fmr'].notna()]
        if len(p):
            c = cov.groupby('substitution_category')[
                ['char_coverage', 'digraph_coverage']].mean().reset_index()
            emit('-' * 80)
            emit('Does coverage predict attack success?')
            emit('-' * 80)
            for mech, sub in p.groupby('mechanism'):
                m = sub.merge(c, on='substitution_category')
                if len(m) < 3:
                    continue
                r_ch = m['digraph_coverage'].corr(m['leaked_fmr'])
                emit(f'{mech}: corr(digraph coverage, leaked-attack FMR) = {r_ch:+.3f} '
                     f'over {len(m)} categories')
                emit(m[['substitution_category', 'digraph_coverage',
                        'leaked_fmr']].sort_values('digraph_coverage')
                     .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
                emit()
            emit('A positive correlation means the attack succeeds precisely where the')
            emit("victim's other passwords cover the target's digraphs, which is the")
            emit('mechanism the composition result needs.')

    with open(os.path.join(RESULT_DIR, 'exp5b_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
