"""
Experiment 15 (R1 revision): re-run the A4 analyses on the two-week leak.

Reviewer 2 (comment 1) found that the headline figures attributed to a two-week
leak were computed from the full-horizon attack. Checking the code showed the
same was true of every downstream A4 analysis: the logistic model (exp6), the
composition table and figure (exp5), and the coverage model (exp10). The
two-week forgeries were already scored, so nothing is re-run except the
analysis. Each function is reused unchanged, with only its attack label (and,
for coverage, the observation window) switched to the two-week horizon.

Outputs sit beside the full-horizon originals with an `_early` suffix, so the
two can be reported side by side.
"""
import os

import numpy as np
import pandas as pd

import exp5_policy
import exp6_model
import exp10_coverage_model
from dataset_cache import load_dataset
from password_metadata import build_password_metadata
from exp2_transfer import canonical_keys, per_password_tables, victim_profile, EARLY_WEEKS
from exp3_analyse import discover

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
EARLY = {'A3': 'A4_global_sampled_early', 'A4': 'A4_keyed_sampled_early'}


def coverage_table(weeks):
    """exp5b's coverage table, for a chosen observation window."""
    dataset = load_dataset()
    meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}
    tables = per_password_tables(dataset, dataset.participant_ids, weeks=weeks)
    rows = []
    for pid, tab in tables.items():
        for pw_id in dataset.password_ids:
            keys = keys_by_pw[pw_id]
            if pw_id not in meta or len(keys) < 3:
                continue
            vp = victim_profile(tab, exclude_pw=pw_id)
            assert pw_id not in vp['used_passwords']
            if not vp['char']:
                continue
            digraphs = list(zip(keys[:-1], keys[1:]))
            rows.append({
                'participant_id': pid, 'password_id': pw_id,
                'substitution_category': meta[pw_id]['substitution_category'],
                'length': meta[pw_id]['length'],
                'char_coverage': float(np.mean([k in vp['char'] for k in keys])),
                'digraph_coverage': float(np.mean([d in vp['digraph'] for d in digraphs])),
            })
    return pd.DataFrame(rows)


def main():
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    meta = {r['password_id']: r for r in build_password_metadata()}
    mechs = discover()

    emit('=' * 84)
    emit('EXPERIMENT 15: the A4 analyses repeated on the two-week leak')
    emit('=' * 84)

    # ---------------------------------------------------------- composition (exp5)
    exp5_policy.LEAKED_RUNG = EARLY['A4']
    emit()
    emit('-' * 84)
    emit('COMPOSITION (as R0 Table 7 / Fig. 4), delta = 0, 5% budget within category')
    emit('-' * 84)
    pol = []
    for mech, (gen, imp, forg) in mechs.items():
        tab = exp5_policy.policy_table(gen, imp, forg, meta, delta=0,
                                       group='substitution_category')
        ra = exp5_policy.rank_agreement(tab, 'substitution_category')
        emit(f'-- {mech}')
        emit(tab[['substitution_category', 'human_bits', 'leaked_fmr', 'leaked_bits']]
             .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        if ra:
            emit(f'   best-to-worst vs human : {ra[0]}')
            emit(f'   best-to-worst vs leak  : {ra[1]}')
            emit(f'   Spearman rho = {ra[2]:+.3f}')
        pol.append(tab.assign(mechanism=mech, grouping='substitution_category'))
        tab_len = exp5_policy.policy_table(gen, imp, forg, meta, delta=0, group='length')
        pol.append(tab_len.assign(mechanism=mech, grouping='length'))
        if 'leaked_bits' in tab_len.columns:
            emit(f'   leaked bits by length: {tab_len["leaked_bits"].min():.3f} '
                 f'to {tab_len["leaked_bits"].max():.3f}')
    pd.concat(pol, ignore_index=True).to_csv(
        os.path.join(RESULT_DIR, 'exp5_policy_early.csv'), index=False)

    # ---------------------------------------------------------- model (exp6)
    exp6_model.LADDER = {**{k: v for k, v in exp6_model.LADDER.items()
                            if not k.startswith('A4_')},
                         EARLY['A3']: 'A3_synth_speed',
                         EARLY['A4']: 'A4_synth_keyed'}
    frames = []
    for mech, (gen, imp, forg) in mechs.items():
        frames.append(exp6_model.attack_rows(mech, gen, imp, forg, meta))
    cells = pd.concat(frames, ignore_index=True)
    cells.to_parquet(os.path.join(RESULT_DIR, 'exp15_cells_early.parquet'), index=False)

    emit()
    emit('-' * 84)
    emit('LOGISTIC MODEL (as exp6), two-week leak for A3 and A4')
    emit('-' * 84)
    model_lines = []
    exp6_model.fit_models(cells, model_lines)
    lines.extend(model_lines)

    # ---------------------------------------------------------- coverage (exp10)
    cov = coverage_table(set(EARLY_WEEKS))
    cov.to_csv(os.path.join(RESULT_DIR, 'exp5b_coverage_early.csv'), index=False)
    emit()
    emit('-' * 84)
    emit('DIGRAPH COVERAGE FROM A TWO-WEEK LEAK, by category')
    emit('-' * 84)
    emit(cov.groupby('substitution_category')['digraph_coverage'].mean()
         .sort_values(ascending=False).to_string(float_format=lambda v: f'{v:.3f}'))

    exp10_coverage_model.ATTACK = EARLY['A4']
    out = exp10_coverage_model.outcome_table()
    feats = exp10_coverage_model.password_features()
    df = (out.merge(cov, on=['participant_id', 'password_id'], how='inner')
             .merge(feats, on='password_id', how='left'))
    df = df[df['n_trials'] > 0].copy()
    df['rate'] = df['n_success'] / df['n_trials']
    for c in ('digraph_coverage', 'length', 'n_shifted', 'n_repeated'):
        df[c + '_c'] = df[c] - df[c].mean()
    models = {
        'coverage only': 'rate ~ digraph_coverage_c + length_c + C(mechanism)',
        'both': ('rate ~ digraph_coverage_c + C(substitution_category) + length_c '
                 '+ n_shifted_c + n_repeated_c + C(mechanism)'),
    }
    for name, formula in models.items():
        emit()
        emit(f'COVERAGE MODEL ({name}), two-way clustered on participant and password')
        tab, _ = exp10_coverage_model.two_way_cluster(None, df, formula,
                                                      'participant_id', 'password_id')
        keep = [i for i in tab.index if 'mechanism' not in i]
        emit(tab.loc[keep].to_string(float_format=lambda v: f'{v:.4f}'))

    with open(os.path.join(RESULT_DIR, 'exp15_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
