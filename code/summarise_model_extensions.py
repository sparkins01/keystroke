"""
Replication table for the two extended model terms.

The published analysis treats a term as supported only if it survives three
independent pipeline replicates and a change of link function. The same bar
is applied here to the new terms, and this script collects the evidence into
one table.

Reads the per-replicate outputs of exp_model_extensions.py and the balanced
subset from exp_model_robustness.py, and writes
results/revision/model_extensions_replication.csv.
"""
import os

import pandas as pd

OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'revision')
REPLICATES = ['primary', 'seed10', 'seed20']
TERMS = ['delta_c', 'length_c', 'delta_c:length_c', 'enrol_week_c',
         'enrol_week_c:delta_c']


def collect():
    frames = []
    for rep in REPLICATES:
        path = os.path.join(OUT_DIR, 'model_extensions_terms_%s.csv' % rep)
        if not os.path.exists(path):
            print('missing: %s' % path)
            continue
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    balanced = os.path.join(OUT_DIR, 'model_robustness_balanced.csv')
    if os.path.exists(balanced):
        b = pd.read_csv(balanced)
        b['spec'] = 'S3_both'
        b['link'] = 'linear'
        b['replicate'] = 'balanced_rectangle'
        df = pd.concat([df, b], ignore_index=True)
    return df


def verdict(group):
    """A term is supported when it is significant and keeps one sign
    everywhere it is tested."""
    sig = group['p'] < 0.05
    same_sign = (group['coef'] > 0).all() or (group['coef'] < 0).all()
    n_sig = int(sig.sum())
    n = len(group)
    if n_sig == n and same_sign:
        return 'supported (%d/%d, consistent sign)' % (n_sig, n)
    if n_sig == 0:
        return 'not supported (0/%d)' % n
    if same_sign:
        return 'mixed (%d/%d significant, consistent sign)' % (n_sig, n)
    return 'unstable (%d/%d significant, sign flips)' % (n_sig, n)


def main():
    df = collect()
    df = df[df['term'].isin(TERMS)]

    print('=' * 94)
    print('TERM BY TERM, ACROSS REPLICATES (linear specification, S3 unless noted)')
    print('=' * 94)
    linear = df[(df['link'] == 'linear') & (df['spec'] == 'S3_both')]
    for term in TERMS:
        g = linear[linear['term'] == term]
        if g.empty:
            continue
        print()
        print('--- %s ---' % term)
        print(g[['replicate', 'coef', 'se', 'z', 'p']].to_string(
            index=False, float_format=lambda v: '%.6g' % v))
        print('    verdict: %s' % verdict(g))

    print()
    print('=' * 94)
    print('LINK-FUNCTION CHECK ON THE PRIMARY RUN')
    print('=' * 94)
    prim = df[(df['replicate'] == 'primary') & (df['spec'] == 'S3_both')]
    print(prim[['link', 'term', 'coef', 'p']].to_string(
        index=False, float_format=lambda v: '%.6g' % v))

    rows = []
    for term in TERMS:
        g = linear[linear['term'] == term]
        if g.empty:
            continue
        rows.append({
            'term': term,
            'n_tested': len(g),
            'n_significant': int((g['p'] < 0.05).sum()),
            'coef_min': g['coef'].min(),
            'coef_max': g['coef'].max(),
            'sign_consistent': bool((g['coef'] > 0).all() or (g['coef'] < 0).all()),
            'verdict': verdict(g),
        })
    summary = pd.DataFrame(rows)
    print()
    print('=' * 94)
    print('SUMMARY')
    print('=' * 94)
    print(summary.to_string(index=False, float_format=lambda v: '%.6g' % v))

    path = os.path.join(OUT_DIR, 'model_extensions_replication.csv')
    summary.to_csv(path, index=False)
    df.to_csv(os.path.join(OUT_DIR, 'model_extensions_all_terms.csv'), index=False)
    print()
    print('wrote %s' % os.path.abspath(path))


if __name__ == '__main__':
    main()
