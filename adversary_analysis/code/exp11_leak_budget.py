"""
Experiment 11: how much typing does the A4 adversary actually observe?

The draft describes the attacker's knowledge as "two weeks" of the victim's
typing. That is a property of this collection schedule, not a transferable
measure. Another study with a different session frequency would mean something
entirely different by the same phrase.

This reports the leak in units that carry across studies: samples, keystrokes,
distinct characters and distinct digraphs, for both attacker horizons.

Revision R1 (reviewer 2, comment 4). The R0 version had two faults. Samples
were counted as the largest per-character dwell count in each password, which
double counts any password with a repeated character (2,852 / 564 = 5.06
keystrokes per sample, below the 6-character minimum). And every total
included the target password, which the attack never sees. Both are fixed
here: samples are counted directly, and each figure is computed per target
over the other 39 passwords, then the median is taken across targets per
victim before the medians across victims are reported.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from dataset_cache import load_dataset
from exp2_transfer import per_password_tables, sample_timings, EARLY_WEEKS

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


def sample_counts(dataset, weeks):
    """{pid: {pw_id: n_samples}}, using the same filter as per_password_tables."""
    out = defaultdict(lambda: defaultdict(int))
    for s in dataset.samples:
        if weeks is not None and s.week not in weeks:
            continue
        keys, _, _ = sample_timings(s)
        if len(keys) < 2:
            continue
        out[s.participant_id][s.password_id] += 1
    return out


def budget(dataset, weeks, label):
    tables = per_password_tables(dataset, dataset.participant_ids, weeks=weeks)
    counts = sample_counts(dataset, weeks)
    all_pws = sorted({pw for tab in tables.values() for pw in tab})
    rows = []
    for pid, tab in tables.items():
        per_target = []
        for target in all_pws:
            chars, digraphs, keystrokes, samples, seen = set(), set(), 0, 0, 0
            for pw, (dw, dg) in tab.items():
                if pw == target:
                    continue
                seen += 1
                for k, vals in dw.items():
                    chars.add(k)
                    keystrokes += len(vals)
                digraphs.update(dg)
                samples += counts[pid][pw]
            per_target.append((seen, samples, keystrokes, len(chars), len(digraphs)))
        med = np.median(np.array(per_target, dtype=float), axis=0)
        rows.append({'horizon': label, 'participant_id': pid,
                     'passwords_seen': med[0], 'samples': med[1],
                     'keystrokes': med[2],
                     'unique_chars': med[3], 'unique_digraphs': med[4]})
    return pd.DataFrame(rows)


def main():
    ds = load_dataset()
    early = budget(ds, set(EARLY_WEEKS), 'early (weeks 1-2)')
    full = budget(ds, None, 'full (weeks 1-8)')
    df = pd.concat([early, full], ignore_index=True)
    df.to_csv(os.path.join(RESULT_DIR, 'exp11_leak_budget.csv'), index=False)

    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 76)
    emit('EXPERIMENT 11: the attacker\'s observation budget, in transferable units')
    emit('=' * 76)
    emit('Per victim, across the passwords other than any given target.')
    emit()
    cols = ['passwords_seen', 'samples', 'keystrokes', 'unique_chars', 'unique_digraphs']
    for label, g in df.groupby('horizon', sort=False):
        emit(f'{label}   n = {len(g)} victims')
        for c in cols:
            emit('  %-17s median %8.0f   IQR %6.0f - %-6.0f  min %6.0f  max %6.0f'
                 % (c, g[c].median(), g[c].quantile(.25), g[c].quantile(.75),
                    g[c].min(), g[c].max()))
        emit()

    e = early.median(numeric_only=True)
    emit('-' * 76)
    emit('For the paper: the two-week horizon gives the adversary a median of')
    emit('%d keystrokes across %d passwords, covering %d distinct characters and'
         % (e['keystrokes'], e['passwords_seen'], e['unique_chars']))
    emit('%d distinct digraphs.' % e['unique_digraphs'])

    with open(os.path.join(RESULT_DIR, 'exp11_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
