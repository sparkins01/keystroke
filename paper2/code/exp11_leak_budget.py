"""
Experiment 11: how much typing does the A4 adversary actually observe?

The draft describes the attacker's knowledge as "two weeks" of the victim's
typing. That is a property of this collection schedule, not a transferable
measure. Another study with a different session frequency would mean something
entirely different by the same phrase.

This reports the leak in units that carry across studies: samples, keystrokes,
distinct characters and distinct digraphs, for both attacker horizons.
"""
import os

import numpy as np
import pandas as pd

from dataset_cache import load_dataset
from exp2_transfer import per_password_tables, EARLY_WEEKS

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


def budget(dataset, weeks, label):
    tables = per_password_tables(dataset, dataset.participant_ids, weeks=weeks)
    rows = []
    for pid, tab in tables.items():
        chars, digraphs, keystrokes, samples = set(), set(), 0, 0
        for pw, (dw, dg) in tab.items():
            for k, vals in dw.items():
                chars.add(k)
                keystrokes += len(vals)
            for k in dg:
                digraphs.add(k)
            samples += max((len(v) for v in dw.values()), default=0)
        rows.append({'horizon': label, 'participant_id': pid,
                     'passwords_seen': len(tab), 'samples': samples,
                     'keystrokes': keystrokes,
                     'unique_chars': len(chars), 'unique_digraphs': len(digraphs)})
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
