"""
Revision experiment for TBIOM reviewer 2, points 2 and 6, and the associate
editor's comment on password memorisation.

The reviewer's objection rests on a premise: that keystroke patterns for a
fixed phrase stabilise after a few repetitions, so the early sessions of this
study measure users learning unfamiliar passwords rather than templates
ageing. The premise is testable directly in the raw timings, without any
matcher, and that is what this script does.

Two timing summaries are computed per attempt:
  total time     first key press to last key release
  mean flight    mean press-to-press interval

Dwell is reported separately as a data-quality series rather than as a
learning curve. In this collection 26.7% of key events carry a release
timestamp identical to their press timestamp, and the median dwell is about
4 ms against the 70 to 150 ms typical of real key holds, so the instrument
did not reliably capture key-up. A ratio built on that baseline is not
meaningful, and the degeneracy is itself worth reporting.

Everything is aggregated on medians. Typing times have a long right tail,
from attempts where a participant paused mid-password, and a single 255-second
attempt is enough to move a weekly mean. One attempt of 94,894 has a
non-positive duration, from an out-of-order event, and is dropped.

Each (participant, password) series is normalised by its own week-1 median,
so participants who are simply fast do not dominate, and what is left is the
shape of the learning curve.

The decisive question for the revision is where the curve flattens. The
cumulative column of the weekly table answers it: it gives the fraction of
the whole week-1-to-week-8 change that has already happened by each week.

Writes results/revision/familiarisation_*.csv and the figure
results/figures/familiarisation.png.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import load_samples

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(RESULT_DIR, 'revision')
FIG_DIR = os.environ.get('KD_FIG_DIR', os.path.join(RESULT_DIR, 'figures'))
METRICS = ['total_time', 'mean_flight']


def attempt_table(samples):
    """One row per attempt. Participant names are present on the Sample
    namedtuple and are deliberately not carried across."""
    rows = []
    for s in samples:
        presses = np.array([c.press.value for c in s.char_events], dtype=np.float64) / 1e9
        releases = np.array([c.release.value for c in s.char_events], dtype=np.float64) / 1e9
        if len(presses) < 2:
            continue
        dwells = releases - presses
        rows.append({
            'participant_id': s.participant_id,
            'password_id': s.password_id,
            'week': s.week,
            'session_num': s.session_num,
            'repetition': s.repetition,
            'total_time': releases[-1] - presses[0],
            'mean_flight': float(np.mean(np.diff(presses))),
            'mean_dwell': float(np.mean(dwells)),
            'zero_dwell_fraction': float(np.mean(dwells <= 0)),
        })
    return pd.DataFrame(rows)


def drop_invalid(df):
    ok = (df['total_time'] > 0) & (df['mean_flight'] > 0)
    dropped = int((~ok).sum())
    if dropped:
        print('dropped %d attempt(s) with a non-positive duration' % dropped)
    return df[ok].copy()


def pair_week_medians(df):
    """Collapse to one robust value per (participant, password, week) before
    any averaging, so a participant with more surviving sessions does not
    weigh more heavily within a week."""
    return (df.groupby(['participant_id', 'password_id', 'week'])[METRICS]
            .median().reset_index())


def normalise_to_week_one(pw):
    base = (pw[pw['week'] == 1]
            .set_index(['participant_id', 'password_id'])[METRICS]
            .rename(columns={m: m + '_base' for m in METRICS}))
    out = pw.merge(base, on=['participant_id', 'password_id'], how='inner')
    for m in METRICS:
        out[m + '_rel'] = out[m] / out[m + '_base']
    return out


def weekly_profile(rel):
    rows = []
    for week, g in rel.groupby('week'):
        row = {'week': int(week), 'n_pairs': len(g)}
        for m in METRICS:
            v = g[m + '_rel'].to_numpy()
            row[m + '_median'] = float(np.median(v))
            row[m + '_q1'] = float(np.percentile(v, 25))
            row[m + '_q3'] = float(np.percentile(v, 75))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values('week').reset_index(drop=True)

    # how much of the whole eight-week change has happened by each week
    start = out.loc[out['week'] == 1, 'total_time_median'].iloc[0]
    end = out.loc[out['week'] == 8, 'total_time_median'].iloc[0]
    total_change = end - start
    out['pct_of_total_change'] = 100 * (out['total_time_median'] - start) / total_change
    return out


def week_on_week_tests(rel):
    """Consecutive weeks compared pairwise on (participant, password), so each
    pair is its own control. Wilcoxon signed-rank rather than a t-test, since
    the per-pair ratios are skewed."""
    rows = []
    for w in range(1, 8):
        a = rel[rel['week'] == w].set_index(['participant_id', 'password_id'])['total_time_rel']
        b = rel[rel['week'] == w + 1].set_index(['participant_id', 'password_id'])['total_time_rel']
        common = a.index.intersection(b.index)
        if len(common) < 10:
            continue
        diff = b.loc[common].to_numpy() - a.loc[common].to_numpy()
        stat, p = sstats.wilcoxon(b.loc[common], a.loc[common])
        rows.append({
            'from_week': w, 'to_week': w + 1, 'n_pairs': len(common),
            'median_change_pct': float(np.median(diff) * 100),
            'wilcoxon_stat': float(stat), 'p': float(p),
            'faster': bool(np.median(diff) < 0),
        })
    return pd.DataFrame(rows)


def repetition_profile(df):
    """Within-session repetition index, 1 to 4. Normalised per (participant,
    password, week) so the comparison is within a sitting."""
    base = (df.groupby(['participant_id', 'password_id', 'week'])['total_time']
            .median().rename('sitting_median').reset_index())
    merged = df.merge(base, on=['participant_id', 'password_id', 'week'])
    merged['rel'] = merged['total_time'] / merged['sitting_median']
    overall = merged.groupby('repetition')['rel'].median().reset_index()
    by_week = merged.groupby(['week', 'repetition'])['rel'].median().unstack()
    return overall, by_week


def consistency_profile(df):
    """Robust coefficient of variation within each (participant, password,
    week): median absolute deviation over the median. A matcher cares about
    spread, not only level."""
    def rcv(x):
        med = np.median(x)
        if med <= 0:
            return np.nan
        return float(np.median(np.abs(x - med)) / med)

    cv = (df.groupby(['participant_id', 'password_id', 'week'])['total_time']
          .apply(rcv).rename('rcv').reset_index())
    return cv.groupby('week')['rcv'].median().reset_index()


def dwell_quality(df):
    """Dwell is degenerate in this collection. Reported as data quality."""
    g = df.groupby('week').agg(
        median_dwell_s=('mean_dwell', 'median'),
        mean_zero_dwell_fraction=('zero_dwell_fraction', 'mean'),
        attempts=('mean_dwell', 'size'),
    ).reset_index()
    return g


def plot(weekly, reps_by_week, consistency, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.fill_between(weekly['week'], weekly['total_time_q1'], weekly['total_time_q3'],
                    color='#1f77b4', alpha=0.15)
    ax.plot(weekly['week'], weekly['total_time_median'], marker='o',
            color='#1f77b4', label='total time')
    ax.plot(weekly['week'], weekly['mean_flight_median'], marker='s',
            color='#2ca02c', linestyle='--', label='mean flight')
    ax.axhline(1.0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Week')
    ax.set_ylabel("Median, relative to that pair's week 1")
    ax.set_title('Typing keeps speeding up', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for w in sorted(reps_by_week.index):
        shade = 0.15 + 0.85 * (w - 1) / 7.0
        ax.plot(reps_by_week.columns, reps_by_week.loc[w], marker='o',
                markersize=3, color=plt.cm.viridis(shade), label='week %d' % w)
    ax.axhline(1.0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Repetition within session')
    ax.set_ylabel('Relative to that sitting')
    ax.set_title('Within-session warm-up, by week', fontsize=10)
    ax.set_xticks(sorted(reps_by_week.columns))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[2]
    ax.plot(consistency['week'], consistency['rcv'], marker='o', color='#d62728')
    ax.set_xlabel('Week')
    ax.set_ylabel('Median absolute deviation / median')
    ax.set_title('Consistency within a week', fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print('wrote %s' % out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = load_samples(use_cache=True)
    print('samples: %d' % len(samples))

    raw = attempt_table(samples)
    df = drop_invalid(raw)
    print('attempts used: %d' % len(df))

    pw = pair_week_medians(df)
    rel = normalise_to_week_one(pw)
    print('(participant, password, week) cells with a week-1 baseline: %d' % len(rel))
    print()

    weekly = weekly_profile(rel)
    tests = week_on_week_tests(rel)
    reps_overall, reps_by_week = repetition_profile(df)
    consistency = consistency_profile(df)
    dwell = dwell_quality(raw)

    print('=' * 76)
    print("RELATIVE TIMING BY WEEK (1.0 = that pair's own week 1, medians)")
    print('=' * 76)
    print(weekly.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print()

    print('=' * 76)
    print('PAIRED WEEK-ON-WEEK CHANGE IN TOTAL TYPING TIME (Wilcoxon)')
    print('=' * 76)
    print(tests.to_string(index=False, float_format=lambda v: '%.4g' % v))
    print()

    print('=' * 76)
    print('WITHIN-SESSION REPETITION EFFECT')
    print('=' * 76)
    print(reps_overall.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print()

    print('=' * 76)
    print('CONSISTENCY (median absolute deviation over median, within week)')
    print('=' * 76)
    print(consistency.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print()

    print('=' * 76)
    print('DWELL DATA QUALITY (not a learning curve)')
    print('=' * 76)
    print(dwell.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print()

    weekly.to_csv(os.path.join(OUT_DIR, 'familiarisation_weekly.csv'), index=False)
    tests.to_csv(os.path.join(OUT_DIR, 'familiarisation_week_tests.csv'), index=False)
    reps_overall.to_csv(os.path.join(OUT_DIR, 'familiarisation_repetition.csv'), index=False)
    consistency.to_csv(os.path.join(OUT_DIR, 'familiarisation_consistency.csv'), index=False)
    dwell.to_csv(os.path.join(OUT_DIR, 'dwell_quality.csv'), index=False)
    plot(weekly, reps_by_week, consistency, os.path.join(FIG_DIR, 'familiarisation.png'))
    print('wrote csv outputs to %s' % os.path.abspath(OUT_DIR))


if __name__ == '__main__':
    main()
