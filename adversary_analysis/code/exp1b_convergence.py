"""
Experiment 1b: why does adversary practice help?

Experiment 1 showed a practised impostor scores closer to the victim's template
than an unpractised one. Two mechanisms produce that, and they imply different
papers:

  (a) Targeted convergence. The attacker specifically drifts toward this
      victim. Unlikely without feedback, but it is what the phrase "attacker
      practice" implies.

  (b) Population convergence. Everybody becomes fluent on a password they have
      typed 30 times, fluent typing is less idiosyncratic than halting typing,
      and so the whole population contracts toward a common style. Nobody is
      targeting anyone; the biometric simply carries less identifying
      information as familiarity grows.

(b) is the more serious finding, because it means fixed-text keystroke
distinctiveness decays for structural reasons rather than because of an
adversary, and it would decay in any deployment where users type their password
regularly.

The test: an F-ratio of between-participant to within-participant variance per
password per week, on the same clipped feature vectors the matcher uses. Falling
discriminability over weeks supports (b).
"""
import os

import numpy as np
import pandas as pd

import data_loading
from dataset import Dataset
from features import IQRClipper

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
MIN_PARTICIPANTS = 10
MIN_REPS = 2


def discriminability_by_week():
    samples = data_loading.load_samples(verbose=False)
    dataset = Dataset(samples)

    # one clipper fitted on everything; this is a descriptive measure of the
    # data rather than a matcher, so there is no train/test split to protect
    feats_by_pw = {pw: [dataset.feature_vectors[i] for i in dataset.by_password[pw]]
                   for pw in dataset.password_ids}
    clipper = IQRClipper().fit(feats_by_pw)

    rows = []
    for pw in dataset.password_ids:
        for week in range(1, 9):
            by_part = {}
            for i in dataset.by_password[pw]:
                s = dataset.samples[i]
                if s.week != week:
                    continue
                by_part.setdefault(s.participant_id, []).append(
                    clipper.transform(dataset.feature_vectors[i], pw))

            usable = {p: np.vstack(v) for p, v in by_part.items() if len(v) >= MIN_REPS}
            if len(usable) < MIN_PARTICIPANTS:
                continue

            means = np.vstack([v.mean(axis=0) for v in usable.values()])
            # within: mean across participants of each one's own per-feature variance
            within = np.mean([v.var(axis=0, ddof=1) for v in usable.values()], axis=0)
            between = means.var(axis=0, ddof=1)

            ok = within > 0
            if not ok.any():
                continue
            f_ratio = float(np.mean(between[ok] / within[ok]))

            rows.append({
                'password_id': pw, 'week': week,
                'n_participants': len(usable),
                'f_ratio': f_ratio,
                'between_var': float(between[ok].mean()),
                'within_var': float(within[ok].mean()),
            })
    return pd.DataFrame(rows)


def main():
    df = discriminability_by_week()
    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULT_DIR, 'exp1b_discriminability.csv'), index=False)

    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 78)
    emit('EXPERIMENT 1b: does the population converge as it becomes fluent?')
    emit('=' * 78)
    emit(f'{df["password_id"].nunique()} passwords x {df["week"].nunique()} weeks')
    emit()

    by_week = df.groupby('week').agg(
        f_ratio=('f_ratio', 'mean'),
        between_var=('between_var', 'mean'),
        within_var=('within_var', 'mean'),
        n=('f_ratio', 'size'),
    ).reset_index()
    emit('Mean across passwords, by week:')
    emit(by_week.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    emit()

    # paired within password, so a change is not driven by which passwords
    # happen to have data in a given week
    wide = df.pivot(index='password_id', columns='week', values='f_ratio').dropna()
    if 1 in wide.columns and 8 in wide.columns:
        d = wide[8] - wide[1]
        rng = np.random.RandomState(0)
        boot = [rng.choice(d.values, len(d), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        emit(f'paired change in F-ratio, week 1 -> week 8, over {len(d)} passwords:')
        emit(f'  mean {d.mean():+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]')
        emit(f'  passwords that became LESS discriminative: {int((d < 0).sum())}/{len(d)}')
        verdict = ('population convergence: the biometric carries less identifying '
                   'information as familiarity grows'
                   if d.mean() < 0 and hi < 0 else
                   'no reliable loss of discriminability over the eight weeks')
        emit(f'  => {verdict}')

    with open(os.path.join(RESULT_DIR, 'exp1b_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
