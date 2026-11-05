"""
Revision experiment for TBIOM reviewer 2, point 7.

The main analysis recalibrates the decision threshold separately for each
temporal gap (Section 3.8). That isolates how separable genuine and impostor
scores are at each gap, but it is not how a deployment behaves. A deployment
fixes its threshold once, at enrolment, and then lives with it.

This script freezes the threshold at Delta=0 and applies it unchanged to every
later gap, so the operating-point drift and the score-distribution shift are
allowed to compound the way they would in service.

Thresholds are derived on the validation participants, matching the
calibration convention of the main analysis, and applied to the held-out test
participants. Three operating points are reported: the EER point, and FMR
budgets of 1% and 0.1% (ISO/IEC 19795-1).

Scores follow the "higher = more genuine" convention, so a trial is accepted
when score >= tau.

Outputs results/revision/fixed_threshold.csv, one row per
(mechanism, fold, operating point, delta), carrying both the frozen-threshold
rates and the per-gap recalibrated rates for contrast.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_curve

FMR_BUDGETS = [0.01, 0.001]
TRIAL_COLUMNS = ['fold', 'mechanism', 'participant_id', 'delta', 'label', 'score']


def _result_dir():
    base = os.environ.get('KD_RESULT_DIR', os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
    out = os.path.join(base, 'revision')
    os.makedirs(out, exist_ok=True)
    return out


RESULTS_ROOT = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
ITAD_DIR = os.path.join(RESULTS_ROOT, 'with_itad')


def load_trials(path, k_enrollment, clipped, itad_name=None):
    """Read one configuration out of a trial file. The files hold every
    (k_enrollment, clipped) combination, so filtering at read time keeps the
    working set to roughly a sixth of the 45M rows.

    ITAD lives in its own files and carries the clipping label
    'not_applicable', because a percentile within the enrolment profile
    cannot be changed by clipping. It is merged in here so the frozen
    threshold is evaluated over all five mechanisms."""
    table = pq.read_table(
        path,
        columns=TRIAL_COLUMNS,
        filters=[('k_enrollment', '=', k_enrollment), ('clipped', '=', clipped)],
    )
    df = table.to_pandas()

    if itad_name:
        itad_path = os.path.join(ITAD_DIR, itad_name)
        if os.path.exists(itad_path):
            extra = pq.read_table(
                itad_path, columns=TRIAL_COLUMNS,
                filters=[('k_enrollment', '=', k_enrollment)]).to_pandas()
            df = pd.concat([df, extra], ignore_index=True)
        else:
            print('note: %s not found, ITAD omitted' % itad_path)
    return df


def threshold_at_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thresholds[idx])


def threshold_at_fmr(labels, scores, budget):
    """The lowest threshold whose false-match rate still sits inside the
    budget. roc_curve returns thresholds in decreasing order with fpr
    non-decreasing, so the admissible thresholds form a prefix."""
    fpr, _, thresholds = roc_curve(labels, scores)
    admissible = np.where(fpr <= budget)[0]
    if len(admissible) == 0:
        return float(thresholds[0])
    return float(thresholds[admissible[-1]])


def calibrate(labels, scores, operating_point):
    if operating_point == 'eer':
        return threshold_at_eer(labels, scores)
    return threshold_at_fmr(labels, scores, float(operating_point))


def rates_at(group, tau):
    """FMR, FNMR and the half-total error rate for one set of trials scored
    against a single threshold."""
    genuine = group['score'][group['label'] == 1].to_numpy()
    impostor = group['score'][group['label'] == 0].to_numpy()
    fnmr = float(np.mean(genuine < tau)) if len(genuine) else np.nan
    fmr = float(np.mean(impostor >= tau)) if len(impostor) else np.nan
    return fmr, fnmr, (fmr + fnmr) / 2.0, len(genuine), len(impostor)


def operating_points():
    yield 'eer'
    for budget in FMR_BUDGETS:
        yield str(budget)


def run(val, test):
    rows = []
    for (mechanism, fold), test_cell in test.groupby(['mechanism', 'fold']):
        val_cell = val[(val['mechanism'] == mechanism) & (val['fold'] == fold)]
        val_at_zero = val_cell[val_cell['delta'] == 0]
        if val_at_zero.empty or val_at_zero['label'].nunique() < 2:
            continue

        for point in operating_points():
            tau_frozen = calibrate(val_at_zero['label'], val_at_zero['score'], point)

            for delta, test_delta in test_cell.groupby('delta'):
                fmr, fnmr, hter, n_gen, n_imp = rates_at(test_delta, tau_frozen)

                # the published convention, recalibrated on this gap's own
                # validation trials, kept alongside for the contrast
                val_delta = val_cell[val_cell['delta'] == delta]
                if val_delta.empty or val_delta['label'].nunique() < 2:
                    tau_recal = np.nan
                    fmr_r = fnmr_r = hter_r = np.nan
                else:
                    tau_recal = calibrate(val_delta['label'], val_delta['score'], point)
                    fmr_r, fnmr_r, hter_r, _, _ = rates_at(test_delta, tau_recal)

                rows.append({
                    'mechanism': mechanism, 'fold': fold, 'operating_point': point,
                    'delta': int(delta), 'n_genuine': n_gen, 'n_impostor': n_imp,
                    'tau_frozen': tau_frozen, 'fmr_frozen': fmr,
                    'fnmr_frozen': fnmr, 'hter_frozen': hter,
                    'tau_recalibrated': tau_recal, 'fmr_recalibrated': fmr_r,
                    'fnmr_recalibrated': fnmr_r, 'hter_recalibrated': hter_r,
                })
    return pd.DataFrame(rows)


def summarise(per_fold):
    """Mean and standard deviation across folds, which is the reporting
    convention used for the EER surface in the main results."""
    metrics = ['fmr_frozen', 'fnmr_frozen', 'hter_frozen',
               'fmr_recalibrated', 'fnmr_recalibrated', 'hter_recalibrated']
    grouped = per_fold.groupby(['mechanism', 'operating_point', 'delta'])
    out = grouped[metrics].agg(['mean', 'std'])
    out.columns = ['%s_%s' % (metric, stat) for metric, stat in out.columns]
    counts = grouped[['n_genuine', 'n_impostor']].sum()
    return out.join(counts).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=RESULTS_ROOT)
    parser.add_argument('--k-enrollment', type=int, default=5)
    parser.add_argument('--clipped', default='clipped', choices=['clipped', 'unclipped'])
    args = parser.parse_args()

    val_path = os.path.join(args.results_dir, 'trials_val.parquet')
    test_path = os.path.join(args.results_dir, 'trials_test.parquet')

    print('loading validation trials')
    val = load_trials(val_path, args.k_enrollment, args.clipped,
                      itad_name='itad_trials_val.parquet')
    print('loading test trials')
    test = load_trials(test_path, args.k_enrollment, args.clipped,
                       itad_name='itad_trials_test.parquet')
    print('validation rows: %d   test rows: %d' % (len(val), len(test)))

    per_fold = run(val, test)
    summary = summarise(per_fold)

    out_dir = _result_dir()
    per_fold_path = os.path.join(out_dir, 'fixed_threshold_per_fold.csv')
    summary_path = os.path.join(out_dir, 'fixed_threshold.csv')
    per_fold.to_csv(per_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    print('wrote %s' % per_fold_path)
    print('wrote %s' % summary_path)

    eer_point = summary[summary['operating_point'] == 'eer']
    print()
    print('Half-total error rate at the EER operating point,')
    print('threshold frozen at enrolment versus recalibrated per gap:')
    print()
    pivot = eer_point.pivot(index='delta', columns='mechanism',
                            values=['hter_frozen_mean', 'hter_recalibrated_mean'])
    print(pivot.round(4).to_string())


if __name__ == '__main__':
    main()
