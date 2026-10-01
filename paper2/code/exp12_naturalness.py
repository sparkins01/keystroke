"""
Experiment 12: evaluate the naturalness check that Section 6.2 proposes.

The draft recommends rejecting samples that are implausibly close to the
template or implausibly consistent, and never tests it. A defence that is
proposed but not measured is worth little, so this implements and evaluates it.

Two detectors, aimed at two different weaknesses of a synthetic sample.

  Centrality. A median forgery sits at the population median on every feature at
  once. Real samples never do. The statistic is the scaled L1 distance from the
  population centroid, and a sample is rejected when that distance is too SMALL.

  Covariance fit. A sampled forgery draws each dwell and interval independently,
  so its marginals are right but the joint structure is wrong. Real typing is
  correlated: a fast repetition is fast throughout. The statistic is the
  Mahalanobis distance under the population covariance, estimated with
  Ledoit-Wolf shrinkage, and a sample is rejected when it is too LARGE.

This also settles the independence question raised against the forgery
generator. If the covariance detector catches the sampled forgeries, then
independent sampling is a real weakness of the attack and the attack numbers are
conservative. If it does not, the attack survives the obvious countermeasure.

Operating points mirror the rest of the paper. The cost to legitimate users is
fixed first, here as the fraction of genuine samples the detector rejects, and
the attack detection rate is then measured at that cost.
"""
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from dataset_cache import load_dataset
from features import IQRClipper
from password_metadata import build_password_metadata
from splits import fold_splits
from exp2_transfer import (canonical_keys, per_password_tables, population_profile,
                           build_forgeries, to_feature_vector, EARLY_WEEKS)

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
BPCER_BUDGETS = [0.01, 0.02, 0.05]     # genuine samples the detector may reject
ATTACKS = ['A2_population', 'A2_sampled',
           'A4_keyed_median_full', 'A4_keyed_sampled_full']


def centrality(X, mu, scale):
    """Scaled L1 distance from the population centroid. Small means suspicious."""
    return (np.abs(X - mu) / scale).sum(axis=1)


def mahalanobis(X, mu, prec):
    d = X - mu
    return np.einsum('ij,jk,ik->i', d, prec, d)


def main():
    dataset = load_dataset()
    meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}

    rows = []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        train_set, test_set = set(train_ids), set(test_ids)
        feats_by_pw = {pw: [dataset.feature_vectors[i] for i in dataset.by_password[pw]
                            if dataset.samples[i].participant_id in train_set]
                       for pw in dataset.password_ids}
        clipper = IQRClipper().fit(feats_by_pw)

        pop = {pw: population_profile(dataset, train_ids, pw) for pw in dataset.password_ids}
        horizons = {
            'full': per_password_tables(dataset, test_ids, weeks=None),
            'early': per_password_tables(dataset, test_ids, weeks=set(EARLY_WEEKS)),
        }

        for pw in dataset.password_ids:
            if pop.get(pw) is None or pw not in meta:
                continue
            keys = keys_by_pw[pw]
            if len(keys) < 3:
                continue

            train_X = np.vstack([clipper.transform(v, pw) for v in feats_by_pw[pw]])
            if len(train_X) < 4 * train_X.shape[1]:
                continue
            mu = train_X.mean(axis=0)
            scale = np.where(train_X.std(axis=0) <= 0, 1.0, train_X.std(axis=0))
            prec = LedoitWolf().fit(train_X).get_precision()

            # genuine test samples
            gen_idx = [i for i in dataset.by_password[pw]
                       if dataset.samples[i].participant_id in test_set]
            if not gen_idx:
                continue
            G = np.vstack([clipper.transform(dataset.feature_vectors[i], pw) for i in gen_idx])
            for c, m in zip(centrality(G, mu, scale), mahalanobis(G, mu, prec)):
                rows.append({'fold': fold, 'password_id': pw, 'kind': 'genuine',
                             'centrality': c, 'mahalanobis': m})

            # forgeries, regenerated with the same per-pair seeds as the attack
            for pid in test_ids:
                forg = build_forgeries(keys, pop[pw], horizons, pid, pw)
                for label, variants in forg.items():
                    if label not in ATTACKS:
                        continue
                    V = np.vstack([clipper.transform(to_feature_vector(keys, d, iv), pw)
                                   for d, iv in variants])
                    for c, m in zip(centrality(V, mu, scale), mahalanobis(V, mu, prec)):
                        rows.append({'fold': fold, 'password_id': pw, 'kind': label,
                                     'centrality': c, 'mahalanobis': m})
        print('  fold %d done (%d rows)' % (fold, len(rows)), flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RESULT_DIR, 'exp12_naturalness.parquet'), index=False)

    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 84)
    emit('EXPERIMENT 12: does the proposed naturalness check work?')
    emit('=' * 84)
    emit('Detection rate at a fixed cost to legitimate users (BPCER).')
    emit('Centrality rejects samples too CLOSE to the population centroid.')
    emit('Covariance rejects samples whose joint structure is implausible.')
    emit()

    gen = df[df['kind'] == 'genuine']
    out = []
    for budget in BPCER_BUDGETS:
        lo = np.percentile(gen['centrality'], budget * 100)
        hi = np.percentile(gen['mahalanobis'], (1 - budget) * 100)
        # both detectors together, each given half the budget
        lo2 = np.percentile(gen['centrality'], budget * 50)
        hi2 = np.percentile(gen['mahalanobis'], (1 - budget / 2) * 100)
        for atk in ATTACKS:
            a = df[df['kind'] == atk]
            if not len(a):
                continue
            out.append({
                'budget': budget, 'attack': atk, 'n': len(a),
                'centrality': float((a['centrality'] < lo).mean()),
                'covariance': float((a['mahalanobis'] > hi).mean()),
                'combined': float(((a['centrality'] < lo2) |
                                   (a['mahalanobis'] > hi2)).mean()),
            })
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(RESULT_DIR, 'exp12_detection.csv'), index=False)

    for budget in BPCER_BUDGETS:
        emit('-' * 84)
        emit(f'cost to legitimate users (BPCER) = {budget:.0%}')
        emit('-' * 84)
        sub = tab[tab['budget'] == budget][['attack', 'n', 'centrality',
                                            'covariance', 'combined']]
        emit(sub.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        emit()

    key = tab[(tab['budget'] == 0.05)].set_index('attack')
    med = key.loc['A4_keyed_median_full'] if 'A4_keyed_median_full' in key.index else None
    samp = key.loc['A4_keyed_sampled_full'] if 'A4_keyed_sampled_full' in key.index else None
    emit('-' * 84)
    emit('Reading')
    emit('-' * 84)
    if med is not None:
        emit(f'median forgeries detected: centrality {med["centrality"]:.1%}, '
             f'combined {med["combined"]:.1%}')
    if samp is not None:
        emit(f'sampled forgeries detected: covariance {samp["covariance"]:.1%}, '
             f'combined {samp["combined"]:.1%}')
        if samp['combined'] > 0.5:
            emit('=> the check is a usable defence against the attack as constructed,')
            emit('   and independent sampling is a real weakness of the generator')
        else:
            emit('=> the check removes the noiseless variant but not the attack itself')

    with open(os.path.join(RESULT_DIR, 'exp12_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
