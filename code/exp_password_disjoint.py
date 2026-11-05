"""
Revision experiment for TBIOM reviewer 2, point 3.

All 40 passwords appear in every training, validation and test split of the
main study. Participants are disjoint, text is not. The reviewer asks how we
know the embedding models learn typing behaviour rather than patterns
specific to the 40 strings they were trained on.

This script answers it by withholding passwords from training as well as
participants. The 40 passwords are split into two halves, stratified by
length and substitution category. M3 and M4 are trained on training
participants typing the training half only, and then evaluated on test
participants typing both halves.

The contrast that matters is within one trained model: seen passwords
against unseen passwords, with unseen participants in both cases. The model,
its weights and its training budget are identical across the two, so any gap
is attributable to password novelty alone. Comparing instead against the
published full-corpus numbers would confound novelty with a halved training
set.

M1 is included as a control. It fits only a population variability estimate
and has no capacity to memorise a password, so whatever seen-to-unseen gap
it shows is a property of the password sets rather than of learning.

M2 is excluded, and that exclusion is itself a result. M2 trains one
classifier per password, so for a password never seen in training there is
no classifier to apply. A per-password mechanism cannot be deployed on a new
password without new training data.

Clipping. SequenceClipper, which M3 and M4 use, is global and password
agnostic, so the clipped condition transfers to an unseen password unchanged
and is used here. IQRClipper, which M1 uses, holds per-password bounds and
silently passes a vector through untouched when the password is unknown.
That would clip seen passwords and not unseen ones. M1 is therefore run
unclipped, so both halves are treated identically.

Writes results/revision/password_disjoint_*.csv.
"""
import argparse
import dataclasses
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import load_samples
from dataset import Dataset
from splits import fold_splits
from embedding_models import (EmbeddingConfig, SequenceClipper, train_embedding_model,
                              EmbeddingMatcher, EnsembleEmbeddingMatcher)
from matchers_classical import M1Manhattan
from protocol import generate_trials
from metrics import compute_eer, compute_fnmr_at_fmr
from password_metadata import build_password_metadata

OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'revision')


class NoOpClipper:
    def transform(self, vec, password_id=None):
        return vec


class PasswordRestrictedView:
    """A read-only view of a Dataset exposing only a subset of passwords.

    Sample indices are the full dataset's, so a model trained through this
    view can still score any sample afterwards. Training only ever reads
    password_ids, by_password, samples and raw_sequences, so overriding the
    first two is enough to keep held-out passwords out of the triplet pool.
    """

    def __init__(self, base, password_ids):
        keep = set(password_ids)
        self._base = base
        self.password_ids = sorted(keep)
        self.by_password = {pw: base.by_password[pw] for pw in self.password_ids}

    def __getattr__(self, name):
        return getattr(self._base, name)


def split_passwords(meta_rows, seed=0):
    """Halve the corpus, stratified by (length, substitution category), so
    both halves span the same difficulty range."""
    strata = defaultdict(list)
    for r in meta_rows:
        strata[(r['length'], r['substitution_category'])].append(r['password_id'])

    train_pws, held_pws = [], []
    for i, key in enumerate(sorted(strata)):
        members = sorted(strata[key])
        for j, pw in enumerate(members):
            # alternate within a stratum, and alternate the starting side
            # between strata so odd-sized strata do not all fall one way
            (train_pws if (i + j) % 2 == 0 else held_pws).append(pw)
    return sorted(train_pws), sorted(held_pws)


def evaluate(trials, label, mechanism, fold):
    df = pd.DataFrame(trials)
    rows = []
    for delta, g in df.groupby('delta'):
        eer, _ = compute_eer(g['label'], g['score'])
        rows.append({
            'mechanism': mechanism, 'fold': fold, 'password_set': label,
            'delta': int(delta), 'n_trials': len(g),
            'n_genuine': int((g['label'] == 1).sum()),
            'eer': eer,
            'fnmr_at_fmr_0.01': compute_fnmr_at_fmr(g['label'], g['score'], 0.01),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--k-enrollment', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tag', default='')
    parser.add_argument('--swap', action='store_true',
                        help='exchange the two password halves, so the half '
                             'held out in the primary run becomes the training '
                             'half. Checks that the answer does not depend on '
                             'which draw was withheld.')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    meta_rows = build_password_metadata()
    password_meta = {r['password_id']: r for r in meta_rows}

    train_pws, held_pws = split_passwords(meta_rows)
    if args.swap:
        train_pws, held_pws = held_pws, train_pws
        print('SWAPPED: training on the half withheld in the primary run')
    meta_seen = {pw: password_meta[pw] for pw in train_pws}
    meta_unseen = {pw: password_meta[pw] for pw in held_pws}
    print('passwords in training half: %d   held out: %d' % (len(train_pws), len(held_pws)))
    print('held out: %s' % ', '.join(held_pws))
    print('device: %s   seeds: %d   folds: %s' % (args.device, args.seeds, args.folds))
    print()

    all_rows = []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids, seed=0):
        if fold not in args.folds:
            continue
        t_fold = time.time()
        view = PasswordRestrictedView(dataset, train_pws)

        # fit the global sequence clipper on training participants typing
        # training passwords only, so nothing about the held-out half leaks
        train_set = set(train_ids)
        train_idxs = [i for pw in train_pws for i in dataset.by_password[pw]
                      if dataset.samples[i].participant_id in train_set]
        seq_clipper = SequenceClipper().fit([dataset.raw_sequences[i][:, 0] for i in train_idxs])

        m3_members, m4_members = [], []
        for s in range(args.seeds):
            cfg = dataclasses.replace(EmbeddingConfig(), seed=s)
            for arch, members, name in [('lstm', m3_members, 'M3'),
                                        ('transformer', m4_members, 'M4')]:
                t0 = time.time()
                model = train_embedding_model(arch, view, train_ids, val_ids,
                                              seq_clipper, cfg,
                                              device=args.device, verbose=False)
                matcher = EmbeddingMatcher('%s_seed%d' % (name, s), model,
                                           seq_clipper, device=args.device)
                matcher.precompute_all(dataset)
                members.append(matcher)
                print('  fold %d %s seed %d trained in %.0fs'
                      % (fold, name, s, time.time() - t0), flush=True)

        matchers = {
            'M3_TypeNet': (EnsembleEmbeddingMatcher('M3_TypeNet', m3_members), None),
            'M4_TypeFormer': (EnsembleEmbeddingMatcher('M4_TypeFormer', m4_members), None),
        }

        # M1 control, unclipped so seen and unseen are handled identically
        m1 = M1Manhattan()
        m1.fit(dataset, train_ids, NoOpClipper())
        matchers['M1_Manhattan_unclipped'] = (m1, NoOpClipper())

        for mech, (matcher, clipper) in matchers.items():
            for label, meta in [('seen', meta_seen), ('unseen', meta_unseen)]:
                trials = generate_trials(
                    dataset, matcher, clipper, test_ids, meta, mech, fold,
                    'clipped' if mech.startswith(('M3', 'M4')) else 'unclipped',
                    k_enrollment=args.k_enrollment,
                    max_impostors_per_trial=None, seed=0)
                all_rows.extend(evaluate(trials, label, mech, fold))
            print('  fold %d %s scored' % (fold, mech), flush=True)

        print('fold %d done in %.1f min' % (fold, (time.time() - t_fold) / 60), flush=True)
        print()

    out = pd.DataFrame(all_rows)
    suffix = ('_' + args.tag) if args.tag else ''
    per_fold_path = os.path.join(OUT_DIR, 'password_disjoint_per_fold%s.csv' % suffix)
    out.to_csv(per_fold_path, index=False)

    summary = (out.groupby(['mechanism', 'password_set', 'delta'])
               .agg(eer_mean=('eer', 'mean'), eer_sd=('eer', 'std'),
                    n_folds=('eer', 'size'), n_trials=('n_trials', 'sum'))
               .reset_index())
    summary_path = os.path.join(OUT_DIR, 'password_disjoint%s.csv' % suffix)
    summary.to_csv(summary_path, index=False)

    print('=' * 72)
    print('EER BY PASSWORD SET (mean across folds)')
    print('=' * 72)
    pivot = summary.pivot_table(index='delta', columns=['mechanism', 'password_set'],
                                values='eer_mean')
    print((pivot * 100).to_string(float_format=lambda v: '%.2f' % v))
    print()
    print('seen-to-unseen gap, averaged over gaps:')
    for mech, g in summary.groupby('mechanism'):
        seen = g[g['password_set'] == 'seen']['eer_mean'].mean()
        unseen = g[g['password_set'] == 'unseen']['eer_mean'].mean()
        print('  %-24s seen %.4f  unseen %.4f  gap %+.4f'
              % (mech, seen, unseen, unseen - seen))
    print()
    print('wrote %s' % per_fold_path)
    print('wrote %s' % summary_path)
    print('total %.1f min' % ((time.time() - t_start) / 60))


if __name__ == '__main__':
    main()
