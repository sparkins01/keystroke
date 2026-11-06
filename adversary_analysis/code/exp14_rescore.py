"""
Experiment 14 (R1 revision): two new sets of scores, for all four mechanisms.

1. Within-session human retries (reviewer 2, comment 3).
   R0 modelled a human's three attempts as their samples from weeks 6, 7 and 8:
   one attempt per week, at three different levels of practice. A lockout
   policy counts consecutive attempts in one sitting. Here the three attempts
   are the impostor's first three repetitions within a single week-8 session.
   Position 0 is the sample R0 already scored as A1, so re-scoring it is a
   built-in check that this script reproduces the original pipeline.

2. A leak that precedes the attack (reviewer 1, comment 2).
   R0's two horizons were not causal. The full horizon uses all eight weeks;
   the early horizon uses weeks 1 and 2 for every gallery, so a gallery enrolled
   in week 1 is attacked with typing from week 2. Here the victim profile is
   rebuilt per enrolment week i from typing the adversary could have observed
   by then:
     A4_keyed_sampled_trailing   weeks {i-1, i}: a two-week leak ending at
                                 enrolment (one week when i = 1)
     A4_keyed_sampled_upto       weeks 1..i: everything observed so far
     A4_keyed_sampled_prior      weeks {i-2, i-1}: a two-week leak that ends
                                 before the enrolment week begins (one week
                                 when i = 2, none when i = 1)
   The first two include the enrolment week itself, so at delta = 0 the leak
   is contemporaneous with the attack. The third is strictly earlier than any
   attack at query week j >= i. The target
   password is still excluded, and that exclusion is still asserted.

Galleries, clippers, fold splits and cached encoders are exactly those of
experiments 1 to 4, so the scores are directly comparable with R0's.

Usage:
    python exp14_rescore.py                     # all mechanisms, all folds
    python exp14_rescore.py --mechanisms M1 --folds 0
"""
import argparse
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch

from dataset_cache import load_dataset
from features import IQRClipper
from splits import fold_splits
from password_metadata import build_password_metadata
from embedding_models import (EmbeddingMatcher, EnsembleEmbeddingMatcher,
                              SequenceClipper, LSTMEncoder, TransformerEncoder,
                              EmbeddingConfig)
from protocol_adversary import ordered_sessions
from exp2_transfer import (canonical_keys, per_password_tables, population_profile,
                           victim_profile, forge, to_raw_sequence, to_feature_vector,
                           score_vector, N_STOCHASTIC, RNG_SEED)
from exp3_m4 import embed_sequences, score_against_gallery, MODEL_DIR
from exp4_ladder import score_vectors_m2

from matchers_classical import M1Manhattan, M2ShallowML

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(RESULT_DIR, 'exp14_rescore')
LOG = os.path.join(RESULT_DIR, 'exp14_rescore.log')
K_ENROLLMENT = 5
N_SEEDS = 3
RETRY_WEEK = 8
N_SESSION_ATTEMPTS = 3
MECH_NAMES = {'M1': 'M1_Manhattan', 'M2': 'M2_ShallowML',
              'M3': 'M3_TypeNet', 'M4': 'M4_TypeFormer'}


def log(msg):
    line = f'{time.strftime("%H:%M:%S")}  {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def session_attempts(dataset, qid, pw_id, week=RETRY_WEEK, n=N_SESSION_ATTEMPTS):
    """The first n samples of this impostor's first session in `week`, in the
    order typed. None if that session holds fewer than n samples."""
    idxs = [i for i in ordered_sessions(dataset, qid, pw_id)
            if dataset.samples[i].week == week]
    if not idxs:
        return None
    first_session = dataset.samples[idxs[0]].session_num
    same = [i for i in idxs if dataset.samples[i].session_num == first_session]
    return same[:n] if len(same) >= n else None


def window_rng(pid, pw_id, horizon, week):
    """Seeded per (victim, password, horizon, enrolment week), so every
    mechanism scores identical forgeries regardless of iteration order."""
    h = zlib.crc32(f'{pid}|{pw_id}|{horizon}|{week}'.encode())
    return np.random.RandomState((RNG_SEED * 1000003 + h) % (2 ** 32))


def build_matcher(mech_key, dataset, fold, train_ids, device):
    train_set = set(train_ids)
    if mech_key in ('M1', 'M2'):
        feats_by_pw = {pw: [dataset.feature_vectors[i] for i in dataset.by_password[pw]
                            if dataset.samples[i].participant_id in train_set]
                       for pw in dataset.password_ids}
        clipper = IQRClipper().fit(feats_by_pw)
        matcher = M1Manhattan() if mech_key == 'M1' else M2ShallowML(seed=0)
        matcher.fit(dataset, train_ids, clipper)
        return matcher, clipper

    durations = [dataset.raw_sequences[i][:, 0] for i in range(len(dataset.samples))
                 if dataset.samples[i].participant_id in train_set]
    clipper = SequenceClipper().fit(durations)
    arch = LSTMEncoder if mech_key == 'M3' else TransformerEncoder
    prefix = 'm3' if mech_key == 'M3' else 'm4'
    members = []
    for seed in range(N_SEEDS):
        path = os.path.join(MODEL_DIR, f'{prefix}_fold{fold}_seed{seed}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'cached encoder missing: {path}')
        cfg = EmbeddingConfig()
        cfg.seed = seed
        model = arch(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        m = EmbeddingMatcher(f'{MECH_NAMES[mech_key]}_seed{seed}', model, clipper, device=device)
        m.precompute_all(dataset)
        members.append(m)
    return EnsembleEmbeddingMatcher(MECH_NAMES[mech_key], members), clipper


def score_forgeries(mech_key, matcher, clipper, gallery, keys, timings, pw_id,
                    device, emb_cache):
    if mech_key == 'M1':
        return [score_vector(gallery, clipper, to_feature_vector(keys, d, iv), pw_id)
                for d, iv in timings]
    if mech_key == 'M2':
        vecs = [to_feature_vector(keys, d, iv) for d, iv in timings]
        return score_vectors_m2(matcher, gallery, clipper, vecs, pw_id)
    seqs = [to_raw_sequence(keys, d, iv) for d, iv in timings]
    per_model = [embed_sequences(m.model, m.clipper, seqs, device) for m in matcher.matchers]
    return list(score_against_gallery(gallery, per_model))


def run_fold(mech_key, dataset, fold, train_ids, test_ids, password_meta, keys_by_pw, device):
    mech = MECH_NAMES[mech_key]
    matcher, clipper = build_matcher(mech_key, dataset, fold, train_ids, device)
    pop = {pw: population_profile(dataset, train_ids, pw) for pw in dataset.password_ids}

    windows = {}
    for i in range(1, 9):
        windows[('trailing', i)] = per_password_tables(dataset, test_ids, weeks={i - 1, i})
        windows[('upto', i)] = per_password_tables(dataset, test_ids, weeks=set(range(1, i + 1)))
        windows[('prior', i)] = per_password_tables(dataset, test_ids, weeks={i - 2, i - 1})

    attempts = {(q, pw): session_attempts(dataset, q, pw)
                for q in test_ids for pw in dataset.password_ids}

    imp_rows, forg_rows = [], []
    for pid in test_ids:
        for pw_id in dataset.password_ids:
            if pw_id not in password_meta:
                continue
            keys = keys_by_pw[pw_id]
            sessions_by_week = {w: dataset.sessions_at_week(pid, pw_id, w) for w in range(1, 9)}
            for i in sorted(w for w, v in sessions_by_week.items() if v):
                pool_idx = dataset.sessions_for(pid, pw_id, max_week=i)
                if not pool_idx:
                    continue
                same_week = set(sessions_by_week[i])
                same_in_pool = [x for x in pool_idx if x in same_week]
                reserved = same_in_pool[-1] if len(same_in_pool) >= 2 else None
                cand = [x for x in pool_idx if x != reserved]
                if not cand:
                    continue
                gallery = matcher.gallery(dataset, clipper, cand[-K_ENROLLMENT:])

                # ---- within-session human attempts
                q_idx, q_meta = [], []
                for qid in test_ids:
                    if qid == pid:
                        continue
                    att = attempts.get((qid, pw_id))
                    if not att:
                        continue
                    for pos, idx in enumerate(att):
                        q_idx.append(idx)
                        q_meta.append((qid, pos))
                if q_idx:
                    scores = matcher.score_many(dataset, clipper, gallery, q_idx)
                    for (qid, pos), sc in zip(q_meta, scores):
                        imp_rows.append((fold, pid, pw_id, i, qid, RETRY_WEEK, pos, float(sc)))

                # ---- forgeries from a leak that precedes the attack
                if pop.get(pw_id) is None or len(keys) < 3:
                    continue
                for horizon in ('trailing', 'upto', 'prior'):
                    tab = windows[(horizon, i)].get(pid)
                    if not tab:
                        continue
                    vp = victim_profile(tab, exclude_pw=pw_id)
                    assert pw_id not in vp['used_passwords'], \
                        f'leak: {pw_id} contributed to its own forgery for {pid}'
                    if not np.isfinite(vp['global_dwell']):
                        continue
                    rng = window_rng(pid, pw_id, horizon, i)
                    timings = [forge(keys, pop[pw_id], vp, 'A4_keyed', rng=rng)
                               for _ in range(N_STOCHASTIC)]
                    scores = score_forgeries(mech_key, matcher, clipper, gallery, keys,
                                             timings, pw_id, device, None)
                    for vi, sc in enumerate(scores):
                        forg_rows.append((fold, pid, pw_id, i,
                                          f'A4_keyed_sampled_{horizon}', vi, float(sc)))

    imp = pd.DataFrame(imp_rows, columns=['fold', 'participant_id', 'password_id', 'enrol_week',
                                          'imp_participant_id', 'imp_week', 'attempt', 'score'])
    forg = pd.DataFrame(forg_rows, columns=['fold', 'participant_id', 'password_id',
                                            'enrol_week', 'attack', 'variant', 'score'])
    imp['mechanism'] = mech
    forg['mechanism'] = mech
    return imp, forg


def main(mechanisms, folds_wanted):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset()
    password_meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}
    log(f'device={device}, mechanisms={mechanisms}')

    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        if folds_wanted is not None and fold not in folds_wanted:
            continue
        assert not (set(train_ids) & set(test_ids)), 'train/test overlap'
        for mech_key in mechanisms:
            mech = MECH_NAMES[mech_key]
            ip = os.path.join(OUT_DIR, f'session_{mech}_fold{fold}.parquet')
            fp = os.path.join(OUT_DIR, f'causal_{mech}_fold{fold}.parquet')
            if os.path.exists(ip) and os.path.exists(fp):
                log(f'fold {fold} {mech}: already complete, skipping')
                continue
            t0 = time.time()
            imp, forg = run_fold(mech_key, dataset, fold, train_ids, test_ids,
                                 password_meta, keys_by_pw, device)
            imp.to_parquet(ip, index=False)
            forg.to_parquet(fp, index=False)
            log(f'fold {fold} {mech}: {len(imp):,} session scores, {len(forg):,} forgery scores '
                f'[{(time.time() - t0) / 60:.1f} min]')
    log('run complete')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mechanisms', nargs='+', default=['M1', 'M2', 'M3', 'M4'],
                    choices=sorted(MECH_NAMES))
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    args = ap.parse_args()
    main(args.mechanisms, set(args.folds) if args.folds else None)
