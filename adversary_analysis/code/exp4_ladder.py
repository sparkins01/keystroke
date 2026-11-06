"""
Experiment 4: put M2 and M3 through the same adversary ladder, so the
accuracy-versus-leakability claim can be tested across all four mechanisms.

Experiment 3 found that M4, the more accurate matcher, lost more marginal
security to a leaked victim profile than M1 did (1.55 bits against 1.02). That
is an interesting claim and currently rests on two points. If the pattern is
monotone in baseline accuracy across all four mechanisms it becomes a real
finding; if M2 and M3 break it, it was a coincidence of two.

M2 is a per-password gradient-boosted classifier over aggregated features, so it
scores a forgery directly as a feature vector. M3 is the TypeNet-style recurrent
embedding and reuses experiment 3's machinery with `architecture='lstm'`.

Every mechanism scores the identical forgeries, since `build_forgeries` returns
dwells and intervals seeded per (victim, password) and each mechanism renders
them into whatever representation it needs.

Usage:
    python exp4_ladder.py --mechanisms M2 M3
    python exp4_ladder.py --mechanisms M3 --folds 0
"""
import argparse
import dataclasses
import os
import time

import numpy as np
import pandas as pd
import torch

from dataset_cache import load_dataset
from features import IQRClipper
from splits import fold_splits
from password_metadata import build_password_metadata
from embedding_models import (EmbeddingConfig, SequenceClipper, EmbeddingMatcher,
                              EnsembleEmbeddingMatcher, train_embedding_model,
                              LSTMEncoder)
from protocol_adversary import generate_trials, trials_to_frame
from exp2_transfer import (canonical_keys, per_password_tables, population_profile,
                           build_forgeries, to_raw_sequence, to_feature_vector,
                           EARLY_WEEKS)
from exp3_m4 import embed_sequences, score_against_gallery, MODEL_DIR

from matchers_classical import M2ShallowML

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
LOG = os.path.join(RESULT_DIR, 'exp4_ladder.log')
N_SEEDS = 3
K_ENROLLMENT = 5
BASE_SEED = 0

MECH_NAMES = {'M2': 'M2_ShallowML', 'M3': 'M3_TypeNet'}


def log(msg):
    line = f'{time.strftime("%H:%M:%S")}  {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def trials_path(mech):
    return os.path.join(RESULT_DIR, f'exp3_trials_{mech}')


def forge_path(mech):
    return os.path.join(RESULT_DIR, f'exp3_forgeries_{mech}')


# ------------------------------------------------------------ M2

def score_vectors_m2(m2, gallery, clipper, vecs, pw_id):
    """M2's genuine-pair probability for feature vectors not in the dataset."""
    V = np.vstack([clipper.transform(v, pw_id) for v in vecs])
    diffs = np.abs(V - gallery)
    model = m2.models.get(pw_id)
    if model is None:
        return (-diffs.sum(axis=1)).tolist()
    return model.predict_proba(diffs)[:, 1].tolist()


# ------------------------------------------------------------ M3

def get_lstm(dataset, fold, seed, train_ids, val_ids, clipper, device):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f'm3_fold{fold}_seed{seed}.pt')
    cfg = dataclasses.replace(EmbeddingConfig(), seed=seed)
    if os.path.exists(path):
        model = LSTMEncoder(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        log(f'  fold {fold} seed {seed}: loaded cached M3 encoder')
        return model
    t0 = time.time()
    model = train_embedding_model('lstm', dataset, train_ids, val_ids,
                                  clipper, cfg, device=device, verbose=False)
    torch.save(model.state_dict(), path)
    log(f'  fold {fold} seed {seed}: M3 trained in {(time.time() - t0) / 60:.1f} min')
    return model


# ------------------------------------------------------------ forgeries

def forgery_rows(dataset, mech, matcher, clipper, fold, train_ids, test_ids,
                 password_meta, keys_by_pw, device):
    pop = {pw: population_profile(dataset, train_ids, pw) for pw in dataset.password_ids}
    horizons = {
        'full': per_password_tables(dataset, test_ids, weeks=None),
        'early': per_password_tables(dataset, test_ids, weeks=set(EARLY_WEEKS)),
    }
    is_embedding = mech.startswith('M3')
    rows = []

    for pid in test_ids:
        for pw_id in dataset.password_ids:
            if pop.get(pw_id) is None or pw_id not in password_meta:
                continue
            keys = keys_by_pw[pw_id]
            if len(keys) < 3:
                continue
            forgeries = build_forgeries(keys, pop[pw_id], horizons, pid, pw_id)
            if not forgeries:
                continue

            flat = [(label, vi, d, iv)
                    for label, variants in forgeries.items()
                    for vi, (d, iv) in enumerate(variants)]

            if is_embedding:
                seqs = [to_raw_sequence(keys, d, iv) for _, _, d, iv in flat]
                per_model = [embed_sequences(m.model, m.clipper, seqs, device)
                             for m in matcher.matchers]
            else:
                vecs = [to_feature_vector(keys, d, iv) for _, _, d, iv in flat]

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
                if is_embedding:
                    scores = score_against_gallery(gallery, per_model)
                else:
                    scores = score_vectors_m2(matcher, gallery, clipper, vecs, pw_id)
                for (label, vi, _, _), sc in zip(flat, scores):
                    rows.append((fold, pid, pw_id, i, label, vi, float(sc)))

    return pd.DataFrame(rows, columns=['fold', 'participant_id', 'password_id',
                                       'enrol_week', 'attack', 'variant', 'score'])


# ------------------------------------------------------------ run

def run(mechanisms, folds_wanted=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset()
    password_meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}
    log(f'device={device}, mechanisms={mechanisms}')

    for mech_key in mechanisms:
        mech = MECH_NAMES[mech_key]
        os.makedirs(trials_path(mech), exist_ok=True)
        os.makedirs(forge_path(mech), exist_ok=True)

    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        if folds_wanted is not None and fold not in folds_wanted:
            continue
        train_set = set(train_ids)
        assert not (train_set & set(test_ids)), 'train/test overlap'

        feats_by_pw = {pw: [dataset.feature_vectors[i] for i in dataset.by_password[pw]
                            if dataset.samples[i].participant_id in train_set]
                       for pw in dataset.password_ids}
        feat_clipper = IQRClipper().fit(feats_by_pw)

        for mech_key in mechanisms:
            mech = MECH_NAMES[mech_key]
            tp = os.path.join(trials_path(mech), f'fold{fold}.parquet')
            fp = os.path.join(forge_path(mech), f'fold{fold}.parquet')
            if os.path.exists(tp) and os.path.exists(fp):
                log(f'fold {fold} {mech}: already complete, skipping')
                continue

            t0 = time.time()
            if mech_key == 'M2':
                matcher = M2ShallowML(seed=BASE_SEED)
                matcher.fit(dataset, train_ids, feat_clipper)
                clipper = feat_clipper
                log(f'  fold {fold}: M2 fitted ({len(matcher.models)} per-password models)')
            else:
                durations = [dataset.raw_sequences[i][:, 0] for i in range(len(dataset.samples))
                             if dataset.samples[i].participant_id in train_set]
                clipper = SequenceClipper().fit(durations)
                members = []
                for s in range(N_SEEDS):
                    seed = BASE_SEED + s
                    model = get_lstm(dataset, fold, seed, train_ids, val_ids, clipper, device)
                    m = EmbeddingMatcher(f'M3_TypeNet_seed{seed}', model, clipper, device=device)
                    m.precompute_all(dataset)
                    members.append(m)
                matcher = EnsembleEmbeddingMatcher('M3_TypeNet', members)
                log(f'  fold {fold}: M3 {N_SEEDS} encoders ready')

            rows = generate_trials(dataset, matcher, clipper, test_ids, password_meta,
                                   mech, fold, 'clipped', k_enrollment=K_ENROLLMENT)
            trials_to_frame(rows).to_parquet(tp, index=False)
            log(f'  fold {fold} {mech}: {len(rows):,} adversary trials')

            forg = forgery_rows(dataset, mech, matcher, clipper, fold, train_ids,
                                test_ids, password_meta, keys_by_pw, device)
            forg.to_parquet(fp, index=False)
            log(f'  fold {fold} {mech}: {len(forg):,} forgery scores '
                f'[{(time.time() - t0) / 60:.1f} min]')

    log('run complete')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mechanisms', nargs='+', default=['M2', 'M3'],
                    choices=sorted(MECH_NAMES))
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    args = ap.parse_args()
    run(args.mechanisms, folds_wanted=set(args.folds) if args.folds else None)
