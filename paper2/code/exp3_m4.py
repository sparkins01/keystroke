"""
Experiment 3 (paper 2): repeat experiments 1 and 2 on M4, the TypeFormer-style
transformer embedding.

Why this matters. Both results so far came from M1, which scores by scaled
distance from a gallery centroid in raw feature space. The noiseless-forgery
artefact found in experiment 2 is a direct consequence of that geometry: an
input with no variability sits closer to any centroid than a real sample does.
M4 also compares against a centroid, but in a learned embedding space reached
through a nonlinear encoder, so there is no guarantee the artefact survives. It
is genuinely possible that M4 rejects the median forgery while still falling to
the keyed one, which would sharpen the paper's claim.

The forgeries are byte-for-byte the ones M1 scored. `build_forgeries` returns
dwells and intervals rather than a rendered representation, and seeds its
generator per (victim, password), so the same synthetic timings are rendered
here as raw sequences instead of aggregated features.

Training cost on this machine is about 23 min per run, so 3 seeds across 5 folds
is roughly 6 hours. Trained encoders are cached to `results/m4_models/`, so an
interrupted run resumes and a re-analysis costs nothing.

Usage:
    python exp3_m4.py --folds 0          # one fold, about 70 min
    python exp3_m4.py                    # all five
    python exp3_m4.py --analyse-only
"""
import argparse
import dataclasses
import os
import time

import numpy as np
import pandas as pd
import torch

from dataset_cache import load_dataset
from splits import fold_splits
from password_metadata import build_password_metadata
from embedding_models import (EmbeddingConfig, SequenceClipper, EmbeddingMatcher,
                              EnsembleEmbeddingMatcher, train_embedding_model,
                              TransformerEncoder, _embed_batch)
from protocol_adversary import generate_trials, trials_to_frame
from exp2_transfer import (canonical_keys, per_password_tables, population_profile,
                           build_forgeries, to_raw_sequence, EARLY_WEEKS)

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
MODEL_DIR = os.path.join(RESULT_DIR, 'm4_models')
MECH = 'M4_TypeFormer'
TRIALS_DIR = os.path.join(RESULT_DIR, f'exp3_trials_{MECH}')
FORGE_DIR = os.path.join(RESULT_DIR, f'exp3_forgeries_{MECH}')
LOG = os.path.join(RESULT_DIR, 'exp3_m4.log')

N_SEEDS = 3
K_ENROLLMENT = 5
BASE_SEED = 0
EMBED_BATCH = 512


def log(msg):
    line = f'{time.strftime("%H:%M:%S")}  {msg}'
    print(line, flush=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ------------------------------------------------------------ training

def get_encoder(dataset, fold, seed, train_ids, val_ids, clipper, device):
    """Train one transformer, or load it if this fold and seed already ran."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f'm4_fold{fold}_seed{seed}.pt')
    cfg = dataclasses.replace(EmbeddingConfig(), seed=seed)

    if os.path.exists(path):
        model = TransformerEncoder(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        log(f'  fold {fold} seed {seed}: loaded cached encoder')
        return model

    t0 = time.time()
    model = train_embedding_model('transformer', dataset, train_ids, val_ids,
                                  clipper, cfg, device=device, verbose=False)
    torch.save(model.state_dict(), path)
    log(f'  fold {fold} seed {seed}: trained in {(time.time() - t0) / 60:.1f} min')
    return model


# ------------------------------------------------------------ forgery scoring

def embed_sequences(model, clipper, seqs, device):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), EMBED_BATCH):
            batch = seqs[start:start + EMBED_BATCH]
            out.append(_embed_batch(model, batch, clipper, device).cpu().numpy())
    return np.vstack(out)


def score_against_gallery(ensemble_gallery, per_model_embeddings):
    """Score-level fusion, matching `EnsembleEmbeddingMatcher.score_many`:
    each member scores the query against its own gallery centroid, and the
    member scores are averaged. Embeddings are never averaged across members,
    since independently trained triplet models share no coordinate system."""
    per_model = [(-np.linalg.norm(E - g, axis=1))
                 for E, g in zip(per_model_embeddings, ensemble_gallery)]
    return np.mean(per_model, axis=0)


def forgery_scores_for_fold(dataset, ensemble, clipper, fold, train_ids, test_ids,
                            password_meta, keys_by_pw, device):
    pop = {pw: population_profile(dataset, train_ids, pw) for pw in dataset.password_ids}
    horizons = {
        'full': per_password_tables(dataset, test_ids, weeks=None),
        'early': per_password_tables(dataset, test_ids, weeks=set(EARLY_WEEKS)),
    }

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

            flat = [(label, vi, to_raw_sequence(keys, d, iv))
                    for label, variants in forgeries.items()
                    for vi, (d, iv) in enumerate(variants)]
            seqs = [s for _, _, s in flat]
            # embed once per member, then reuse across every enrolment week
            per_model = [embed_sequences(m.model, m.clipper, seqs, device)
                         for m in ensemble.matchers]

            sessions_by_week = {w: dataset.sessions_at_week(pid, pw_id, w) for w in range(1, 9)}
            weeks_with_data = sorted(w for w, v in sessions_by_week.items() if v)
            for i in weeks_with_data:
                pool_idx = dataset.sessions_for(pid, pw_id, max_week=i)
                if not pool_idx:
                    continue
                same_week = set(sessions_by_week[i])
                same_in_pool = [x for x in pool_idx if x in same_week]
                reserved = same_in_pool[-1] if len(same_in_pool) >= 2 else None
                cand = [x for x in pool_idx if x != reserved]
                if not cand:
                    continue
                gallery = ensemble.gallery(dataset, clipper, cand[-K_ENROLLMENT:])
                scores = score_against_gallery(gallery, per_model)
                for (label, vi, _), sc in zip(flat, scores):
                    rows.append((fold, pid, pw_id, i, label, vi, float(sc)))

    return pd.DataFrame(rows, columns=['fold', 'participant_id', 'password_id',
                                       'enrol_week', 'attack', 'variant', 'score'])


# ------------------------------------------------------------ run

def run(folds_wanted=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for d in (TRIALS_DIR, FORGE_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    dataset = load_dataset()
    password_meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}
    log(f'device={device}, {len(dataset.samples):,} samples')

    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        if folds_wanted is not None and fold not in folds_wanted:
            continue
        trials_path = os.path.join(TRIALS_DIR, f'fold{fold}.parquet')
        forge_path = os.path.join(FORGE_DIR, f'fold{fold}.parquet')
        if os.path.exists(trials_path) and os.path.exists(forge_path):
            log(f'fold {fold}: already complete, skipping')
            continue

        t_fold = time.time()
        train_set = set(train_ids)
        assert not (train_set & set(test_ids)), 'train/test overlap'

        durations = [dataset.raw_sequences[i][:, 0] for i in range(len(dataset.samples))
                     if dataset.samples[i].participant_id in train_set]
        clipper = SequenceClipper().fit(durations)

        members = []
        for s in range(N_SEEDS):
            seed = BASE_SEED + s
            model = get_encoder(dataset, fold, seed, train_ids, val_ids, clipper, device)
            m = EmbeddingMatcher(f'M4_TypeFormer_seed{seed}', model, clipper, device=device)
            m.precompute_all(dataset)
            members.append(m)
        ensemble = EnsembleEmbeddingMatcher('M4_TypeFormer', members)
        log(f'  fold {fold}: {N_SEEDS} encoders ready, embedding cache built')

        rows = generate_trials(dataset, ensemble, clipper, test_ids, password_meta,
                               'M4_TypeFormer', fold, 'clipped', k_enrollment=K_ENROLLMENT)
        trials_to_frame(rows).to_parquet(trials_path, index=False)
        log(f'  fold {fold}: {len(rows):,} adversary trials')

        forg = forgery_scores_for_fold(dataset, ensemble, clipper, fold, train_ids,
                                       test_ids, password_meta, keys_by_pw, device)
        forg.to_parquet(forge_path, index=False)
        log(f'  fold {fold}: {len(forg):,} forgery scores '
            f'[fold total {(time.time() - t_fold) / 60:.1f} min]')

    log('run complete')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    ap.add_argument('--analyse-only', action='store_true')
    args = ap.parse_args()

    if not args.analyse_only:
        run(folds_wanted=set(args.folds) if args.folds else None)

    from exp3_analyse import analyse
    analyse()
