"""
Decomposes the "fold" variance seen in the main results into two pieces:
genuine population variation (different participants/passwords in each
fold's train/val/test split) vs. neural-network training-seed noise (each
fold's M3/M4 was trained exactly once, so the two were previously
conflated).

For each of the 5 folds (clipped condition, k=5, matching the primary
reported configuration), M3 and M4 are retrained from scratch with three
different random seeds on that fold's *same* data split, and pooled EER is
computed each time. The spread across those 3 seeds (same data, different
training luck) is compared against the spread already observed across the
5 folds (different data) in eer_by_mechanism_delta.csv.

This does not change any headline number -- it's a diagnostic answering
"how much of the fold-to-fold spread you see in the main results is the
model training being noisy, versus a real difference between which
participants ended up in which fold". Output: Result/stats_model/
seed_robustness_by_fold.csv and a short summary markdown.
"""
import argparse
import dataclasses
import os
import time

import pandas as pd

from data_loading import load_samples
from dataset import Dataset
from splits import fold_splits
from embedding_models import EmbeddingConfig, SequenceClipper, train_embedding_model, EmbeddingMatcher
from protocol import generate_trials
from metrics import compute_eer
from password_metadata import build_password_metadata

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
SEEDS = [0, 1, 2]
K_ENROLLMENT = 5


def main(device='cpu', base_cfg=None):
    t0 = time.time()
    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    meta_rows = build_password_metadata()
    password_meta = {r['password_id']: r for r in meta_rows}

    base_cfg = base_cfg or EmbeddingConfig()
    rows = []

    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        durations_train = [dataset.raw_sequences[i][:, 0] for i in range(len(dataset.samples))
                            if dataset.samples[i].participant_id in set(train_ids)]
        seq_clipper = SequenceClipper().fit(durations_train)

        for seed in SEEDS:
            cfg = dataclasses.replace(base_cfg, seed=seed)
            for architecture, mech_name in [('lstm', 'M3_TypeNet'), ('transformer', 'M4_TypeFormer')]:
                t_start = time.time()
                model = train_embedding_model(architecture, dataset, train_ids, val_ids, seq_clipper,
                                               cfg, device=device, verbose=False)
                matcher = EmbeddingMatcher(mech_name, model, seq_clipper, device=device)
                trials = generate_trials(dataset, matcher, None, test_ids, password_meta,
                                          mech_name, fold, 'clipped', k_enrollment=K_ENROLLMENT, seed=seed)
                df = pd.DataFrame(trials)
                eer, _ = compute_eer(df['label'], df['score'])
                elapsed = time.time() - t_start
                rows.append({'fold': fold, 'seed': seed, 'mechanism': mech_name,
                             'eer_pooled': eer, 'n_trials': len(df), 'train_seconds': elapsed})
                print(f'fold={fold} seed={seed} {mech_name}: eer={eer:.4f} '
                      f'({len(df)} trials, {elapsed:.1f}s)', flush=True)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULT_DIR, 'stats_model'), exist_ok=True)
    out_csv = os.path.join(RESULT_DIR, 'stats_model', 'seed_robustness_by_fold.csv')
    result_df.to_csv(out_csv, index=False)

    summary_lines = ['# Seed-robustness check: training noise vs. genuine fold variation\n']
    for mechanism, g in result_df.groupby('mechanism'):
        within_fold_sd = g.groupby('fold')['eer_pooled'].std().mean()
        across_fold_sd = g.groupby('fold')['eer_pooled'].mean().std()
        summary_lines.append(f'## {mechanism}\n')
        summary_lines.append(f'- Mean within-fold SD across 3 seeds (training noise, data held fixed): '
                              f'{within_fold_sd:.4f}\n')
        summary_lines.append(f'- SD across the 5 folds\' seed-0 means (genuine data variation + noise): '
                              f'{across_fold_sd:.4f}\n')
        per_fold = g.pivot_table(index='fold', columns='seed', values='eer_pooled')
        summary_lines.append('\n' + per_fold.to_markdown() + '\n')

    summary_path = os.path.join(RESULT_DIR, 'stats_model', 'seed_robustness_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f'\nWrote {out_csv} and {summary_path}')
    print(f'Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden-size', type=int, default=None)
    parser.add_argument('--embedding-dim', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--batch-triplets', type=int, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in {
        'hidden_size': args.hidden_size, 'embedding_dim': args.embedding_dim,
        'epochs': args.epochs, 'steps_per_epoch': args.steps_per_epoch,
        'batch_triplets': args.batch_triplets,
    }.items() if v is not None}
    cfg = dataclasses.replace(EmbeddingConfig(), **overrides)
    print(f'Device: {args.device}, embedding config: {cfg}')
    main(device=args.device, base_cfg=cfg)
