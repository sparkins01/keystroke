import os
RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
"""
Orchestrates the full experiment set described in Section 3 of the paper and
writes every table/figure/log into C:\\Keystroke\\Result.

Runs on CPU by default; pass --device cuda to use a GPU (ROCm builds expose
themselves under the same 'cuda' device string as NVIDIA/CUDA builds do).
See RunConfig / the --hidden-size, --epochs etc. CLI flags to scale the
embedding models up further once more compute is available.
"""
import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loading import load_samples
from dataset import Dataset
from splits import fold_splits
from features import IQRClipper
from embedding_models import (EmbeddingConfig, SequenceClipper, train_embedding_model,
                               EmbeddingMatcher, EnsembleEmbeddingMatcher)
from matchers_classical import M1Manhattan, M2ShallowML
from protocol import generate_trials
from metrics import summarize_by_mechanism_delta, plot_aging_curves, plot_det_curves
from stats_model import calibrate_thresholds, build_model_dataframe, fit_mixed_model
from password_metadata import build_password_metadata

DEFAULT_RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


class NoOpClipper:
    def transform(self, vec, password_id=None):
        return vec


class Progress:
    """Prints a running [done/total = pct%] line to the log as work
    completes, so a long unattended run's status can be read straight off
    the log file (`grep progress run_log.txt` or just tailing stdout)
    instead of inferred from timing guesses."""

    def __init__(self, total, log_fn):
        self.total = total
        self.done = 0
        self.log_fn = log_fn
        self.t0 = time.time()

    def step(self, label):
        self.done += 1
        pct = 100 * self.done / self.total
        elapsed = time.time() - self.t0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else float('nan')
        self.log_fn(f'[progress {self.done}/{self.total} = {pct:5.1f}%] {label} '
                     f'(elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m)')


@dataclass
class RunConfig:
    folds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    k_values: list = field(default_factory=lambda: [1, 3, 5])
    primary_k: int = 5
    clip_conditions: list = field(default_factory=lambda: ['clipped', 'unclipped'])
    max_impostors_per_trial: int = None  # None = every other test participant
    embedding_cfg: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    n_bootstrap: int = 300
    seed: int = 0
    device: str = 'cpu'
    n_ensemble_seeds: int = 3  # M3/M4 are trained this many times per fold and score-fused
    result_dir: str = DEFAULT_RESULT_DIR


def fit_matchers(dataset, train_ids, val_ids, clip_condition, cfg: RunConfig, progress: Progress, fold):
    """Returns {mechanism_name: (matcher, clipper_for_features_or_None)}."""
    if clip_condition == 'clipped':
        feats_by_pw = {}
        for pw_id in dataset.password_ids:
            idxs = [i for i in dataset.by_password[pw_id] if dataset.samples[i].participant_id in set(train_ids)]
            feats_by_pw[pw_id] = [dataset.feature_vectors[i] for i in idxs]
        feat_clipper = IQRClipper().fit(feats_by_pw)

        durations_train = [dataset.raw_sequences[i][:, 0] for i in range(len(dataset.samples))
                            if dataset.samples[i].participant_id in set(train_ids)]
        seq_clipper = SequenceClipper().fit(durations_train)
    else:
        feat_clipper = NoOpClipper()
        seq_clipper = None

    m1 = M1Manhattan()
    m1.fit(dataset, train_ids, feat_clipper)
    progress.step(f'fold {fold} [{clip_condition}] M1 fit')

    m2 = M2ShallowML(seed=cfg.seed)
    m2.fit(dataset, train_ids, feat_clipper)
    progress.step(f'fold {fold} [{clip_condition}] M2 fit')

    m3_members, m4_members = [], []
    for i in range(cfg.n_ensemble_seeds):
        seed = cfg.seed + i
        emb_cfg = dataclasses.replace(cfg.embedding_cfg, seed=seed)

        lstm_model = train_embedding_model('lstm', dataset, train_ids, val_ids, seq_clipper,
                                            emb_cfg, device=cfg.device, verbose=False)
        m3_single = EmbeddingMatcher(f'M3_TypeNet_seed{seed}', lstm_model, seq_clipper, device=cfg.device)
        m3_single.precompute_all(dataset)
        m3_members.append(m3_single)
        progress.step(f'fold {fold} [{clip_condition}] M3 seed {seed} trained')

        transformer_model = train_embedding_model('transformer', dataset, train_ids, val_ids, seq_clipper,
                                                   emb_cfg, device=cfg.device, verbose=False)
        m4_single = EmbeddingMatcher(f'M4_TypeFormer_seed{seed}', transformer_model, seq_clipper, device=cfg.device)
        m4_single.precompute_all(dataset)
        m4_members.append(m4_single)
        progress.step(f'fold {fold} [{clip_condition}] M4 seed {seed} trained')

    m3 = EnsembleEmbeddingMatcher('M3_TypeNet', m3_members)
    m4 = EnsembleEmbeddingMatcher('M4_TypeFormer', m4_members)

    return {
        'M1_Manhattan': (m1, feat_clipper),
        'M2_ShallowML': (m2, feat_clipper),
        'M3_TypeNet': (m3, None),
        'M4_TypeFormer': (m4, None),
    }


def run(cfg: RunConfig):
    result_dir = cfg.result_dir
    os.makedirs(result_dir, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    log(f'{len(samples)} samples, {len(dataset.participant_ids)} participants, '
        f'{len(dataset.password_ids)} passwords loaded in {time.time() - t0:.1f}s')

    meta_rows = build_password_metadata()
    password_meta = {r['password_id']: r for r in meta_rows}

    total_steps = len(cfg.folds) * len(cfg.clip_conditions) * (2 + cfg.n_ensemble_seeds * 2)
    progress = Progress(total_steps, log)
    log(f'Total work units to fit: {total_steps} '
        f'({len(cfg.folds)} folds x {len(cfg.clip_conditions)} clip conditions x '
        f'(M1 + M2 + {cfg.n_ensemble_seeds} seeds x [M3, M4]))')

    all_test_trials = []
    all_val_trials = []

    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids, seed=cfg.seed):
        if fold not in cfg.folds:
            continue
        for clip_condition in cfg.clip_conditions:
            t_fold = time.time()
            matchers = fit_matchers(dataset, train_ids, val_ids, clip_condition, cfg, progress, fold)
            log(f'fold {fold} [{clip_condition}]: matchers fit in {time.time() - t_fold:.1f}s')

            for mech_name, (matcher, clipper) in matchers.items():
                for k in cfg.k_values:
                    t_trial = time.time()
                    test_trials = generate_trials(
                        dataset, matcher, clipper, test_ids, password_meta,
                        mech_name, fold, clip_condition, k_enrollment=k,
                        max_impostors_per_trial=cfg.max_impostors_per_trial, seed=cfg.seed)
                    all_test_trials.extend(test_trials)

                    if k == cfg.primary_k:
                        val_trials = generate_trials(
                            dataset, matcher, clipper, val_ids, password_meta,
                            mech_name, fold, clip_condition, k_enrollment=k,
                            max_impostors_per_trial=cfg.max_impostors_per_trial, seed=cfg.seed + 1)
                        all_val_trials.extend(val_trials)

                    log(f'  fold {fold} [{clip_condition}] {mech_name} k={k}: '
                        f'{len(test_trials)} test trials in {time.time() - t_trial:.1f}s')

    test_trials_df = pd.DataFrame(all_test_trials)
    val_trials_df = pd.DataFrame(all_val_trials)
    # Parquet, not CSV: with every other test participant used as an impostor
    # these tables run into the tens of millions of rows / multiple GB as CSV.
    test_trials_df.to_parquet(os.path.join(result_dir, 'trials_test.parquet'), index=False)
    val_trials_df.to_parquet(os.path.join(result_dir, 'trials_val.parquet'), index=False)
    log(f'\n{len(test_trials_df)} test trials, {len(val_trials_df)} val trials written to {result_dir} (parquet)')

    log('\nComputing EER / DET / FNMR summary...')
    summary_df = summarize_by_mechanism_delta(test_trials_df, n_boot=cfg.n_bootstrap, seed=cfg.seed)
    summary_df.to_csv(os.path.join(result_dir, 'eer_by_mechanism_delta.csv'), index=False)

    figures_dir = os.path.join(result_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    primary_summary = summary_df[summary_df['k_enrollment'] == cfg.primary_k]
    plot_aging_curves(primary_summary, os.path.join(figures_dir, 'aging_curve.png'))
    for clip_condition in cfg.clip_conditions:
        plot_det_curves(test_trials_df, os.path.join(figures_dir, f'det_curve_{clip_condition}.png'),
                         cfg.primary_k, clip_condition)
    log(f'EER summary + figures written to {result_dir}')

    log('\nFitting mixed-effects model (primary k, clipped condition)...')
    model_input = test_trials_df[(test_trials_df['k_enrollment'] == cfg.primary_k) &
                                  (test_trials_df['clipped'] == 'clipped')]
    val_for_thresholds = val_trials_df[(val_trials_df['k_enrollment'] == cfg.primary_k) &
                                        (val_trials_df['clipped'] == 'clipped')]
    thresholds = calibrate_thresholds(val_for_thresholds)
    model_df = build_model_dataframe(model_input, thresholds)
    model_df.to_parquet(os.path.join(result_dir, 'mixed_model_input.parquet'), index=False)

    try:
        fit = fit_mixed_model(model_df, seed=cfg.seed)
        stats_dir = os.path.join(result_dir, 'stats_model')
        os.makedirs(stats_dir, exist_ok=True)
        with open(os.path.join(stats_dir, 'mixed_model_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(f"delta centred on mean={fit['centering']['delta_mean']:.3f}, "
                     f"length centred on mean={fit['centering']['length_mean']:.3f}\n\n")
            f.write(f'FULL MODEL (delta_c * mechanism + length_c + substitution + hardware_change) '
                     f"[fit method: {fit['full_method']}, converged={fit['full_result'].converged}]\n")
            f.write(str(fit['full_result'].summary()))
            f.write(f"\n\nREDUCED MODEL (no delta:mechanism interaction) "
                     f"[fit method: {fit['reduced_method']}, converged={fit['reduced_result'].converged}]\n")
            f.write(str(fit['reduced_result'].summary()))
            f.write(f"\n\nLikelihood-ratio test for the delta x mechanism interaction:\n"
                     f"LR stat = {fit['lr_test']['lr_stat']:.3f}, df = {fit['lr_test']['df']}, "
                     f"p = {fit['lr_test']['p_value']:.4g}\n")
            f.write(f"\nRows used (after subsampling cap): {fit['n_rows_used']}\n")
        log(f"Mixed model fit. LR test for delta x mechanism: "
            f"stat={fit['lr_test']['lr_stat']:.2f}, df={fit['lr_test']['df']}, p={fit['lr_test']['p_value']:.4g}")
    except Exception as e:
        log(f'Mixed model fitting failed: {e!r}')

    with open(os.path.join(result_dir, 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f'\nTotal wall time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    parser.add_argument('--k-values', type=int, nargs='*', default=[1, 3, 5])
    parser.add_argument('--clip-conditions', type=str, nargs='*', default=['clipped', 'unclipped'])
    parser.add_argument('--max-impostors', type=int, default=None,
                         help='cap impostors per genuine trial; omit to use every other test participant')
    parser.add_argument('--epochs', type=int, default=None, help='override EmbeddingConfig.epochs')
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' or 'cuda' (ROCm builds use 'cuda' too)")
    parser.add_argument('--hidden-size', type=int, default=None, help='override EmbeddingConfig.hidden_size')
    parser.add_argument('--embedding-dim', type=int, default=None, help='override EmbeddingConfig.embedding_dim')
    parser.add_argument('--steps-per-epoch', type=int, default=None, help='override EmbeddingConfig.steps_per_epoch')
    parser.add_argument('--batch-triplets', type=int, default=None, help='override EmbeddingConfig.batch_triplets')
    parser.add_argument('--n-ensemble-seeds', type=int, default=3,
                         help='train M3/M4 this many times per fold and average their scores (score-level fusion)')
    parser.add_argument('--seed', type=int, default=0,
                         help='base seed: ensemble members use seed, seed+1, ...; also used for fold split, '
                              'M2 training, and bootstrap/subsampling. Vary this to get an independent replicate '
                              'run (different training runs, not just a different subsample of the same trials).')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR,
                         help='where to write trials/figures/stats_model for this run; use a distinct directory '
                              'per --seed to run independent replicates without overwriting each other')
    args = parser.parse_args()

    cfg = RunConfig(
        folds=args.folds, k_values=args.k_values, clip_conditions=args.clip_conditions,
        max_impostors_per_trial=args.max_impostors, device=args.device,
        n_ensemble_seeds=args.n_ensemble_seeds, seed=args.seed, result_dir=args.output_dir,
    )
    if args.epochs is not None:
        cfg.embedding_cfg.epochs = args.epochs
    if args.hidden_size is not None:
        cfg.embedding_cfg.hidden_size = args.hidden_size
    if args.embedding_dim is not None:
        cfg.embedding_cfg.embedding_dim = args.embedding_dim
    if args.steps_per_epoch is not None:
        cfg.embedding_cfg.steps_per_epoch = args.steps_per_epoch
    if args.batch_triplets is not None:
        cfg.embedding_cfg.batch_triplets = args.batch_triplets
    print(f'Device: {cfg.device}, embedding config: {cfg.embedding_cfg}')
    run(cfg)
