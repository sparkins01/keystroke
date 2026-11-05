"""
Generates the additional figures for the paper draft (results/figures/):
  - k_sweep.png            EER vs. enrolment size k, per mechanism
  - seed_robustness.png    per-fold, per-seed EER strip plot, M3 vs M4
  - forest_plot.png        mixed-effects model fixed-effect coefficients + 95% CI
  - trial_design.png       schematic of the enrolment-week x query-week trial grid
  - det_curves_combined.png  side-by-side clipped/unclipped DET curves

Run after run_experiments.py; reads from results/, writes to results/figures/.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'figures')

MECH_COLORS = {
    'M1_Manhattan': '#1f77b4', 'M2_ShallowML': '#2ca02c',
    'M3_TypeNet': '#9467bd', 'M4_TypeFormer': '#e377c2',
}
MECH_LABELS = {
    'M1_Manhattan': 'M1 Manhattan', 'M2_ShallowML': 'M2 Shallow ML',
    'M3_TypeNet': 'M3 TypeNet (LSTM)', 'M4_TypeFormer': 'M4 TypeFormer',
}


def plot_k_sweep():
    df = pd.read_csv(os.path.join(RESULT_DIR, 'eer_by_mechanism_delta.csv'))
    sub = df[df['clipped'] == 'clipped']
    agg = sub.groupby(['mechanism', 'k_enrollment'])['eer_pooled'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    for mech in ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer']:
        g = agg[agg['mechanism'] == mech].sort_values('k_enrollment')
        ax.plot(g['k_enrollment'], g['eer_pooled'] * 100, marker='o',
                color=MECH_COLORS[mech], label=MECH_LABELS[mech])
    ax.set_xlabel('Enrolment sessions ($k$)')
    ax.set_ylabel('Mean EER (%), pooled across $\\Delta$')
    ax.set_xticks([1, 3, 5])
    ax.set_title('Verification accuracy vs. enrolment burden')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'k_sweep.png'), dpi=200)
    plt.close(fig)


def plot_seed_robustness():
    df = pd.read_csv(os.path.join(RESULT_DIR, 'stats_model', 'seed_robustness_by_fold.csv'))
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.RandomState(0)
    mechs = ['M3_TypeNet', 'M4_TypeFormer']
    offsets = {'M3_TypeNet': -0.12, 'M4_TypeFormer': 0.12}
    for mech in mechs:
        g = df[df['mechanism'] == mech]
        for fold, gg in g.groupby('fold'):
            x = fold + offsets[mech] + rng.uniform(-0.03, 0.03, size=len(gg))
            ax.scatter(x, gg['eer_pooled'] * 100, color=MECH_COLORS[mech], alpha=0.85,
                       label=MECH_LABELS[mech] if fold == 0 else None, zorder=3)
        # connect fold means for visual guide
        means = g.groupby('fold')['eer_pooled'].mean() * 100
        ax.plot(means.index + offsets[mech], means.values, color=MECH_COLORS[mech],
                alpha=0.4, linestyle='--', zorder=2)
    ax.set_xlabel('Fold')
    ax.set_ylabel('EER (%), single-seed (no fusion)')
    ax.set_xticks(range(5))
    ax.set_title('Training-seed spread per fold (3 seeds each, unfused)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'seed_robustness.png'), dpi=200)
    plt.close(fig)


def plot_forest():
    # (term, coef, z) from stats_model/mixed_model_summary.txt, full model
    terms = [
        ('Mechanism: M4 vs. M1', -0.065, -30.15),
        ('Mechanism: M2 vs. M1', -0.040, -18.30),
        ('Mechanism: M3 vs. M1', 0.052, 23.90),
        ('Length (centred)', -0.020, -11.27),
        ('Hardware change (wk 5)', 0.005, 2.08),
        ('$\\Delta$ (centred)', 0.017, 21.60),
        ('$\\Delta$:M2 interaction', -0.002, -2.187),
        ('$\\Delta$:M3 interaction', -0.002, -1.985),
        ('$\\Delta$:M4 interaction', -0.000, -0.416),
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ys = np.arange(len(terms))[::-1]
    for y, (name, coef, z) in zip(ys, terms):
        se = abs(coef / z) if z != 0 else 0
        ci = 1.96 * se
        significant = abs(z) >= 1.96
        color = '#1f77b4' if significant else '#999999'
        ax.errorbar(coef, y, xerr=ci, fmt='o', color=color, capsize=3)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(ys)
    ax.set_yticklabels([t[0] for t in terms], fontsize=8)
    ax.set_xlabel('Coefficient (probability-of-decision-error scale)')
    ax.set_title('Mixed-effects model: fixed effects (95% CI)')
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'forest_plot.png'), dpi=200)
    plt.close(fig)


def plot_trial_design():
    fig, ax = plt.subplots(figsize=(5, 5))
    weeks = np.arange(1, 9)
    grid = np.full((8, 8), np.nan)
    for i in weeks:
        for j in weeks:
            if j >= i:
                grid[j - 1, i - 1] = j - i  # delta

    im = ax.imshow(grid, origin='lower', cmap='viridis', extent=[0.5, 8.5, 0.5, 8.5])
    for i in weeks:
        for j in weeks:
            if j >= i:
                ax.text(i, j, str(j - i), ha='center', va='center', fontsize=7,
                        color='white' if (j - i) > 3 else 'black')
    ax.set_xlabel('Enrolment week $i$')
    ax.set_ylabel('Query week $j$')
    ax.set_title('Trial grid: cell value = $\\Delta = j - i$')
    ax.set_xticks(weeks)
    ax.set_yticks(weeks)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$\\Delta$ (weeks)')
    ax.text(1.6, 7.6, 'genuine: own\nquery sample\nimpostor: other\ntest participant',
            fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'trial_design.png'), dpi=200)
    plt.close(fig)


def combine_det_curves():
    img1 = mpimg.imread(os.path.join(OUT_DIR, 'det_curve_clipped.png'))
    img2 = mpimg.imread(os.path.join(OUT_DIR, 'det_curve_unclipped.png'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Clipped', fontsize=10)
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Unclipped', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'det_curves_combined.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_k_sweep()
    plot_seed_robustness()
    plot_forest()
    plot_trial_design()
    combine_det_curves()
    print('Wrote figures to', OUT_DIR)
