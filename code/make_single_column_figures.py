"""
Redraw the three main figures at single-column width.

Scaling a wide figure down to a column makes its text unreadable, so each is
redrawn on a column-sized canvas with type sized for it rather than shrunk
into it.

Figure 1 shows the clipped condition only. At column width nine series and a
nine-entry legend are unreadable, and the unclipped numbers are already
tabulated in full in the appendix, so the figure keeps the comparison it is
there to make and the caption points to the table for the rest.

Figures 2 and 3 keep both panels but stack them, since two panels side by
side in a column leaves each about 1.7 inches wide.

Writes *_1col.png beside the originals, leaving the existing files alone.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

RESULTS = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
ITAD = os.path.join(RESULTS, 'with_itad')
REV = os.path.join(RESULTS, 'revision')
OUT = os.environ.get('KD_FIG_DIR', os.path.join(RESULTS, 'figures'))

COL_W = 3.45          # IEEE two-column text: one column is about 3.45 inches
BASE_FONT = 7.0

MECHS = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer', 'M5_ITAD']
LABEL = {'M1_Manhattan': 'M1', 'M2_ShallowML': 'M2', 'M3_TypeNet': 'M3',
         'M4_TypeFormer': 'M4', 'M5_ITAD': 'M5'}
COLOUR = {'M1_Manhattan': '#0072B2', 'M2_ShallowML': '#009E73',
          'M3_TypeNet': '#CC79A7', 'M4_TypeFormer': '#E69F00',
          'M5_ITAD': '#000000'}
MARKER = {'M1_Manhattan': 'o', 'M2_ShallowML': 's', 'M3_TypeNet': '^',
          'M4_TypeFormer': 'D', 'M5_ITAD': 'v'}


def style():
    plt.rcParams.update({
        'font.size': BASE_FONT,
        'axes.labelsize': BASE_FONT,
        'axes.titlesize': BASE_FONT,
        'xtick.labelsize': BASE_FONT - 0.5,
        'ytick.labelsize': BASE_FONT - 0.5,
        'legend.fontsize': BASE_FONT - 0.5,
        'lines.linewidth': 1.1,
        'lines.markersize': 3.2,
    })


def fig1():
    df = pd.read_csv(os.path.join(ITAD, 'eer_by_mechanism_delta.csv'),
                     keep_default_na=False, na_values=[''])
    df = df[(df['k_enrollment'] == 5)
            & (df['clipped'].isin(['clipped', 'not_applicable']))]
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    for m in MECHS:
        g = df[df['mechanism'] == m].sort_values('delta')
        if g.empty:
            continue
        ax.plot(g['delta'], g['eer_pooled'] * 100, marker=MARKER[m],
                color=COLOUR[m], label=LABEL[m])
        ax.fill_between(g['delta'], g['eer_ci_lo'] * 100, g['eer_ci_hi'] * 100,
                        color=COLOUR[m], alpha=0.07, linewidth=0)
    ax.set_xlabel(r'Enrolment-to-query gap $\Delta$ (weeks)')
    ax.set_ylabel('EER (%)')
    ax.set_xticks(range(8))
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(ncol=5, fontsize=BASE_FONT - 1, columnspacing=0.9,
              handletextpad=0.35, borderpad=0.3, loc='upper left',
              framealpha=0.95)
    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, 'aging_curve_1col.png')
    fig.savefig(p, dpi=400)
    plt.close(fig)
    print('wrote %s' % p)


def fig2():
    cols = ['mechanism', 'clipped', 'k_enrollment', 'label', 'score']
    main = pd.read_parquet(os.path.join(RESULTS, 'trials_test.parquet'),
                           columns=cols)
    itad = pd.read_parquet(os.path.join(ITAD, 'itad_trials_test.parquet'),
                           columns=cols)
    df = pd.concat([main, itad], ignore_index=True)
    df = df[df['k_enrollment'] == 5]

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    anchors = np.array([0.01, 0.1, 1, 10])
    sub = df[(df['clipped'] == 'clipped') | (df['clipped'] == 'not_applicable')]
    for m in MECHS:
        g = sub[sub['mechanism'] == m]
        if g.empty:
            continue
        fpr, tpr, _ = roc_curve(g['label'].values, g['score'].values)
        fnr = 1 - tpr
        keep = (fpr > 0) & (fnr > 0)
        x, y = fpr[keep] * 100, fnr[keep] * 100
        ax.plot(x, y, color=COLOUR[m], linewidth=1.2)
        ax.plot(anchors, np.interp(anchors, x, y), linestyle='none',
                marker=MARKER[m], markerfacecolor='white',
                markeredgecolor=COLOUR[m], label=LABEL[m])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5e-3, 100)
    ax.set_ylim(3, 105)
    ax.set_xlabel('False match rate (%)')
    ax.set_ylabel('FNMR (%)')
    ax.grid(alpha=0.3, which='both', linestyle=':')
    ax.legend(ncol=5, fontsize=BASE_FONT - 1, columnspacing=0.8,
              handletextpad=0.3, loc='lower left', framealpha=0.95)
    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, 'det_curves_1col.png')
    fig.savefig(p, dpi=400)
    plt.close(fig)
    print('wrote %s' % p)


def fig3():
    df = pd.read_csv(os.path.join(REV, 'fixed_threshold.csv'))
    e = df[df['operating_point'] == 'eer']
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    for m in MECHS:
        g = e[e['mechanism'] == m].sort_values('delta')
        if g.empty:
            continue
        ax.plot(g['delta'], g['fnmr_frozen_mean'] * 100, marker=MARKER[m],
                color=COLOUR[m], label=LABEL[m])
        ax.plot(g['delta'], g['fmr_frozen_mean'] * 100, linestyle=':',
                color=COLOUR[m], alpha=0.6)
    ax.set_xlabel(r'Enrolment-to-query gap $\Delta$ (weeks)')
    ax.set_ylabel('Error rate (%)')
    ax.set_xticks(range(8))
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(ncol=5, fontsize=BASE_FONT - 1, columnspacing=0.8,
              handletextpad=0.3, loc='upper left', framealpha=0.95)
    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, 'fixed_threshold_1col.png')
    fig.savefig(p, dpi=400)
    plt.close(fig)
    print('wrote %s' % p)


if __name__ == '__main__':
    style()
    fig1()
    fig2()
    fig3()
