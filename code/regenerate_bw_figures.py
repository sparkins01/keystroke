"""Regenerate Figures 1 and 2 so they remain readable in black and white.

Both figures previously relied on colour alone to separate the four matching
mechanisms. Here each mechanism gets a distinct *marker shape* and each
clipping condition a distinct *line style*, so the figures survive greyscale
printing and photocopying; colour is retained as a redundant cue only, using
the Okabe-Ito colourblind-safe palette with deliberately spread luminance.

Reads from results/ and writes into results/figures/. Does NOT require re-running
the experiment pipeline: the ageing curve comes from the saved EER summary and
the DET panel from the saved test-trial scores.

  python regenerate_bw_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.environ.get('KD_FIG_DIR', os.path.join(RESULT_DIR, 'figures'))

# Okabe-Ito, chosen so the four also separate by luminance in greyscale.
MECHS = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer',
         'M5_ITAD']
LABEL = {
    'M1_Manhattan':  'M1 scaled Manhattan',
    'M2_ShallowML':  'M2 shallow ML',
    'M3_TypeNet':    'M3 TypeNet (LSTM)',
    'M4_TypeFormer': 'M4 TypeFormer',
    'M5_ITAD':       'M5 ITAD',
}
# Okabe-Ito colourblind-safe palette, ordered so the four also separate by
# luminance (blue darkest -> orange lightest) and therefore stay distinct if
# the colour is thrown away at print time.
COLOUR = {
    'M1_Manhattan':  '#0072B2',   # blue          (lum ~0.28)
    'M2_ShallowML':  '#009E73',   # bluish green  (lum ~0.45)
    'M3_TypeNet':    '#CC79A7',   # reddish purple(lum ~0.55)
    'M4_TypeFormer': '#E69F00',   # orange        (lum ~0.65)
    'M5_ITAD':       '#000000',   # black         (lum 0)
}
MARKER = {
    'M1_Manhattan':  'o',
    'M2_ShallowML':  's',
    'M3_TypeNet':    '^',
    'M4_TypeFormer': 'D',
    'M5_ITAD':       'v',
}
# clipping condition -> line style
STYLE = {'clipped': '-', 'unclipped': '--', 'not_applicable': '-'}
# ITAD scores percentiles within the enrolment profile, so the feature
# clipper cannot change it and there is only one curve to draw.
NO_CLIPPING = 'not_applicable'

PRIMARY_K = 5
ITAD_DIR = os.path.join(RESULT_DIR, 'with_itad')


def _with_itad(df, filename, columns=None):
    """Append the ITAD rows if they have been generated."""
    path = os.path.join(ITAD_DIR, filename)
    if not os.path.exists(path):
        print('note: %s not found, figures exclude ITAD' % path)
        return df
    if path.endswith('.csv'):
        extra = pd.read_csv(path)
        extra = extra[extra['mechanism'] == 'M5_ITAD']
    else:
        extra = pd.read_parquet(path, columns=columns)
    return pd.concat([df, extra], ignore_index=True)


def fig1_ageing_curve():
    """Figure 1: EER vs enrolment-to-query gap, by mechanism and clipping."""
    df = pd.read_csv(os.path.join(RESULT_DIR, 'eer_by_mechanism_delta.csv'))
    df = _with_itad(df, 'eer_by_mechanism_delta.csv')
    df = df[df['k_enrollment'] == PRIMARY_K]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for mech in MECHS:
        conditions = [NO_CLIPPING] if mech == 'M5_ITAD' else ['clipped', 'unclipped']
        for cond in conditions:
            g = df[(df['mechanism'] == mech) & (df['clipped'] == cond)].sort_values('delta')
            if g.empty:
                continue
            filled = (cond != 'unclipped')
            ax.plot(
                g['delta'], g['eer_pooled'] * 100,
                linestyle=STYLE[cond], color=COLOUR[mech], marker=MARKER[mech],
                markersize=6.5, markerfacecolor=(COLOUR[mech] if filled else 'white'),
                markeredgecolor=COLOUR[mech], markeredgewidth=1.3,
                linewidth=1.8 if filled else 1.4,
                label=(LABEL[mech] if cond == NO_CLIPPING
                       else f'{LABEL[mech]} ({cond})'),
            )
            if filled:   # CI band only on the primary (clipped) condition
                ax.fill_between(g['delta'], g['eer_ci_lo'] * 100, g['eer_ci_hi'] * 100,
                                color=COLOUR[mech], alpha=0.08, linewidth=0)

    ax.set_xlabel(r'Enrolment-to-query gap $\Delta$ (weeks)', fontsize=11)
    ax.set_ylabel('EER (%)', fontsize=11)
    ax.set_xticks(range(0, 8))
    ax.grid(alpha=0.3, linestyle=':')
    ax.tick_params(labelsize=10)

    # order the legend by mechanism, clipped above unclipped
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: (l[i].split(' (')[0], 'unclipped' in l[i]))
    ax.legend([h[i] for i in order], [l[i] for i in order],
              fontsize=8.5, ncol=2, framealpha=0.95, loc='upper left')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'aging_curve.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('wrote', out)


def fig2_det_panel():
    """Figure 2: DET curves, clipped and unclipped, drawn natively.

    The previous version pasted two already-rendered PNGs side by side, which
    resampled both and cost resolution. This draws both panels directly.
    """
    cols = ['mechanism', 'clipped', 'k_enrollment', 'label', 'score']
    df = pd.read_parquet(os.path.join(RESULT_DIR, 'trials_test.parquet'), columns=cols)
    df = _with_itad(df, 'itad_trials_test.parquet', columns=cols)
    df = df[df['k_enrollment'] == PRIMARY_K]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)

    # Markers are placed at fixed FMR anchors rather than at fixed array
    # positions, so they stay evenly spread across the visible region.
    ANCHORS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    for ax, cond in zip(axes, ['clipped', 'unclipped']):
        sub = df[(df['clipped'] == cond) | (df['clipped'] == NO_CLIPPING)]
        for mech in MECHS:
            g = sub[sub['mechanism'] == mech]
            if g.empty:
                continue
            fpr, tpr, _ = roc_curve(g['label'].values, g['score'].values)
            fnr = 1 - tpr
            keep = (fpr > 0) & (fnr > 0)
            x, y = fpr[keep] * 100, fnr[keep] * 100

            ax.plot(x, y, linestyle='-', color=COLOUR[mech], linewidth=2.0, zorder=2)
            ax.plot(ANCHORS, np.interp(ANCHORS, x, y),
                    linestyle='none', marker=MARKER[mech], markersize=7.0,
                    markerfacecolor='white', markeredgecolor=COLOUR[mech],
                    markeredgewidth=1.6, label=LABEL[mech], zorder=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        # Bound the axes to the region where the mechanisms actually separate.
        # The unbounded ROC tails (FNMR -> 0 as FMR -> 100%) otherwise stretch
        # the y-axis over five decades and squash every curve into one corner.
        ax.set_xlim(5e-3, 100)
        ax.set_ylim(3, 105)
        ax.set_xlabel('False match rate (%)', fontsize=11)
        ax.set_title(cond.capitalize(), fontsize=11)
        ax.grid(alpha=0.3, which='both', linestyle=':')
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel('False non-match rate (%)', fontsize=11)
    axes[0].legend(fontsize=9, framealpha=0.95, loc='lower left')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'det_curves_combined.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('wrote', out)


def greyscale_proofs():
    """Write greyscale copies so the print appearance can be checked directly."""
    try:
        from PIL import Image
    except ImportError:
        print('Pillow not available; skipping greyscale proofs')
        return
    for name in ['aging_curve.png', 'det_curves_combined.png']:
        src = os.path.join(OUT_DIR, name)
        dst = os.path.join(OUT_DIR, name.replace('.png', '_greyscale_proof.png'))
        Image.open(src).convert('L').save(dst)
        print('wrote', dst)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_ageing_curve()
    fig2_det_panel()
    greyscale_proofs()
