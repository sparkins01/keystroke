"""
Figures for the paper.

Every series carries three redundant encodings: colour, marker and line style.
The palette is Okabe-Ito, validated for colour-vision deficiency, and the
mechanism-to-colour assignment is the one used in the template ageing paper so
that a reader moving between the two sees the same mechanism in the same colour.
The validator flags the pink/green pair at delta-E 7.6 under deuteranopia, which
is inside the floor band and permitted only with secondary encoding, hence the
markers and line styles rather than as decoration.

A greyscale proof of every figure is written alongside it, since Computers &
Security is read in print as often as on screen.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FIG_DIR = os.environ.get('KD_ADVERSARY_FIG_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures'))
PROOF_DIR = os.path.join(FIG_DIR, 'greyscale_proofs')

MECHS = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer']
# R1: labels match the names used in the text and in the method diagram
# (reviewer 2, comment 7), rather than the implementation names
LABEL = {'M1_Manhattan': 'M1 distance', 'M2_ShallowML': 'M2 classifier',
         'M3_TypeNet': 'M3 recurrent', 'M4_TypeFormer': 'M4 transformer'}

# Okabe-Ito, same assignment as the template ageing paper
COLOUR = {'M1_Manhattan': '#0072B2', 'M2_ShallowML': '#009E73',
          'M3_TypeNet': '#CC79A7', 'M4_TypeFormer': '#E69F00'}
MARKER = {'M1_Manhattan': 'o', 'M2_ShallowML': 's',
          'M3_TypeNet': '^', 'M4_TypeFormer': 'D'}
LINE = {'M1_Manhattan': '-', 'M2_ShallowML': '--',
        'M3_TypeNet': '-.', 'M4_TypeFormer': (0, (3, 1, 1, 1))}

ADV_COLOUR = {'A1': '#0072B2', 'A2': '#009E73', 'A4': '#D55E00', 'A4f': '#CC79A7'}
ADV_MARKER = {'A1': 'o', 'A2': 's', 'A4': 'D', 'A4f': 'v'}
ADV_LINE = {'A1': '-', 'A2': '--', 'A4': '-.', 'A4f': ':'}

CHAR_BITS = 4.70   # one extra random lowercase character

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'lines.linewidth': 1.6, 'lines.markersize': 5,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PROOF_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    png = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    Image.open(png).convert('L').save(
        os.path.join(PROOF_DIR, f'{name}_greyscale.png'))
    print(f'  {name}.pdf')


def load():
    return (pd.read_csv(os.path.join(RESULT_DIR, 'exp3_comparison.csv')),
            pd.read_csv(os.path.join(RESULT_DIR, 'exp8_retry_curve.csv')),
            # composition and coverage on the two-week leak, the paper's primary horizon
            pd.read_csv(os.path.join(RESULT_DIR, 'exp5_policy_early.csv')),
            pd.read_csv(os.path.join(RESULT_DIR, 'exp5b_coverage_early.csv')))


# ------------------------------------------------------------------ fig 1

def fig_ladder(comp):
    """Attack success across the adversary model. A line rather than grouped
    bars, because the ordering of the adversary levels is the finding."""
    rungs = [('A0_unpractised', 'A0\nunpractised\nhuman'),
             ('A1_practised', 'A1\npractised\nhuman'),
             ('A2_sampled', 'A2\npopulation\nstatistics'),
             ('A4_global_sampled_early', 'A3\nvictim\nspeed\n(2-week leak)'),
             ('A4_keyed_sampled_early', 'A4\nvictim timings\n(2-week leak)'),
             ('A4_keyed_sampled_full', 'A4\nvictim timings\n(full leak)')]
    x = np.arange(len(rungs))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    d0 = comp[comp['delta'] == 0]
    for m in MECHS:
        sub = d0[d0['mechanism'] == m].set_index('attack')['fmr']
        y = [sub.get(k, np.nan) for k, _ in rungs]
        ax.plot(x, y, color=COLOUR[m], marker=MARKER[m], linestyle=LINE[m],
                label=LABEL[m], markerfacecolor='white', markeredgewidth=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in rungs])
    ax.set_ylabel('Attack success (false match rate)')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.35, len(rungs) - 0.65)
    ax.axhspan(0, 0.5, color='0.92', zorder=0)
    ax.text(-0.28, 0.025, 'adversary fails more often than not', fontsize=7,
            color='0.4', va='bottom')
    ax.legend(loc='upper left', frameon=False, ncol=2)
    ax.set_title('Attack success rises with adversary knowledge of the victim,\n'
                 'not with adversary effort (newly enrolled template)', loc='left')
    save(fig, 'fig1_adversary_ladder')


# ------------------------------------------------------------------ fig 2

def fig_ageing(comp):
    """Marginal security against template age, one panel per mechanism."""
    # R1: both leak horizons are shown and labelled (reviewer 2, comment 1)
    rungs = [('A1_practised', 'A1'), ('A2_sampled', 'A2'),
             ('A4_keyed_sampled_early', 'A4'), ('A4_keyed_sampled_full', 'A4f')]
    names = {'A1': 'A1 practised human', 'A2': 'A2 population statistics',
             'A4': 'A4 victim timings, 2-week leak',
             'A4f': 'A4 victim timings, full leak'}

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), MECHS):
        sub = comp[comp['mechanism'] == m]
        for key, short in rungs:
            s = sub[sub['attack'] == key].sort_values('delta')
            ax.plot(s['delta'], s['bits'], color=ADV_COLOUR[short],
                    marker=ADV_MARKER[short], linestyle=ADV_LINE[short],
                    label=names[short], markerfacecolor='white',
                    markeredgewidth=1.3)
        ax.set_title(LABEL[m], loc='left')
        ax.set_ylim(0, 2.4)
        ax.set_xlim(-0.3, 7.3)

    for ax in axes[1]:
        ax.set_xlabel('Enrolment to query gap $\\Delta$ (weeks)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Marginal security (bits)')

    axes[0, 0].legend(loc='upper right', frameon=False)
    fig.suptitle('Threshold recalibrated at each gap to hold the false non-match rate at 5%',
                 fontsize=9, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'fig2_marginal_security')


# ------------------------------------------------------------------ fig 3

def fig_retry(curve):
    """Cumulative attack success against the number of permitted attempts.

    R1 (reviewer 2, comment 2): plotted from the empirical per-gallery curve of
    experiment 8, not from 1 - (1 - p)^n on the pooled rate, and only as far as
    the data supports: three attempts for a human, five for a forgery.
    """
    panels = [('A1 practised human', 'Practised human impostor'),
              ('A4 synthetic, keyed, 2-week leak', 'Victim timings, 2-week leak'),
              ('A4 synthetic, keyed, full leak', 'Victim timings, full leak')]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), sharey=True)
    c0 = curve[curve['delta'] == 0]
    for ax, (adv, title) in zip(axes, panels):
        for m in MECHS:
            s = c0[(c0['mechanism'] == m) & (c0['adversary'] == adv)].sort_values('attempts')
            if not len(s):
                continue
            ax.plot(s['attempts'], s['empirical'], color=COLOUR[m], marker=MARKER[m],
                    linestyle=LINE[m], label=LABEL[m], markerfacecolor='white',
                    markeredgewidth=1.3)
        ax.axvline(3, color='0.4', linewidth=0.9, linestyle=':')
        ax.set_title(title, loc='left', fontsize=8)
        ax.set_xlabel('Permitted attempts')
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0.7, 5.3)
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[0].text(3.12, 0.06, 'retry\nlimit', fontsize=7, color='0.35')
    axes[0].set_ylabel('Cumulative attack success')
    axes[0].legend(loc='upper left', frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, 'fig3_retry_policy')


# ------------------------------------------------------------------ fig 4

def fig_policy(pol):
    """Slope chart per mechanism: the ranking of composition policies under two
    adversaries. A slope chart is used because the finding is the change in
    ordering, not the levels."""
    sub = pol[(pol['grouping'] == 'substitution_category') & pol['leaked_bits'].notna()]
    cats = ['combination', 'none', 'numeric', 'symbol', 'uppercase']
    cat_colour = {'combination': '#D55E00', 'none': '#0072B2', 'numeric': '#009E73',
                  'symbol': '#CC79A7', 'uppercase': '#56B4E9'}
    cat_marker = {'combination': 'D', 'none': 'o', 'numeric': 's',
                  'symbol': '^', 'uppercase': 'v'}

    fig, axes = plt.subplots(1, 4, figsize=(6.6, 3.4), sharey=True)
    for ax, m in zip(axes, MECHS):
        s = sub[sub['mechanism'] == m].set_index('substitution_category')
        for c in cats:
            if c not in s.index:
                continue
            h, l = s.loc[c, 'human_bits'], s.loc[c, 'leaked_bits']
            lw = 2.2 if c == 'combination' else 1.1
            ax.plot([0, 1], [h, l], color=cat_colour[c], linewidth=lw,
                    marker=cat_marker[c], markersize=5, markerfacecolor='white',
                    markeredgewidth=1.3, zorder=3 if c == 'combination' else 2)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['human', '2-week\nleak'])
        ax.set_title(LABEL[m], loc='left', fontsize=8)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Marginal security (bits)')

    handles = [plt.Line2D([], [], color=cat_colour[c], marker=cat_marker[c],
                          markerfacecolor='white', markeredgewidth=1.3,
                          linewidth=2.2 if c == 'combination' else 1.1, label=c)
               for c in cats]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Composition policies change rank between the two adversaries; on M2 to M4,\n'
                 'combination is worst against a human and best against a leak',
                 fontsize=9, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, 'fig4_composition')


# ------------------------------------------------------------------ fig 5

def fig_coverage(pol, cov):
    """Attack success against how much of the target the adversary can cover."""
    cg = cov.groupby('substitution_category')['digraph_coverage'].mean()
    sub = pol[(pol['grouping'] == 'substitution_category') & pol['leaked_fmr'].notna()]

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for m in MECHS:
        s = sub[sub['mechanism'] == m].set_index('substitution_category')
        xs = [cg[c] for c in s.index]
        ys = list(s['leaked_fmr'])
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], color=COLOUR[m],
                marker=MARKER[m], linestyle=LINE[m], label=LABEL[m],
                markerfacecolor='white', markeredgewidth=1.4, alpha=0.95)

    for c in cg.index:
        ax.annotate(c, (cg[c], 0.12), fontsize=7, rotation=90,
                    ha='center', va='bottom', color='0.4')
    ax.set_xlim(0.33, 0.67)

    ax.set_xlabel('Proportion of the target password\'s digraphs\n'
                  'covered by the victim\'s other typing')
    ax.set_ylabel('Attack success (false match rate)')
    ax.set_ylim(0.08, 1.02)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32),
              ncol=2, frameon=False)
    ax.set_title('Attack success against digraph coverage,\n'
                 'two-week leak', loc='left')
    fig.tight_layout()
    save(fig, 'fig5_coverage')


def main():
    comp, retry, pol, cov = load()
    print('writing figures:')
    fig_ladder(comp)
    fig_ageing(comp)
    fig_retry(retry)
    fig_policy(pol)
    fig_coverage(pol, cov)
    print(f'\nfigures -> {FIG_DIR}')
    print(f'greyscale proofs -> {PROOF_DIR}')


if __name__ == '__main__':
    main()
