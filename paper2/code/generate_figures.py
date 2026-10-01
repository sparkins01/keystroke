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

RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FIG_DIR = os.environ.get('KD_P2_FIG_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures'))
PROOF_DIR = os.path.join(FIG_DIR, 'greyscale_proofs')

MECHS = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer']
LABEL = {'M1_Manhattan': 'M1 Manhattan', 'M2_ShallowML': 'M2 Shallow ML',
         'M3_TypeNet': 'M3 TypeNet', 'M4_TypeFormer': 'M4 TypeFormer'}

# Okabe-Ito, same assignment as the template ageing paper
COLOUR = {'M1_Manhattan': '#0072B2', 'M2_ShallowML': '#009E73',
          'M3_TypeNet': '#CC79A7', 'M4_TypeFormer': '#E69F00'}
MARKER = {'M1_Manhattan': 'o', 'M2_ShallowML': 's',
          'M3_TypeNet': '^', 'M4_TypeFormer': 'D'}
LINE = {'M1_Manhattan': '-', 'M2_ShallowML': '--',
        'M3_TypeNet': '-.', 'M4_TypeFormer': (0, (3, 1, 1, 1))}

ADV_COLOUR = {'A1': '#0072B2', 'A2': '#009E73', 'A4': '#D55E00'}
ADV_MARKER = {'A1': 'o', 'A2': 's', 'A4': 'D'}
ADV_LINE = {'A1': '-', 'A2': '--', 'A4': '-.'}

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
            pd.read_csv(os.path.join(RESULT_DIR, 'exp6_retry.csv')),
            pd.read_csv(os.path.join(RESULT_DIR, 'exp5_policy.csv')),
            pd.read_csv(os.path.join(RESULT_DIR, 'exp5b_coverage.csv')))


# ------------------------------------------------------------------ fig 1

def fig_ladder(comp):
    """Attack success across the adversary model. A line rather than grouped
    bars, because the ordering of the adversary levels is the finding."""
    rungs = [('A0_unpractised', 'A0\nunpractised\nhuman'),
             ('A1_practised', 'A1\npractised\nhuman'),
             ('A2_sampled', 'A2\npopulation\nstatistics'),
             ('A4_global_sampled_full', 'A3\nvictim\nspeed'),
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
    rungs = [('A1_practised', 'A1'), ('A2_sampled', 'A2'),
             ('A4_keyed_sampled_full', 'A4')]
    names = {'A1': 'A1 practised human', 'A2': 'A2 population statistics',
             'A4': 'A4 victim timings'}

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
    fig.suptitle('One extra random lowercase character is worth 4.70 bits, '
                 'off the top of every panel', fontsize=9, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'fig2_marginal_security')


# ------------------------------------------------------------------ fig 3

def fig_retry(retry):
    """Cumulative attack success against the number of permitted attempts."""
    n = np.arange(1, 11)
    panels = [('A1_human_practised', 'Practised human impostor'),
              ('A4_synth_keyed', 'Adversary holding victim timings')]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    r0 = retry[retry['delta'] == 0]
    for ax, (adv, title) in zip(axes, panels):
        for m in MECHS:
            row = r0[(r0['mechanism'] == m) & (r0['adversary'] == adv)]
            if not len(row):
                continue
            p = float(row['p'].iloc[0])
            ax.plot(n, 1 - (1 - p) ** n, color=COLOUR[m], marker=MARKER[m],
                    linestyle=LINE[m], label=LABEL[m], markerfacecolor='white',
                    markeredgewidth=1.3, markevery=1)
        ax.axvline(3, color='0.4', linewidth=0.9, linestyle=':')
        ax.text(3.15, 0.06, 'typical retry\nlimit', fontsize=7, color='0.35')
        ax.set_title(title, loc='left')
        ax.set_xlabel('Permitted attempts')
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 3, 5, 7, 9])
    axes[0].set_ylabel('Cumulative attack success')
    axes[0].legend(loc='lower right', frameon=False)
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
        ax.set_xticklabels(['human', 'leak'])
        ax.set_title(LABEL[m], loc='left', fontsize=8)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Marginal security (bits)')

    handles = [plt.Line2D([], [], color=cat_colour[c], marker=cat_marker[c],
                          markerfacecolor='white', markeredgewidth=1.3,
                          linewidth=2.2 if c == 'combination' else 1.1, label=c)
               for c in cats]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Composition policies swap rank between the two adversaries;\n'
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
    ax.set_title('The transfer attack succeeds where the adversary\n'
                 'has seen the digraphs before', loc='left')
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
