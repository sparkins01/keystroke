"""
Figure for the fixed-enrolment-threshold analysis (reviewer 2, point 7).

Left panel: half-total error rate against temporal gap, with the threshold
frozen at enrolment and with it recalibrated at each gap. The gap between the
two lines is the cost of the calibration convention itself.

Right panel: the same frozen-threshold trials split into their two error
types. This is the substantive finding. Under a frozen threshold the false
match rate does not rise, so the system does not become less secure as the
template ages. The false non-match rate rises steeply instead, so it becomes
progressively unusable.

Markers and line styles carry the same information as colour, so the figure
survives greyscale printing.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_ROOT = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
RESULT_DIR = os.path.join(RESULTS_ROOT, 'revision')
OUT_DIR = os.environ.get('KD_FIG_DIR',
                         os.path.join(RESULTS_ROOT, 'figures'))

MECH_COLORS = {
    'M1_Manhattan': '#1f77b4', 'M2_ShallowML': '#2ca02c',
    'M3_TypeNet': '#9467bd', 'M4_TypeFormer': '#e377c2',
    'M5_ITAD': '#000000',
}
MECH_LABELS = {
    'M1_Manhattan': 'M1 Manhattan', 'M2_ShallowML': 'M2 Shallow ML',
    'M3_TypeNet': 'M3 TypeNet (LSTM)', 'M4_TypeFormer': 'M4 TypeFormer',
    'M5_ITAD': 'M5 ITAD',
}
MECH_MARKERS = {
    'M1_Manhattan': 'o', 'M2_ShallowML': 's',
    'M3_TypeNet': '^', 'M4_TypeFormer': 'D',
    'M5_ITAD': 'v',
}
ORDER = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer',
         'M5_ITAD']


def main():
    df = pd.read_csv(os.path.join(RESULT_DIR, 'fixed_threshold.csv'))
    eer = df[df['operating_point'] == 'eer']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    for mech in ORDER:
        g = eer[eer['mechanism'] == mech].sort_values('delta')
        ax.plot(g['delta'], g['hter_frozen_mean'] * 100,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                linestyle='-', markersize=4, label=MECH_LABELS[mech])
        ax.plot(g['delta'], g['hter_recalibrated_mean'] * 100,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                linestyle='--', markersize=4, alpha=0.55)
    ax.set_xlabel(r'Temporal gap $\Delta$ (weeks)')
    ax.set_ylabel('Half-total error rate (%)')
    ax.set_title('Frozen (solid) vs recalibrated (dashed)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for mech in ORDER:
        g = eer[eer['mechanism'] == mech].sort_values('delta')
        ax.plot(g['delta'], g['fnmr_frozen_mean'] * 100,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                linestyle='-', markersize=4, label=MECH_LABELS[mech])
        ax.plot(g['delta'], g['fmr_frozen_mean'] * 100,
                marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                linestyle=':', markersize=4, alpha=0.55)
    ax.set_xlabel(r'Temporal gap $\Delta$ (weeks)')
    ax.set_ylabel('Error rate (%)')
    ax.set_title('Frozen threshold: FNMR (solid) vs FMR (dotted)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fixed_threshold.png')
    fig.savefig(out, dpi=200)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
