"""
Why does M3 age more slowly? A decomposition of the ageing curve.

Reviewer 1 asked for a mechanism, not a rejected hypothesis. The EER at a
given gap is set by how far the genuine and impostor score distributions sit
apart relative to their width. Ageing is the genuine distribution sliding
towards the impostor one. Two mechanisms can therefore show different EER
slopes for two quite different reasons:

  (a) the genuine distribution slides less, meaning the representation is
      genuinely more stable over time; or
  (b) the distributions are wide, so the same slide costs less accuracy.

These are not the same claim, and only (a) is a property worth having. The
floor-effect account tested earlier was about the error *level*; this is
about the geometry that produces it.

Scores are on arbitrary and wildly different scales across mechanisms, so
everything here is made scale-free before comparison. The impostor
distribution at the shortest gap is used as the ruler, since impostor
comparisons involve two different people and have no reason to drift with
the gap in the way genuine comparisons do.

Reported per mechanism:
  d_prime(gap)      standardised separation of the two distributions
  drift             how far the genuine mean moves per week, in units of the
                    impostor spread
  spread_ratio      genuine spread over impostor spread, at the shortest gap
  impostor_drift    the same drift measure applied to impostor scores, as a
                    control: it should be small if the ruler is stable

Writes results/revision/why_m3_*.csv and results/figures/ageing_decomposition.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
ITAD_DIR = os.path.join(RESULTS, 'with_itad')
OUT_DIR = os.path.join(RESULTS, 'revision')
FIG_DIR = os.environ.get('KD_FIG_DIR', os.path.join(RESULTS, 'figures'))

COLS = ['mechanism', 'fold', 'delta', 'label', 'score']
ORDER = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer', 'M5_ITAD']
LABEL = {
    'M1_Manhattan': 'M1 Manhattan', 'M2_ShallowML': 'M2 shallow ML',
    'M3_TypeNet': 'M3 TypeNet', 'M4_TypeFormer': 'M4 TypeFormer',
    'M5_ITAD': 'M5 ITAD',
}
COLOUR = {
    'M1_Manhattan': '#0072B2', 'M2_ShallowML': '#009E73',
    'M3_TypeNet': '#CC79A7', 'M4_TypeFormer': '#E69F00',
    'M5_ITAD': '#000000',
}
MARKER = {'M1_Manhattan': 'o', 'M2_ShallowML': 's', 'M3_TypeNet': '^',
          'M4_TypeFormer': 'D', 'M5_ITAD': 'v'}


def load():
    main = pq.read_table(
        os.path.join(RESULTS, 'trials_test.parquet'), columns=COLS,
        filters=[('k_enrollment', '=', 5), ('clipped', '=', 'clipped')]).to_pandas()
    itad_path = os.path.join(ITAD_DIR, 'itad_trials_test.parquet')
    if os.path.exists(itad_path):
        itad = pq.read_table(itad_path, columns=COLS,
                             filters=[('k_enrollment', '=', 5)]).to_pandas()
        main = pd.concat([main, itad], ignore_index=True)
    return main[np.isfinite(main['score'])]


def moments(df):
    """Mean and spread of each score distribution, per mechanism, fold and gap."""
    rows = []
    for (mech, fold, delta), g in df.groupby(['mechanism', 'fold', 'delta']):
        gen = g['score'][g['label'] == 1].to_numpy()
        imp = g['score'][g['label'] == 0].to_numpy()
        if len(gen) < 30 or len(imp) < 30:
            continue
        rows.append({
            'mechanism': mech, 'fold': fold, 'delta': int(delta),
            'mu_gen': gen.mean(), 'sd_gen': gen.std(ddof=1),
            'mu_imp': imp.mean(), 'sd_imp': imp.std(ddof=1),
        })
    m = pd.DataFrame(rows)
    pooled = np.sqrt((m['sd_gen'] ** 2 + m['sd_imp'] ** 2) / 2.0)
    m['d_prime'] = (m['mu_gen'] - m['mu_imp']) / pooled
    return m


def decompose(m):
    """Per mechanism and fold, express the drift on a scale-free ruler."""
    rows = []
    for (mech, fold), g in m.groupby(['mechanism', 'fold']):
        g = g.sort_values('delta')
        ref = g[g['delta'] == 0]
        if ref.empty:
            continue
        ruler = ref['sd_imp'].iloc[0]          # impostor spread at gap 0
        if not np.isfinite(ruler) or ruler <= 0:
            continue
        # genuine mean falls as the template ages; express per week on the ruler
        slope_gen = sstats.linregress(g['delta'], g['mu_gen']).slope / ruler
        slope_imp = sstats.linregress(g['delta'], g['mu_imp']).slope / ruler
        slope_dp = sstats.linregress(g['delta'], g['d_prime']).slope
        rows.append({
            'mechanism': mech, 'fold': fold,
            'genuine_drift_per_week': -slope_gen,
            'impostor_drift_per_week': -slope_imp,
            'd_prime_at_0': ref['d_prime'].iloc[0],
            'd_prime_slope': slope_dp,
            'spread_ratio': ref['sd_gen'].iloc[0] / ruler,
        })
    return pd.DataFrame(rows)


def plot(m, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    agg = m.groupby(['mechanism', 'delta'])[['d_prime']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    for mech in ORDER:
        g = agg[agg['mechanism'] == mech].sort_values('delta')
        if g.empty:
            continue
        ax.plot(g['delta'], g['d_prime'], marker=MARKER[mech],
                color=COLOUR[mech], markersize=5, label=LABEL[mech])
    ax.set_xlabel(r'Enrolment-to-query gap $\Delta$ (weeks)')
    ax.set_ylabel("Standardised separation $d'$")
    ax.set_title('Separation of genuine and impostor scores', fontsize=10)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(fontsize=8)

    ax = axes[1]
    norm = agg.copy()
    for mech in ORDER:
        g = norm[norm['mechanism'] == mech].sort_values('delta')
        if g.empty:
            continue
        base = g['d_prime'].iloc[0]
        ax.plot(g['delta'], 100 * g['d_prime'] / base, marker=MARKER[mech],
                color=COLOUR[mech], markersize=5, label=LABEL[mech])
    ax.axhline(100, color='black', linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r'Enrolment-to-query gap $\Delta$ (weeks)')
    ax.set_ylabel("$d'$ as a percentage of its value at $\\Delta$=0")
    ax.set_title('Proportional loss of separation', fontsize=10)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print('wrote %s' % out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    print('trials: %d across %d mechanisms' % (len(df), df['mechanism'].nunique()))

    m = moments(df)
    d = decompose(m)

    print()
    print('=' * 84)
    print('DECOMPOSITION, MEAN ACROSS FOLDS')
    print('=' * 84)
    summary = d.groupby('mechanism')[
        ['d_prime_at_0', 'd_prime_slope', 'genuine_drift_per_week',
         'impostor_drift_per_week', 'spread_ratio']].mean()
    summary['d_prime_loss_pct_per_week'] = (
        -100 * summary['d_prime_slope'] / summary['d_prime_at_0'])
    summary = summary.reindex([x for x in ORDER if x in summary.index])
    print(summary.to_string(float_format=lambda v: '%.4f' % v))

    print()
    print('Reading: genuine_drift_per_week is how far the genuine mean moves')
    print('towards the impostors each week, measured in impostor standard')
    print('deviations. impostor_drift_per_week is the control and should be')
    print('small. spread_ratio is how wide the genuine distribution is')
    print('relative to the impostor one at the shortest gap.')

    m.to_csv(os.path.join(OUT_DIR, 'why_m3_moments.csv'), index=False)
    d.to_csv(os.path.join(OUT_DIR, 'why_m3_decomposition.csv'), index=False)
    summary.to_csv(os.path.join(OUT_DIR, 'why_m3_summary.csv'))
    plot(m, os.path.join(FIG_DIR, 'ageing_decomposition.png'))


if __name__ == '__main__':
    main()
