"""
Does the EER-versus-separation curve account for M3's shallower EER slope?

The decomposition in exp_why_m3_ages_slowly.py shows that M3's representation
is not unusually stable. Measured in impostor standard deviations, its
genuine distribution drifts at about M1's rate, and it loses a slightly
larger share of its separation per week than M1 or M2 do.

That raises a specific possibility. EER is a compressive function of
separation. Where separation is already small, as it is for M3, a given loss
of separation moves the EER less than it would for a mechanism that starts
well separated. M3's shallower EER slope could therefore be a property of
where it sits on that curve rather than of recurrence.

This is a sharper claim than the floor-effect account tested in Section V-A,
and it is testable. Under a Gaussian approximation the two are related by

    EER = Phi(-d_prime / 2)

so each mechanism's measured d_prime trajectory implies an EER trajectory.
If the implied EER slopes reproduce the observed ordering, the curve
explains the effect. If M3's observed slope is shallower than its own
separation implies, something else is going on.

The Gaussian form is an approximation, so the test is whether the ordering
and rough magnitudes are reproduced, not whether the values match exactly.

Writes results/revision/m3_slope_prediction.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

RESULTS = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(RESULTS, 'revision')
ORDER = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer', 'M5_ITAD']


def main():
    m = pd.read_csv(os.path.join(OUT_DIR, 'why_m3_moments.csv'))

    observed = pd.read_csv(os.path.join(RESULTS, 'with_itad',
                                        'eer_by_mechanism_delta.csv'),
                           keep_default_na=False, na_values=[''])
    observed = observed[(observed['k_enrollment'] == 5)
                        & (observed['clipped'].isin(['clipped', 'not_applicable']))]
    obs = observed.pivot(index='delta', columns='mechanism', values='eer_pooled')

    rows = []
    for mech, g in m.groupby('mechanism'):
        per_delta = g.groupby('delta')['d_prime'].mean().sort_index()
        implied = sstats.norm.cdf(-per_delta.values / 2.0)
        deltas = per_delta.index.to_numpy(float)

        implied_slope = sstats.linregress(deltas, implied).slope * 100
        if mech not in obs.columns:
            continue
        observed_slope = sstats.linregress(obs.index.to_numpy(float),
                                           obs[mech].to_numpy()).slope * 100
        rows.append({
            'mechanism': mech,
            'd_prime_at_0': per_delta.iloc[0],
            'implied_eer_at_0': implied[0] * 100,
            'observed_eer_at_0': obs[mech].iloc[0] * 100,
            'implied_slope': implied_slope,
            'observed_slope': observed_slope,
            'observed_minus_implied': observed_slope - implied_slope,
        })

    out = pd.DataFrame(rows).set_index('mechanism')
    out = out.reindex([x for x in ORDER if x in out.index])

    print('=' * 92)
    print('EER SLOPE: OBSERVED AGAINST WHAT THE SEPARATION CURVE IMPLIES')
    print('=' * 92)
    print(out.to_string(float_format=lambda v: '%.3f' % v))
    print()
    print('Slopes are EER points per week.')
    print()

    obs_rank = list(out['observed_slope'].sort_values().index)
    imp_rank = list(out['implied_slope'].sort_values().index)
    print('observed slope, shallowest first: %s' % [x.split('_')[0] for x in obs_rank])
    print('implied  slope, shallowest first: %s' % [x.split('_')[0] for x in imp_rank])
    print('ordering reproduced: %s' % (obs_rank == imp_rank))
    print()

    r, p = sstats.pearsonr(out['implied_slope'], out['observed_slope'])
    print('correlation between implied and observed slope: r = %.3f (p = %.3g)' % (r, p))
    print()
    m3 = out.loc['M3_TypeNet']
    print('M3 specifically:')
    print('  observed slope %.3f, implied %.3f, difference %+.3f'
          % (m3['observed_slope'], m3['implied_slope'], m3['observed_minus_implied']))
    others = out.drop('M3_TypeNet')['observed_minus_implied']
    print('  other mechanisms differ from their implied slope by %.3f to %.3f'
          % (others.min(), others.max()))
    print()
    if abs(m3['observed_minus_implied']) <= max(abs(others.min()), abs(others.max())):
        print('M3 is not an outlier against its own implied slope. The curve')
        print('accounts for its shallower EER slope as well as it accounts for')
        print('the others.')
    else:
        print('M3 departs from its implied slope by more than any other')
        print('mechanism, so the curve does not fully account for it.')

    out.to_csv(os.path.join(OUT_DIR, 'm3_slope_prediction.csv'))


if __name__ == '__main__':
    main()
