import os
RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

def eer_from_scores(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fnr[idx] + fpr[idx]) / 2

df = pd.read_parquet(os.path.join(RESULT_DIR, 'trials_test.parquet'))
df = df[(df['clipped'] == 'clipped') & (df['k_enrollment'] == 5)]

mechs = ['M1_Manhattan', 'M2_ShallowML', 'M3_TypeNet', 'M4_TypeFormer']

print(f"{'Mechanism':<15}{'Delta':>6}{'Pooled EER':>12}{'ParticBal EER':>15}{'n_particip':>12}")
results = []
for mech in mechs:
    sub = df[df['mechanism'] == mech]
    for delta in range(8):
        dsub = sub[sub['delta'] == delta]
        pooled = eer_from_scores(dsub['label'].values, dsub['score'].values) * 100

        # participant-balanced: compute EER per genuine-participant (using that
        # participant's own genuine trials + the full impostor pool of that cell),
        # then average unweighted across participants. A participant here is
        # identified via their genuine trials; impostor trials are shared pool
        # for threshold purposes so we instead do a per-participant *contribution*
        # balanced version: EER computed once per participant using only trials
        # where that participant is the genuine claimant, pooling impostor trials
        # from the same delta/mechanism cell as the negative class.
        per_p_eers = []
        genuine_participants = dsub[dsub['label'] == 1]['participant_id'].unique()
        impostor_scores_pool = dsub[dsub['label'] == 0]
        for p in genuine_participants:
            p_genuine = dsub[(dsub['label'] == 1) & (dsub['participant_id'] == p)]
            if len(p_genuine) == 0:
                continue
            combined_labels = np.concatenate([np.ones(len(p_genuine)), np.zeros(len(impostor_scores_pool))])
            combined_scores = np.concatenate([p_genuine['score'].values, impostor_scores_pool['score'].values])
            e = eer_from_scores(combined_labels, combined_scores)
            if not np.isnan(e):
                per_p_eers.append(e)
        particip_balanced = np.mean(per_p_eers) * 100 if per_p_eers else np.nan
        results.append({'mechanism': mech, 'delta': delta, 'pooled_eer': pooled,
                         'particip_balanced_eer': particip_balanced, 'n_participants': len(per_p_eers)})
        print(f"{mech:<15}{delta:>6}{pooled:>12.1f}{particip_balanced:>15.1f}{len(per_p_eers):>12}")

out = pd.DataFrame(results)
out.to_csv(os.path.join(RESULT_DIR, 'stats_model', 'participant_balanced_eer.csv'), index=False)
print("\nSaved to Result/stats_model/participant_balanced_eer.csv")

print("\n=== Summary: pooled vs participant-balanced at Delta=0 and Delta=7 ===")
for mech in mechs:
    r0 = out[(out.mechanism==mech) & (out.delta==0)].iloc[0]
    r7 = out[(out.mechanism==mech) & (out.delta==7)].iloc[0]
    print(f"{mech}: D0 pooled={r0.pooled_eer:.1f} balanced={r0.particip_balanced_eer:.1f} | D7 pooled={r7.pooled_eer:.1f} balanced={r7.particip_balanced_eer:.1f}")
