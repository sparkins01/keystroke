# Adversary Analysis

Code and reported results accompanying *"The Marginal Security of Keystroke
Dynamics as a Second Authentication Factor Against a Password-Knowing
Adversary"*.

S. Parkinson (Manchester Metropolitan University) and S. Khan (University of
Huddersfield).

This work uses the **same dataset** as the template ageing study in the
repository root. Nothing here replaces or modifies that material.

## What this adds

The ageing study measured how verification error grows as a template ages,
against conventional zero-effort impostors. This study asks a different
question: how much security the biometric delivers once the adversary already
holds the password, which is the only situation in which a second factor
decides access.

Five adversary levels are evaluated against four matching mechanisms, at fixed
usability budgets:

| Level | Adversary |
|---|---|
| A0 | Unpractised human, typing the password for the first time |
| A1 | Practised human, after about 28 repetitions |
| A2 | Injected forgery built from population statistics |
| A3 | Injected forgery rescaled to the victim's overall typing speed |
| A4 | Injected forgery built from the victim's timings on other passwords |

A2 to A4 assume that the adversary can inject timings into the authentication
client. The results for those levels apply only where that is possible.

## Contents

| Path | Description |
|---|---|
| `code/` | Analysis pipeline for this study |
| `results/` | The summary tables behind the figures and tables in the paper |

`code/` is self-contained. The nine shared modules (`data_loading.py`,
`password_metadata.py`, `features.py`, `raw_sequences.py`, `dataset.py`,
`splits.py`, `metrics.py`, `embedding_models.py`, `matchers_classical.py`) are
copies of the versions in the repository root, so this directory runs on its
own without reaching outside itself.

## Running it

```bash
pip install -r ../requirements.txt
cd code
```

Then run the scripts in this order, since later scripts read what earlier ones
write.

| Order | Script | What it does |
|---|---|---|
| 1 | `exp1_practice.py`, `exp1b_convergence.py` | Adversary practice, and the ageing decomposition |
| 2 | `exp2_transfer.py` | Forgeries and M1 scores for the synthetic levels |
| 3 | `exp3_m4.py`, `exp4_ladder.py` | The same on M4, then on M2 and M3 |
| 4 | `exp3_analyse.py` | Attack success across the adversary levels |
| 5 | `exp5_policy.py`, `exp5b_coverage.py` | Password composition and digraph coverage, full leak |
| 6 | `exp6_model.py` | Statistical model, full leak |
| 7 | `exp7_cmu_calibration.py` | Calibration against a published benchmark |
| 8 | `exp8_retry_empirical.py` | Multi-attempt success from distinct attempts |
| 9 | `exp9_intervals.py` | Bootstrap intervals at every usability budget |
| 10 | `exp10_coverage_model.py` | Digraph coverage at the victim and password level |
| 11 | `exp11_leak_budget.py` | The size of each leak in samples, keystrokes and digraphs |
| 12 | `exp12_naturalness.py` | Evaluation of the naturalness defence |
| 13 | `exp14_rescore.py` | Scores for within-session retries and for leaks that precede the attack |
| 14 | `exp15_two_week.py` | The composition, model and coverage analyses on the two-week leak |
| 15 | `exp16_robustness.py` | Marginal effects, dependence between trials, threshold calibration |
| 16 | `exp17_rescore_analyse.py` | Analysis of the scores from `exp14_rescore.py` |
| 17 | `generate_figures.py` | The figures, written to `figures/` |

There is no experiment 13. `lopo.py` holds the threshold procedure used by every
analysis: leave-one-participant-out within each fold, calibrated separately at
each enrolment-to-query gap.

Paths default to `../../data` for the dataset and `../results` for outputs, so a
fresh clone runs with no configuration. Override them with `KD_DATA_DIR`,
`KD_ADVERSARY_RESULT_DIR` and `KD_ADVERSARY_FIG_DIR`.

Scripts that train the embedding models (`exp3_m4.py`, `exp4_ladder.py`) take
roughly 40 minutes each on a consumer GPU and cache the trained encoders.
`exp14_rescore.py` reuses those encoders.

## Where each result comes from

| In the paper | File in `results/` |
|---|---|
| Table 1 | `exp7_summary.txt`, `exp7_cmu_calibration.csv` |
| Tables 2 and 3, Figures 2 and 3 | `exp3_comparison.csv` |
| RQ2 | `exp1_summary.txt`, `exp1b_summary.txt` |
| Tables 4 and 6 | `exp9_intervals.csv` |
| Table 5, Figure 4 | `exp8_retry.csv`, `exp8_retry_curve.csv` |
| Table 7, Figure 5 | `exp5_policy_early.csv`, `exp15_summary.txt` |
| Table 8, Figure 6 | `exp5b_coverage_early.csv`, `exp15_summary.txt` |
| Statistical model, two-week leak | `exp15_summary.txt` |
| Statistical model, full leak | `exp6_summary.txt` |
| Coverage model | `exp15_summary.txt`, and `exp10_summary.txt` for the full leak |
| Leak budget | `exp11_summary.txt` |
| Marginal effects, dependence, thresholds | `exp16_summary.txt`, `exp16_thresholds.csv` |
| Table 9, within-session retries | `exp17_summary.txt`, `exp17_causal.csv`, `exp17_session_retry.csv` |
| Table 10 | `exp12_summary.txt`, `exp12_detection.csv` |

Figure 1 is a diagram drawn in the paper's source. `generate_figures.py` writes
`fig1` to `fig5`, which are Figures 2 to 6 in the paper.

`exp1_security_table.csv`, `exp2_attack_table.csv`, `exp2_summary.txt` and
`exp6_retry.csv` come from earlier stages of the pipeline and are not cited in
the paper. Use the files above for the reported figures.

## Labels used in the code and results

| Label | Meaning |
|---|---|
| `A2_sampled` | A2 in the paper |
| `A4_global_sampled_*` | A3 in the paper |
| `A4_keyed_sampled_*` | A4 in the paper |
| `*_early` | The two-week leak, weeks 1 and 2 |
| `*_full` | The full leak, weeks 1 to 8 |
| `*_median_*`, `A2_population` | The median form, reported only in Table 3 |
| `M1` to `M4`, `Manhattan`, `ShallowML`, `TypeNet`, `TypeFormer` | M1 distance, M2 classifier, M3 recurrent, M4 transformer |
| paper 1 | The template ageing study in the repository root |
| R0 | The original submission of this paper |
| R1 | The revised submission |

Comments that mention a reviewer point to the review comment that led to the
analysis.

## One external dataset is not included

`exp7_cmu_calibration.py` calibrates the classical matcher against the benchmark
of Killourhy and Maxion (2009). That dataset is theirs to distribute and is not
redistributed here. Download `DSL-StrongPasswordData.csv` from
<https://www.cs.cmu.edu/~keystroke/> and either place it at
`adversary_analysis/data_cmu/DSL-StrongPasswordData.csv` or point `KD_CMU_CSV`
at it.

## What is not in `results/`

The summary tables are included. The intermediate trial scores are not: they run
to several hundred megabytes of Parquet, and every one is regenerated by the
script that produced it. The sample caches and trained model weights are likewise
omitted for size and are rebuilt on first run.

## Privacy

The dataset in the repository root is pseudonymised. Participant names were
removed and replaced with identifiers, and the re-identification key is not
distributed. The code here reads the pseudonymised files directly and never
requires a name.
