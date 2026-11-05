# Longitudinal Fixed-Text Keystroke Dynamics Dataset

Event-level keystroke timing data accompanying *"Template Ageing and
Longitudinal Verification in Fixed-Text Keystroke Dynamics: A Subject-Disjoint
Study Across Eight Weeks"*.

S. Parkinson, S. Khan, N. Liu (University of Huddersfield) and Q. Xu (Tianjin
University).

A second study reuses this dataset without modifying it. Its code and results
are in [`paper2/`](paper2/), and everything described below is unchanged by it.

## Contents

| File | Description |
|---|---|
| `data/week_1.csv.gz` … `week_8.csv.gz` | Event-level key timings, one file per collection week |
| `data/password_metadata.csv` | The 40 passwords, with length and substitution category |
| `data/sessions_summary.csv` | Per-participant, per-week session accounting |
| `data/skipped_sessions.csv` | Sessions excluded during cleaning, with reasons |
| `data/pairing_notes.csv` | Notes on timing/accuracy file pairing |
| `code/` | The full analysis pipeline |
| `requirements.txt` | Pinned package versions used for the published results |
| `paper2/` | Code and results for a second study using the same dataset |

## Running the analysis

```bash
pip install -r requirements.txt
bash code/run.sh
```

Paths default to `./data` and `./results` relative to the repository root, so a
fresh clone runs with no configuration. Figures are written to
`./results/figures`. Set `KD_DATA_DIR`, `KD_RESULT_DIR` or `KD_FIG_DIR` to put
any of the three somewhere else.

`run.sh` has two modes. The default is a **quick verification run** (~10
minutes): one fold, one training seed, a reduced training budget and
sub-sampled impostors. It exercises every stage of the pipeline, M5 included,
but the absolute error rates it produces will *not* match the paper.

For the **full reproduction** (~20–30 h on CPU):

```bash
REPRODUCTION_MODE=full KD_DEVICE=cuda bash code/run.sh
```

That runs the primary analysis, both independent end-to-end replicates, the
training-seed variance decomposition, the logistic sensitivity check, the
participant-balanced EER check, and the analyses listed under *Revision
analyses* below.

The published results were produced on a single consumer-grade GPU, at roughly
2.8 h per replicate. The two password-disjoint runs retrain M3 and M4, so they
cost about as much again.

### Pipeline

| Stage | Module |
|---|---|
| Raw log parsing (raw logs not published, see below) | `parse_keystrokes.py` |
| Loading and backspace replay | `data_loading.py` |
| Password corpus recovery | `password_metadata.py` |
| Feature extraction, IQR clipping | `features.py`, `raw_sequences.py` |
| Subject-disjoint folds | `splits.py` |
| M1 scaled Manhattan, M2 gradient boosting | `matchers_classical.py` |
| M3 TypeNet-style LSTM, M4 TypeFormer-style Transformer | `embedding_models.py` |
| M5 ITAD, instance-based tail-area density | `matcher_itad.py` |
| Genuine/impostor trial generation | `protocol.py` |
| EER, FNMR, DET, bootstrap CIs | `metrics.py` |
| Linear mixed-effects model | `stats_model.py` |
| Logistic sensitivity checks | `glmm_logit_clusterrobust.py`, `glmm_sensitivity_check.py` |
| Training-seed variance decomposition | `seed_robustness_check.py` |
| Participant-balanced EER check | `participant_balanced_eer.py` |
| Orchestration | `run_experiments.py` |

### Revision analyses

Added during revision and run by `REPRODUCTION_MODE=full`. Each writes to
`results/revision/`, except the M5 grid, which writes to `results/with_itad/`
in the schema of the main run, so the published tables, figures and model read
it without modification.

| Analysis | Paper section | Module |
|---|---|---|
| M5 benchmarked on its own | Baselines for Comparison | `exp_itad.py` |
| M5 scored across the full grid | Accuracy and Template Ageing | `exp_itad_integrate.py` |
| M5 rows for the appendix tables | Ablations | `itad_appendix_rows.py` |
| Mixed-effects model over all five mechanisms | Mixed-Effects Model | `refit_model_with_itad.py` |
| Dwell against the digraph timings, and the provenance of the dwell channel | (supports the M5 feature weighting) | `exp_feature_ablation.py`, `exp_dwell_provenance.py` |
| Password length and enrolment week as model terms | Mixed-Effects Model | `exp_model_extensions.py`, `exp_model_robustness.py`, `summarise_model_extensions.py` |
| Length by gap, within each mechanism | Mixed-Effects Model | `exp_length_by_mechanism.py` |
| Familiarisation against template ageing | Learning Effects and Template Ageing | `exp_familiarisation.py` |
| A threshold fixed at enrolment | A Threshold Fixed at Enrolment | `exp_fixed_threshold.py`, `plot_fixed_threshold.py` |
| Passwords withheld from training as well as participants | Unseen Passwords | `exp_password_disjoint.py`, `analyse_password_disjoint.py`, `estimate_memorisation.py` |
| Why M3 ages more slowly | Mechanism-Dependent Ageing Rates | `exp_why_m3_ages_slowly.py`, `exp_m3_slope_prediction.py`, `exp_floor_effect.py` |
| The published figures | – | `regenerate_bw_figures.py`, `make_single_column_figures.py` |

The two password-disjoint runs are a pair. The swapped run is what separates
intrinsic difficulty between the two password halves from the cost of a
password the model never trained on, so `estimate_memorisation.py` needs both.

### Which version matches the paper

Tags mark the state behind each version of the paper, so a reader can retrieve
the exact code rather than infer it from the current head.

| Tag | Paper version |
|---|---|
| `paper1-template-ageing` | first submission |
| `paper1-template-ageing-r1` | revised submission, M1 to M5 |

### Determinism

Fold assignment and triplet sampling are seeded, so a given `--seed` reproduces
its own fold split deterministically. Neural network training is **not**
bit-reproducible across different hardware or BLAS backends. This matters more
here than usual: the paper's finding is precisely that mechanism comparisons are
sensitive to training-run randomness, with 58% of the recurrent model's
fold-to-fold variance attributable to seed noise. Expect the qualitative pattern
to reproduce, with small numerical differences in the third significant figure.

## The keystroke data

85 participants × 40 passwords × 4 repetitions × 8 weekly sessions, totalling
**1,431,011 key events**. Loading all eight files and applying the cleaning
described in the paper yields **94,894 clean samples from 99,607 raw attempts**.

Columns:

| Column | Meaning |
|---|---|
| `week` | Collection week, 1–8 |
| `participant_id` | Pseudonymous integer identifier, stable across weeks |
| `session_date` | Date of the collection session |
| `session_num` | Session index within that week |
| `phrase_index` | Which of the 40 passwords, in that participant's presentation order |
| `phrase_text` | The password as presented |
| `repetition` | Repetition number, 1–4 |
| `key_seq` | Position of this key event within the attempt |
| `key` | The key pressed |
| `press_time` | Key-down timestamp (ISO 8601, microsecond precision) |
| `elapsed_seconds` | Seconds since the previous event |
| `release_time` | Key-up timestamp |

Dwell, flight/digraph and trigraph features are all derivable from
`press_time` and `release_time`.

Note that the presentation *order* of the 40 passwords is shuffled per
participant and stable across that participant's eight weeks, so
`phrase_index` is not comparable between participants. Join on `phrase_text`
(or its alphabetic skeleton) instead.

## Pseudonymisation

The source collection identified participants by full name. This release does
not. Specifically:

- the `participant_name` column has been removed from every file;
- participants appear only as the arbitrary integer `participant_id`;
- the file mapping `participant_id` to real names is **not** included in this
  repository, and must not be added to it;
- free-text fields that embedded participant names inside filesystem paths have
  been reduced to bare filenames.

This was verified by scanning every file against all name tokens from the
source records. No match remains.

The raw collection tree is **not** published, because its directory names
encode participant names.

## Permitted use

The data were collected under University of Huddersfield ethical approval, as
described in the source study (Parkinson et al., *IET Biometrics*, 12(1),
2023). They are made available to support verification and replication of the
analyses in the accompanying paper. **Attempting to re-identify participants is
not a permitted use.**

## Citing

If you use this data, please cite the accompanying paper. Please also cite the
original collection study:

> S. Parkinson, S. Khan, A.-M. Badea, A. Crampton, N. Liu and Q. Xu, "An
> Empirical Analysis of Keystroke Dynamics in Passwords: A Longitudinal Study,"
> *IET Biometrics*, vol. 12, no. 1, pp. 25–37, 2023. doi:10.1049/bme2.12087
