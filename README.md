# Longitudinal Fixed-Text Keystroke Dynamics Dataset

Event-level keystroke timing data accompanying *"Template Ageing and
Longitudinal Verification in Fixed-Text Keystroke Dynamics: A Subject-Disjoint
Study Across Eight Weeks"*.

S. Parkinson, S. Khan, N. Liu (University of Huddersfield) and Q. Xu (Tianjin
University).

## Contents

| File | Description |
|---|---|
| `data/week_1.csv.gz` … `week_8.csv.gz` | Event-level key timings, one file per collection week |
| `data/password_metadata.csv` | The 40 passwords, with length and substitution category |
| `data/sessions_summary.csv` | Per-participant, per-week session accounting |
| `data/skipped_sessions.csv` | Sessions excluded during cleaning, with reasons |
| `data/pairing_notes.csv` | Notes on timing/accuracy file pairing |

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
