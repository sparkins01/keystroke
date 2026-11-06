"""
Adversary-aware verification protocol.

Differs from paper 1's `protocol.py` in one structural way. There, the impostor
query was drawn from week `j`, the same week as the genuine query, so growing
the enrolment-to-query gap delta = j - i also advanced the impostor along their
own practice curve. Genuine template drift and impostor skill growth were
therefore entangled in every reported ageing number.

Here the impostor query week `m` is a free variable, independent of `j`. The
impostor score depends only on (victim, password, enrolment week, impostor,
impostor week), never on `j`, so impostor trials are generated once per gallery
rather than once per (gallery, query week). That decouples the two effects and
happens to cut the impostor trial count by roughly the number of query weeks.

Adversary levels realised here (see notes/design.md for the full ladder):
  A0  impostor's first-ever exposure to the password (practice ordinal 0)
  A1  impostor's most practised attempt (week 8, ~28 prior repetitions)
Both are read off the same practice curve rather than generated separately, so
they are guaranteed to be scored against identical galleries and identical
genuine trials.

Impostor samples are the *first repetition of each week*, so every point on the
practice curve sits at the same position within its session. Taking a random
repetition instead would mix within-session warm-up into the between-week
practice effect.
"""
import numpy as np

N_WEEKS = 8
HARDWARE_CHANGE_WEEK = 5

TRIAL_COLUMNS = [
    'fold', 'mechanism', 'clipped', 'k_enrollment',
    'participant_id', 'password_id', 'length', 'substitution_category',
    'enrol_week', 'role', 'query_week', 'delta',
    'imp_participant_id', 'imp_week', 'imp_practice',
    'hardware_change', 'label', 'score',
]


def ordered_sessions(dataset, participant_id, password_id):
    """This participant's samples of this password in the order they were
    actually typed. `Dataset.sessions_for` sorts by (week, repetition) only,
    which mis-orders the handful of participant-weeks holding two sessions."""
    idxs = dataset.by_participant_password.get((participant_id, password_id), [])
    return sorted(idxs, key=lambda i: (dataset.samples[i].week,
                                       dataset.samples[i].session_num,
                                       dataset.samples[i].repetition))


def practice_points(dataset, participant_id, password_id):
    """(sample_idx, week, practice_ordinal) for the first repetition of each
    week this participant typed this password.

    `practice_ordinal` counts how many times they had already typed this exact
    password before this attempt, so ordinal 0 is a true first exposure and is
    the only sample in the corpus that satisfies adversary level A0.
    """
    ordered = ordered_sessions(dataset, participant_id, password_id)
    out = []
    seen_weeks = set()
    for ordinal, idx in enumerate(ordered):
        week = dataset.samples[idx].week
        if week in seen_weeks:
            continue
        seen_weeks.add(week)
        out.append((idx, week, ordinal))
    return out


def build_impostor_bank(dataset, test_participant_ids, password_ids):
    """{(impostor_id, password_id): [(sample_idx, week, practice_ordinal), ...]}

    Built once per fold and reused across every victim, since an impostor's own
    samples do not depend on who they are attacking.
    """
    bank = {}
    for qid in test_participant_ids:
        for pw_id in password_ids:
            pts = practice_points(dataset, qid, pw_id)
            if pts:
                bank[(qid, pw_id)] = pts
    return bank


def generate_trials(dataset, matcher, clipper, test_participant_ids, password_meta,
                    mechanism_name, fold, clipped_label, k_enrollment=5):
    """Genuine and impostor trials for one fold, one mechanism.

    Genuine rows carry `delta`; impostor rows carry `imp_week` / `imp_practice`.
    The two are combined at analysis time, which is what allows any genuine
    delta to be paired with any impostor practice level.
    """
    test_ids = list(test_participant_ids)
    bank = build_impostor_bank(dataset, test_ids, dataset.password_ids)
    rows = []

    for pid in test_ids:
        for pw_id in dataset.password_ids:
            meta = password_meta.get(pw_id)
            if meta is None:
                continue

            sessions_by_week = {w: dataset.sessions_at_week(pid, pw_id, w)
                                for w in range(1, N_WEEKS + 1)}
            weeks_with_data = sorted(w for w, idxs in sessions_by_week.items() if idxs)
            if not weeks_with_data:
                continue

            # the attacking population for this (victim, password): every other
            # test participant who ever typed this password
            impostor_pts = [(qid, bank[(qid, pw_id)]) for qid in test_ids
                            if qid != pid and (qid, pw_id) in bank]
            if not impostor_pts:
                continue

            for i in weeks_with_data:
                pool = dataset.sessions_for(pid, pw_id, max_week=i)
                if not pool:
                    continue

                # reserve one of week i's own sessions so delta=0 has something
                # to query against (paper 1's protocol.py docstring explains why)
                same_week = set(sessions_by_week[i])
                same_week_in_pool = [x for x in pool if x in same_week]
                reserved = same_week_in_pool[-1] if len(same_week_in_pool) >= 2 else None
                candidate_pool = [x for x in pool if x != reserved]
                if not candidate_pool:
                    continue

                enroll_idxs = candidate_pool[-k_enrollment:]
                enroll_set = set(enroll_idxs)
                gallery = matcher.gallery(dataset, clipper, enroll_idxs)

                query_idxs = []
                query_meta = []  # (role, j, delta, imp_id, imp_week, imp_practice, label)

                for j in weeks_with_data:
                    if j < i:
                        continue
                    for q_idx in sessions_by_week[j]:
                        if q_idx in enroll_set:
                            continue
                        query_idxs.append(q_idx)
                        query_meta.append(('genuine', j, j - i, -1, -1, -1, 1))

                for qid, pts in impostor_pts:
                    for idx, m, ordinal in pts:
                        query_idxs.append(idx)
                        query_meta.append(('impostor', -1, -1, qid, m, ordinal, 0))

                if not query_idxs:
                    continue

                scores = matcher.score_many(dataset, clipper, gallery, query_idxs)

                for (role, j, delta, imp_id, imp_week, imp_prac, label), score in zip(query_meta, scores):
                    effective_week = j if role == 'genuine' else imp_week
                    rows.append((
                        fold, mechanism_name, clipped_label, k_enrollment,
                        pid, pw_id, meta['length'], meta['substitution_category'],
                        i, role, j, delta,
                        imp_id, imp_week, imp_prac,
                        1 if effective_week == HARDWARE_CHANGE_WEEK else 0,
                        label, float(score),
                    ))

    return rows


def trials_to_frame(rows):
    import pandas as pd
    df = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    # password_id is a string key ('action', 'fstr', ...), not an integer
    for col in ('fold', 'k_enrollment', 'participant_id', 'length',
                'enrol_week', 'query_week', 'delta', 'imp_participant_id',
                'imp_week', 'imp_practice', 'hardware_change', 'label'):
        df[col] = df[col].astype(np.int32)
    return df
