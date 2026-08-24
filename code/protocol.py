"""
Section 3.4: unified verification protocol linking matching mechanism and
temporal gap. For a held-out test participant p, password w, enrolment week i
and query week j (i <= j, Delta = j - i in {0..7}):
  - a genuine trial compares the gallery built from p's sessions up to and
    including week i against p's own query sample from week j;
  - an impostor trial compares that same gallery against a different test
    participant's query sample from week j (same password).

Every other test participant with data at the query week is used as an
impostor by default (`max_impostors_per_trial=None`); pass an int to instead
subsample a fixed, seeded number of impostor identities per genuine trial if
the trial count needs to be kept smaller.

Delta=0 (querying the same week the gallery was built from) needs at least
one of that week's own sessions left over, unused by the gallery, to query
against. Naively taking "the most recent k_enrollment sessions up to and
including week i" can consume *all* of week i's own sessions into the
gallery (this happened routinely here: with k=5 and ~4 sessions/week, most
enrolment weeks' own sessions were entirely absorbed into the gallery),
leaving zero genuine trials at Delta=0 for that enrolment week -- this is
exactly why Delta=0 had far fewer trials, and a far noisier EER, than every
other Delta in this study's first full run. One session from week i is now
always reserved out of the gallery (whenever week i has 2+ sessions) so
Delta=0 trials remain available.

All queries sharing one gallery are scored in a single `score_many()` call
(matchers implement a vectorised batch path) instead of one `score()` call
per trial -- for M2's per-password classifier this is a large speedup, since
a single predict_proba() call over N rows is far cheaper than N separate
single-row calls.
"""
import random


def generate_trials(dataset, matcher, clipper, test_participant_ids, password_meta,
                     mechanism_name, fold, clipped_label, k_enrollment=5,
                     max_impostors_per_trial=None, seed=0):
    rng = random.Random(seed)
    test_ids = list(test_participant_ids)
    trials = []

    for pid in test_ids:
        for pw_id in dataset.password_ids:
            meta = password_meta.get(pw_id)
            if meta is None:
                continue
            sessions_by_week = {w: dataset.sessions_at_week(pid, pw_id, w) for w in range(1, 9)}
            weeks_with_data = sorted(w for w, idxs in sessions_by_week.items() if idxs)
            if not weeks_with_data:
                continue

            for i in weeks_with_data:
                pool = dataset.sessions_for(pid, pw_id, max_week=i)
                if not pool:
                    continue
                same_week = set(sessions_by_week[i])
                same_week_in_pool = [x for x in pool if x in same_week]
                reserved_for_delta0 = same_week_in_pool[-1] if len(same_week_in_pool) >= 2 else None
                candidate_pool = [x for x in pool if x != reserved_for_delta0]
                enroll_idxs = candidate_pool[-k_enrollment:]
                gallery = matcher.gallery(dataset, clipper, enroll_idxs)
                enroll_set = set(enroll_idxs)

                query_idxs = []
                query_info = []  # (j, delta, hardware_change, label)

                for j in weeks_with_data:
                    if j < i:
                        continue
                    delta = j - i
                    hardware_change = 1 if j == 5 else 0

                    for q_idx in sessions_by_week[j]:
                        if q_idx in enroll_set:
                            continue
                        query_idxs.append(q_idx)
                        query_info.append((j, delta, hardware_change, 1))

                    other_pids = [p2 for p2 in test_ids
                                  if p2 != pid and dataset.sessions_at_week(p2, pw_id, j)]
                    rng.shuffle(other_pids)
                    impostor_pids = other_pids if max_impostors_per_trial is None \
                        else other_pids[:max_impostors_per_trial]
                    for p2 in impostor_pids:
                        cand = dataset.sessions_at_week(p2, pw_id, j)
                        q_idx2 = rng.choice(cand)
                        query_idxs.append(q_idx2)
                        query_info.append((j, delta, hardware_change, 0))

                if not query_idxs:
                    continue
                scores = matcher.score_many(dataset, clipper, gallery, query_idxs) \
                    if hasattr(matcher, 'score_many') \
                    else [matcher.score(dataset, clipper, gallery, q) for q in query_idxs]

                for (j, delta, hardware_change, label), score in zip(query_info, scores):
                    trials.append(_trial_row(
                        fold, mechanism_name, clipped_label, k_enrollment,
                        pid, pw_id, meta, i, j, delta, hardware_change, label, score))

    return trials


def _trial_row(fold, mechanism, clipped_label, k_enrollment, pid, pw_id, meta,
                i, j, delta, hardware_change, label, score):
    return {
        'fold': fold, 'mechanism': mechanism, 'clipped': clipped_label, 'k_enrollment': k_enrollment,
        'participant_id': pid, 'password_id': pw_id, 'length': meta['length'],
        'substitution_category': meta['substitution_category'],
        'enrol_week': i, 'query_week': j, 'delta': delta,
        'hardware_change': hardware_change, 'label': label, 'score': score,
    }
