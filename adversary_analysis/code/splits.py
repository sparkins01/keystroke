"""
Section 3.2: 5-fold subject-level cross-validation. Participants are split into
five roughly equal folds; each of the five runs uses three folds for training,
one for validation (early stopping / threshold calibration) and one as the
held-out test set.

The paper's dataset had 65 participants; our recovered raw dataset has 86
(not every registrant completed all 8 weeks -- see Output/participants.csv and
Output/sessions_summary.csv). Rather than arbitrarily drop 21 people to match
the paper's headline number, every participant with any usable data is
assigned to a fold; a participant who is missing sessions for a given week
simply contributes fewer trials for that week, which the verification
protocol already handles.
"""
import random

N_FOLDS = 5


def make_folds(participant_ids, seed=42):
    ids = sorted(set(participant_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = [ids[i::N_FOLDS] for i in range(N_FOLDS)]
    return folds


def fold_splits(participant_ids, seed=42):
    """Yields (fold_index, train_ids, val_ids, test_ids) for each of the 5 runs."""
    folds = make_folds(participant_ids, seed=seed)
    for k in range(N_FOLDS):
        test_ids = folds[k]
        val_ids = folds[(k + 1) % N_FOLDS]
        train_ids = [pid for i, f in enumerate(folds) if i not in (k, (k + 1) % N_FOLDS) for pid in f]
        yield k, sorted(train_ids), sorted(val_ids), sorted(test_ids)


if __name__ == '__main__':
    from data_loading import load_samples
    samples = load_samples(use_cache=True, verbose=False)
    pids = {s.participant_id for s in samples}
    print(f'{len(pids)} participants with usable data')
    for k, train_ids, val_ids, test_ids in fold_splits(pids):
        print(f'fold {k}: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}')
        overlap = (set(train_ids) & set(val_ids)) | (set(train_ids) & set(test_ids)) | (set(val_ids) & set(test_ids))
        assert not overlap, overlap
    print('no train/val/test participant overlap in any fold - OK')
