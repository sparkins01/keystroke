"""
M1 (Manhattan distance-threshold) and M2 (shallow-ML pairwise-difference
classifier), Section 3.5. Both operate on the aggregated feature
representation; because feature dimensionality depends on password length,
M2 trains one small classifier per password.

M1 is the *scaled* Manhattan distance (each feature's absolute difference is
divided by that feature's variability before summing), not plain Manhattan.
Killourhy & Maxion's classic keystroke-benchmark comparison (the standard
reference point for a classical distance-threshold matcher on fixed-text
passwords) found scaled Manhattan clearly beats plain Manhattan on their
data (~9.6% vs ~15.3% EER); an earlier, unscaled version of M1 here landed
around 27% EER at Delta=0, well above even their plain-Manhattan number, so
scaling was worth adopting rather than treating as a minor variant. The
per-feature scale is estimated from within-gallery variability (how
variable *this participant's own* enrolment sessions are for each feature),
falling back to the population-level (training-participants) scale when the
gallery is too small (k<2) to estimate it, per feature, to avoid dividing by
a near-zero, noise-driven estimate.
"""
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class M1Manhattan:
    name = 'M1_Manhattan'

    def __init__(self, min_std_floor_frac=0.1):
        self.population_std = {}
        self.min_std_floor_frac = min_std_floor_frac

    def fit(self, dataset, train_participant_ids, clipper):
        train_ids = set(train_participant_ids)
        for pw_id in dataset.password_ids:
            idxs = [i for i in dataset.by_password[pw_id] if dataset.samples[i].participant_id in train_ids]
            if not idxs:
                continue
            vecs = np.vstack([clipper.transform(dataset.feature_vectors[i], pw_id) for i in idxs])
            self.population_std[pw_id] = vecs.std(axis=0)

    def gallery(self, dataset, clipper, sample_idxs):
        pw_id = dataset.samples[sample_idxs[0]].password_id
        vecs = np.vstack([clipper.transform(dataset.feature_vectors[i], pw_id) for i in sample_idxs])
        centroid = vecs.mean(axis=0)
        pop_std = self.population_std.get(pw_id, np.ones_like(centroid))
        floor = self.min_std_floor_frac * pop_std
        if len(sample_idxs) >= 2:
            scale = np.maximum(vecs.std(axis=0, ddof=1), floor)
        else:
            scale = np.maximum(pop_std, floor)
        scale = np.where(scale <= 0, 1.0, scale)
        return centroid, scale

    def score(self, dataset, clipper, gallery, query_idx):
        return self.score_many(dataset, clipper, gallery, [query_idx])[0]

    def score_many(self, dataset, clipper, gallery, query_idxs):
        centroid, scale = gallery
        pw_id = dataset.samples[query_idxs[0]].password_id
        Q = np.vstack([clipper.transform(dataset.feature_vectors[i], pw_id) for i in query_idxs])
        return (-np.abs(Q - centroid) / scale).sum(axis=1).tolist()  # higher = more genuine


class M2ShallowML:
    name = 'M2_ShallowML'

    def __init__(self, seed=0, max_pairs_per_password=4000):
        self.seed = seed
        self.max_pairs_per_password = max_pairs_per_password
        self.models = {}

    def fit(self, dataset, train_participant_ids, clipper):
        rng = np.random.RandomState(self.seed)
        train_ids = set(train_participant_ids)
        self.models = {}

        for pw_id in dataset.password_ids:
            by_part = defaultdict(list)
            for i in dataset.by_password[pw_id]:
                pid = dataset.samples[i].participant_id
                if pid in train_ids:
                    by_part[pid].append(i)
            parts = [p for p, idxs in by_part.items() if len(idxs) >= 2]
            if len(parts) < 2:
                continue

            X, y = [], []
            for pid in parts:
                idxs = by_part[pid]
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        va = clipper.transform(dataset.feature_vectors[idxs[a]], pw_id)
                        vb = clipper.transform(dataset.feature_vectors[idxs[b]], pw_id)
                        X.append(np.abs(va - vb))
                        y.append(1)
            n_genuine = len(X)
            if n_genuine == 0:
                continue

            all_parts = list(by_part.keys())
            attempts = 0
            max_attempts = n_genuine * 10 + 50
            while len(y) < 2 * n_genuine and attempts < max_attempts:
                attempts += 1
                p1, p2 = rng.choice(all_parts, 2, replace=False)
                i1 = rng.choice(by_part[p1])
                i2 = rng.choice(by_part[p2])
                va = clipper.transform(dataset.feature_vectors[i1], pw_id)
                vb = clipper.transform(dataset.feature_vectors[i2], pw_id)
                X.append(np.abs(va - vb))
                y.append(0)

            if len(set(y)) < 2:
                continue
            X = np.vstack(X)
            y = np.array(y)
            if len(y) > self.max_pairs_per_password:
                sel = rng.choice(len(y), self.max_pairs_per_password, replace=False)
                X, y = X[sel], y[sel]

            model = HistGradientBoostingClassifier(max_depth=4, max_iter=60, random_state=self.seed)
            model.fit(X, y)
            self.models[pw_id] = model

    def gallery(self, dataset, clipper, sample_idxs):
        vecs = [clipper.transform(dataset.feature_vectors[i], dataset.samples[i].password_id)
                for i in sample_idxs]
        return np.mean(vecs, axis=0)

    def score(self, dataset, clipper, gallery, query_idx):
        return self.score_many(dataset, clipper, gallery, [query_idx])[0]

    def score_many(self, dataset, clipper, gallery, query_idxs):
        pw_id = dataset.samples[query_idxs[0]].password_id
        Q = np.vstack([clipper.transform(dataset.feature_vectors[i], pw_id) for i in query_idxs])
        diffs = np.abs(Q - gallery)
        model = self.models.get(pw_id)
        if model is None:
            return (-diffs.sum(axis=1)).tolist()
        return model.predict_proba(diffs)[:, 1].tolist()
