"""
ITAD, the Instance-based Tail Area Density metric of Ayotte, Banavar, Hou and
Schuckers (IEEE TBIOM 2020), added for TBIOM reviewer 2, point 5.

The metric as published, from Section V-A4 of that paper:

    s_i = CDF_{g_i}(x_i)        if x_i <= M_{g_i}
    s_i = 1 - CDF_{g_i}(x_i)    if x_i >  M_{g_i}
    S   = (1/N) sum_i s_i^p

where CDF_{g_i} is the empirical cumulative distribution of the i-th graph in
the profile, M_{g_i} its median, x_i the query's value for that graph, N the
number of graphs shared by query and profile, and p a scaling exponent. The
paper uses p = 1/2. The five feature types are the monograph M (dwell) and
the four digraphs DD, UD, DU and UU. Per-feature scores are fused by a
weighted average whose weights are random-forest mean-decrease-in-impurity
importances. Higher S means more similar.

Adapting it to this study, and what that costs.

ITAD was designed for free text, where a profile pools thousands of
occurrences of each distinct key pair. Here the text is fixed and short, so a
"graph" is a position in the password rather than a key pair, and a profile
holds exactly k values per graph, one per enrolment session. At the paper's
primary k=5 that is five values, which clears their stated minimum of four
occurrences per graph but only just. An empirical CDF over five points has a
resolution of 0.2, so s_i is coarse. That is a real limitation of applying
ITAD in this setting rather than a defect of the metric, and it is the main
thing this benchmark measures.

Choices the paper leaves open, made here and recorded so they can be checked:

  Empirical CDF convention. The paper does not state one. A plain
  "fraction <= x" convention breaks the paper's own statement that s lies in
  [0, 0.5], because at the median it returns 1.0 rather than 0.5. The
  mid-rank convention is used instead:

      CDF(x) = ( #{profile < x} + 0.5 * #{profile == x} ) / k

  which gives 0.5 at the median and keeps s in [0, 0.5] as the paper
  describes. It also degrades gracefully when a profile graph is constant,
  which matters here because roughly a quarter of this collection's key
  events carry a release timestamp identical to their press timestamp, so
  dwell profiles are often degenerate.

  Median convention: numpy's, averaging the two central values for even k.

  Feature fusion: both variants are provided. 'equal' averages the five
  feature scores. 'mdi' uses random-forest importances fitted on the
  training participants, pooled over participants rather than the paper's
  one-vs-all per user, which is not available at this profile size.

  Graphs with fewer than four profile occurrences are dropped, matching the
  paper's minimum. With k >= 4 none are dropped in practice.
"""
import numpy as np

FEATURES = ['M', 'DD', 'UD', 'DU', 'UU']
MIN_PROFILE_OCCURRENCES = 4
DEFAULT_P = 0.5


def feature_arrays(char_events):
    """The five ITAD feature vectors for one attempt, in seconds."""
    press = np.array([c.press.value for c in char_events], dtype=np.float64) / 1e9
    release = np.array([c.release.value for c in char_events], dtype=np.float64) / 1e9
    return {
        'M': release - press,
        'DD': press[1:] - press[:-1],
        'UD': press[1:] - release[:-1],
        'DU': release[1:] - press[:-1],
        'UU': release[1:] - release[:-1],
    }


def _midrank_cdf(profile, queries):
    """profile: (k, n). queries: (Q, n). Returns (Q, n) mid-rank CDF values."""
    k = profile.shape[0]
    p = profile[None, :, :]
    q = queries[:, None, :]
    less = (p < q).sum(axis=1)
    equal = (p == q).sum(axis=1)
    return (less + 0.5 * equal) / float(k)


def itad_feature_score(profile, queries, p_exponent=DEFAULT_P,
                       min_occurrences=MIN_PROFILE_OCCURRENCES):
    """Equation 5 for one feature type.

    profile: (k, n) enrolment values. queries: (Q, n). Returns (Q,).
    """
    if profile.shape[0] < min_occurrences or profile.shape[1] == 0:
        return np.full(queries.shape[0], np.nan)
    median = np.median(profile, axis=0)
    cdf = _midrank_cdf(profile, queries)
    s = np.where(queries <= median[None, :], cdf, 1.0 - cdf)
    return np.mean(np.power(np.clip(s, 0.0, None), p_exponent), axis=1)


class ITADMatcher:
    """Follows the same gallery/score_many interface as the other matchers,
    so it drops into the unchanged trial protocol."""

    name = 'M5_ITAD'

    def __init__(self, fusion='equal', p_exponent=DEFAULT_P, seed=0,
                 max_pairs_per_password=4000,
                 min_occurrences=MIN_PROFILE_OCCURRENCES):
        if fusion not in ('equal', 'mdi'):
            raise ValueError(fusion)
        self.fusion = fusion
        self.p_exponent = p_exponent
        self.min_occurrences = min_occurrences
        self.seed = seed
        self.max_pairs_per_password = max_pairs_per_password
        self.weights = np.ones(len(FEATURES)) / len(FEATURES)
        self._cache = {}

    # ------------------------------------------------------------------
    def _features(self, dataset, idx):
        cached = self._cache.get(idx)
        if cached is None:
            cached = feature_arrays(dataset.samples[idx].char_events)
            self._cache[idx] = cached
        return cached

    def _stack(self, dataset, idxs, feature):
        return np.vstack([self._features(dataset, i)[feature] for i in idxs])

    def _per_feature_scores(self, dataset, gallery, query_idxs):
        """(Q, 5) matrix of per-feature ITAD scores."""
        out = np.empty((len(query_idxs), len(FEATURES)))
        for j, feature in enumerate(FEATURES):
            profile = gallery[feature]
            queries = self._stack(dataset, query_idxs, feature)
            out[:, j] = itad_feature_score(profile, queries, self.p_exponent,
                                           self.min_occurrences)
        return out

    # ------------------------------------------------------------------
    def fit(self, dataset, train_participant_ids, clipper):
        """Only the 'mdi' fusion needs fitting. The weights come from a random
        forest over the five per-feature ITAD scores, trained on genuine and
        impostor pairs drawn from the training participants only."""
        if self.fusion == 'equal':
            return self

        from collections import defaultdict
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(self.seed)
        train_ids = set(train_participant_ids)
        rows, labels = [], []

        for pw_id in dataset.password_ids:
            by_part = defaultdict(list)
            for i in dataset.by_password[pw_id]:
                pid = dataset.samples[i].participant_id
                if pid in train_ids:
                    by_part[pid].append(i)
            parts = [p for p, idxs in by_part.items()
                     if len(idxs) >= self.min_occurrences + 1]
            if len(parts) < 2:
                continue

            budget = max(1, self.max_pairs_per_password // max(1, len(parts)))
            for pid in parts:
                idxs = sorted(by_part[pid])
                enrol = idxs[:self.min_occurrences + 1]
                rest = idxs[self.min_occurrences + 1:]
                if not rest:
                    continue
                gal = self.gallery(dataset, None, enrol)

                genuine = rest[:budget]
                rows.append(self._per_feature_scores(dataset, gal, genuine))
                labels.extend([1] * len(genuine))

                others = [p for p in parts if p != pid]
                if not others:
                    continue
                picked = rng.choice(others, size=min(len(others), budget), replace=False)
                imp_idxs = [by_part[p][rng.randint(len(by_part[p]))] for p in picked]
                rows.append(self._per_feature_scores(dataset, gal, imp_idxs))
                labels.extend([0] * len(imp_idxs))

        if not rows:
            return self
        X = np.vstack(rows)
        y = np.asarray(labels)
        ok = ~np.isnan(X).any(axis=1)
        X, y = X[ok], y[ok]
        if len(np.unique(y)) < 2:
            return self

        forest = RandomForestClassifier(n_estimators=200, random_state=self.seed,
                                        n_jobs=-1, min_samples_leaf=5)
        forest.fit(X, y)
        importances = forest.feature_importances_
        total = importances.sum()
        self.weights = (importances / total) if total > 0 else self.weights
        return self

    # ------------------------------------------------------------------
    def gallery(self, dataset, clipper, sample_idxs):
        """ITAD builds an empirical distribution per graph, so the gallery is
        the stacked enrolment values rather than a centroid. The clipper is
        ignored: the metric is percentile based and the paper applies no
        scaling."""
        return {f: self._stack(dataset, sample_idxs, f) for f in FEATURES}

    def score(self, dataset, clipper, gallery, query_idx):
        return self.score_many(dataset, clipper, gallery, [query_idx])[0]

    def score_many(self, dataset, clipper, gallery, query_idxs):
        per_feature = self._per_feature_scores(dataset, gallery, query_idxs)
        weights = self.weights
        valid = ~np.isnan(per_feature)
        safe = np.where(valid, per_feature, 0.0)
        denom = (valid * weights[None, :]).sum(axis=1)
        numer = (safe * weights[None, :]).sum(axis=1)
        scores = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)
        return scores.tolist()  # higher = more genuine
