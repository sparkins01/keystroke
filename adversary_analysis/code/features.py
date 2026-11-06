"""
Section 3.1 "Aggregated feature representation": seven feature definitions
computed per sample, concatenated into one feature vector:

  full timing        - 1 value:  last release - first press
  press-to-release   - n values: release[i] - press[i]                 (dwell)
  press-to-press     - n-1 values: press[i+1] - press[i]
  release-to-press   - n-1 values: press[i+1] - release[i]              (flight)
  release-to-release - n-1 values: release[i+1] - release[i]
  trigraph PP         - n-2 values: press[i+2] - press[i]
  trigraph RR         - n-2 values: release[i+2] - release[i]

All durations are in seconds. A password of length n therefore yields a
1 + n + 3*(n-1) + 2*(n-2) = 6n - 6 dimensional feature vector. Because n
varies by password, feature vectors are only ever compared within the same
password (as the verification protocol requires).
"""
import numpy as np


def feature_names(n):
    names = ['full_timing']
    names += [f'PR_{i}' for i in range(n)]
    names += [f'PP_{i}' for i in range(n - 1)]
    names += [f'RP_{i}' for i in range(n - 1)]
    names += [f'RR_{i}' for i in range(n - 1)]
    names += [f'triPP_{i}' for i in range(n - 2)]
    names += [f'triRR_{i}' for i in range(n - 2)]
    return names


def feature_dim(n):
    return 1 + n + 3 * (n - 1) + 2 * (n - 2)


def compute_feature_vector(char_events):
    n = len(char_events)
    press = np.array([e.press.value for e in char_events], dtype=np.float64) / 1e9  # seconds
    release = np.array([e.release.value for e in char_events], dtype=np.float64) / 1e9

    full_timing = np.array([release[-1] - press[0]])
    pr = release - press
    pp = press[1:] - press[:-1]
    rp = press[1:] - release[:-1]
    rr = release[1:] - release[:-1]
    tri_pp = press[2:] - press[:-2]
    tri_rr = release[2:] - release[:-2]

    return np.concatenate([full_timing, pr, pp, rp, rr, tri_pp, tri_rr])


class IQRClipper:
    """1.5x IQR fence, fit per-password on a given set of samples' feature
    matrices, then applied to any partition (train/val/test) unchanged --
    exactly the leakage-avoidance rule in Section 3.1."""

    def __init__(self):
        self.bounds_by_password = {}

    def fit(self, features_by_password):
        for pw_id, feats in features_by_password.items():
            if len(feats) == 0:
                continue
            arr = np.vstack(feats)
            q1 = np.percentile(arr, 25, axis=0)
            q3 = np.percentile(arr, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds_by_password[pw_id] = (lower, upper)
        return self

    def transform(self, feature_vector, password_id):
        bounds = self.bounds_by_password.get(password_id)
        if bounds is None:
            return feature_vector
        lower, upper = bounds
        return np.clip(feature_vector, lower, upper)
