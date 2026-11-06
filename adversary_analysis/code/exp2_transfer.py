"""
Experiment 2: the cross-password transfer attack, A4.

Threat model. The attacker holds the victim's password, and separately holds
keystroke timings for the victim typing *other* text, from a side channel, a
compromised site or a keylogger on an unrelated service. They have never seen
the victim type the target password. Can they synthesise a forgery good enough
to pass?

This is the attack the second-factor claim has never been evaluated against,
and it is the one that decides whether the study has a headline attack or only a
headline negative.

Levels, weakest first:

  A2        population forgery. The median way this password is typed, by
            people other than the victim. No victim knowledge at all.
  A4-global population forgery rescaled by the victim's overall typing speed,
            which is all a coarse leak would give.
  A4-keyed  the victim's own dwell time per character and interval per digraph,
            learned from the passwords they typed *other than* the target, and
            assembled into a forgery for the target.

Two attacker information horizons are run, bracketing the answer:
  full   the victim's other-password typing across the whole study, an upper
         bound on what a leak would yield
  early  weeks 1-2 only, a realistic single early leak

Leakage discipline. The victim's samples of the target password are never
touched by synthesis; they appear only as the enrolment gallery and the genuine
queries. Population statistics come from training-fold participants only. Both
are asserted at run time rather than trusted.

Forgeries are synthesised as press/release timestamps and then passed through
the same `compute_feature_vector` as real samples, so a forgery is internally
consistent: its trigraph and full-timing features follow from its dwells and
intervals rather than being invented separately. Synthesising the feature
vector directly would produce a structurally impossible sample that a matcher
might reject for the wrong reason.
"""
import argparse
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

import data_loading
from dataset import Dataset
from features import IQRClipper, compute_feature_vector
from raw_sequences import compute_raw_sequence
from password_metadata import build_password_metadata
from splits import fold_splits
from metrics import compute_eer

from matchers_classical import M1Manhattan

RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
K_ENROLLMENT = 5
FNMR_BUDGETS = [0.01, 0.02, 0.05]
EARLY_WEEKS = (1, 2)
N_STOCHASTIC = 5          # extra sampled forgeries per attack, for retry modelling
RNG_SEED = 0


# ------------------------------------------------------------ raw timings

def sample_timings(sample):
    """(keys, dwells, press_intervals) for one sample, in seconds."""
    ev = sample.char_events
    press = np.array([e.press.value for e in ev], dtype=np.float64) / 1e9
    release = np.array([e.release.value for e in ev], dtype=np.float64) / 1e9
    keys = tuple(e.key for e in ev)
    return keys, release - press, np.diff(press)


def canonical_keys(dataset, pw_id):
    """The character sequence for this password, as the modal key tuple across
    every sample of it. A handful of samples reconstruct to a different length
    after backspace replay; those are excluded from synthesis targets."""
    counts = defaultdict(int)
    for i in dataset.by_password[pw_id]:
        counts[tuple(e.key for e in dataset.samples[i].char_events)] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ------------------------------------------------------------ profiles

def per_password_tables(dataset, participant_ids, weeks=None):
    """{pid: {pw_id: (char->[dwells], digraph->[intervals])}}

    Kept split by password so that excluding the target password later is a
    filter rather than a rebuild, and so that a password typed more often does
    not dominate the victim's profile.
    """
    want = set(participant_ids)
    out = defaultdict(lambda: defaultdict(lambda: (defaultdict(list), defaultdict(list))))
    for idx, s in enumerate(dataset.samples):
        if s.participant_id not in want:
            continue
        if weeks is not None and s.week not in weeks:
            continue
        keys, dwells, intervals = sample_timings(s)
        if len(keys) < 2:
            continue
        dw, dg = out[s.participant_id][s.password_id]
        for k, d in zip(keys, dwells):
            dw[k].append(d)
        for a, b, v in zip(keys[:-1], keys[1:], intervals):
            dg[(a, b)].append(v)
    return out


def victim_profile(tables_for_pid, exclude_pw):
    """Median dwell per character and median interval per digraph for one
    victim, over every password except the target.

    Medians are taken per password first and then across passwords, so a
    character that happens to appear in many repetitions of one password does
    not swamp the estimate.
    """
    char_by_pw = defaultdict(list)
    digraph_by_pw = defaultdict(list)
    all_dwells, all_intervals = [], []
    used = set()
    for pw_id, (dw, dg) in tables_for_pid.items():
        if pw_id == exclude_pw:
            continue
        used.add(pw_id)
        for k, vals in dw.items():
            char_by_pw[k].append(float(np.median(vals)))
            all_dwells.extend(vals)
        for k, vals in dg.items():
            digraph_by_pw[k].append(float(np.median(vals)))
            all_intervals.extend(vals)
    return {
        'char': {k: float(np.median(v)) for k, v in char_by_pw.items()},
        'digraph': {k: float(np.median(v)) for k, v in digraph_by_pw.items()},
        'char_pool': {k: v for k, v in char_by_pw.items()},
        'digraph_pool': {k: v for k, v in digraph_by_pw.items()},
        'global_dwell': float(np.median(all_dwells)) if all_dwells else np.nan,
        'global_interval': float(np.median(all_intervals)) if all_intervals else np.nan,
        'used_passwords': used,
    }


def population_profile(dataset, train_ids, pw_id):
    """Median dwell and interval at each position of this password, over
    training-fold participants."""
    want = set(train_ids)
    n = len(canonical_keys(dataset, pw_id))
    dwells, intervals = [], []
    for i in dataset.by_password[pw_id]:
        s = dataset.samples[i]
        if s.participant_id not in want:
            continue
        keys, d, iv = sample_timings(s)
        if len(keys) != n:
            continue
        dwells.append(d)
        intervals.append(iv)
    if not dwells:
        return None
    D, I = np.vstack(dwells), np.vstack(intervals)
    return {
        'pos_dwell': np.median(D, axis=0),
        'pos_interval': np.median(I, axis=0),
        # kept so a forgery can be given realistic per-position variability,
        # which is the control that separates "this captures the victim" from
        # "a noiseless sample sits closer to any centroid than a real one does"
        'dwell_pool': D,
        'interval_pool': I,
        'global_dwell': float(np.median(np.concatenate(dwells))),
        'global_interval': float(np.median(np.concatenate(intervals))),
    }


# ------------------------------------------------------------ synthesis

class _Stamp:
    __slots__ = ('value',)

    def __init__(self, seconds):
        self.value = seconds * 1e9      # data_loading stores nanoseconds


class _FakeEvent:
    __slots__ = ('key', 'press', 'release')

    def __init__(self, key, press_s, release_s):
        self.key = key
        self.press = _Stamp(press_s)
        self.release = _Stamp(release_s)


def fake_events(keys, dwells, intervals):
    """char_event-like objects built from synthesised dwells and press-to-press
    intervals.

    Everything downstream derives from these, so a forgery passes through
    exactly the same feature and sequence builders as a real sample, and M1 and
    M4 are scoring the identical forgery rather than two separately synthesised
    approximations of one.
    """
    press = np.concatenate([[0.0], np.cumsum(intervals)])
    release = press + dwells
    assert len(press) == len(dwells) == len(keys)
    return [_FakeEvent(k, p, r) for k, p, r in zip(keys, press, release)]


def to_feature_vector(keys, dwells, intervals):
    return compute_feature_vector(fake_events(keys, dwells, intervals))


def to_raw_sequence(keys, dwells, intervals):
    return compute_raw_sequence(fake_events(keys, dwells, intervals))


def forge(keys, pop, victim, mode, rng=None):
    """One synthetic feature vector for `keys` under the given attack mode."""
    n = len(keys)
    pos_d = np.asarray(pop['pos_dwell'], dtype=float).copy()
    pos_i = np.asarray(pop['pos_interval'], dtype=float).copy()

    if mode == 'A2_population':
        d, iv = pos_d, pos_i

    elif mode == 'A2_sampled':
        # the same population knowledge, but with realistic per-position spread:
        # each value is an independent draw from how the population actually
        # types that position, rather than its median
        D, I = pop['dwell_pool'], pop['interval_pool']
        d = D[rng.randint(0, len(D), size=n), np.arange(n)]
        iv = I[rng.randint(0, len(I), size=n - 1), np.arange(n - 1)]

    elif mode == 'A4_global':
        rd = victim['global_dwell'] / pop['global_dwell'] if pop['global_dwell'] else 1.0
        ri = victim['global_interval'] / pop['global_interval'] if pop['global_interval'] else 1.0
        if rng is not None:
            D, I = pop['dwell_pool'], pop['interval_pool']
            base_d = D[rng.randint(0, len(D), size=n), np.arange(n)]
            base_i = I[rng.randint(0, len(I), size=n - 1), np.arange(n - 1)]
        else:
            base_d, base_i = pos_d, pos_i
        d, iv = base_d * rd, base_i * ri

    elif mode == 'A4_keyed':
        rd = victim['global_dwell'] / pop['global_dwell'] if pop['global_dwell'] else 1.0
        ri = victim['global_interval'] / pop['global_interval'] if pop['global_interval'] else 1.0
        d = np.empty(n)
        for i, k in enumerate(keys):
            pool = victim['char_pool'].get(k)
            if pool:
                d[i] = rng.choice(pool) if rng is not None else victim['char'][k]
            else:
                d[i] = pos_d[i] * rd          # unseen character, fall back to speed only
        iv = np.empty(n - 1)
        for i, pair in enumerate(zip(keys[:-1], keys[1:])):
            pool = victim['digraph_pool'].get(pair)
            if pool:
                iv[i] = rng.choice(pool) if rng is not None else victim['digraph'][pair]
            else:
                iv[i] = pos_i[i] * ri         # unseen digraph
    else:
        raise ValueError(mode)

    d = np.maximum(np.asarray(d, dtype=float), 1e-4)
    iv = np.maximum(np.asarray(iv, dtype=float), 1e-4)
    return d, iv          # representation-agnostic; see to_feature_vector / to_raw_sequence


def score_vector(gallery, clipper, vec, pw_id):
    """M1's scaled Manhattan score for a vector that is not in the dataset."""
    centroid, scale = gallery
    v = clipper.transform(vec, pw_id)
    return float((-np.abs(v - centroid) / scale).sum())


# ------------------------------------------------------------ run

MODES = ['A2_population', 'A2_sampled', 'A4_global', 'A4_keyed']

# modes that need no victim data, so the attacker information horizon is moot
POPULATION_ONLY = {'A2_population', 'A2_sampled'}
# modes whose forgery is a draw rather than a point estimate
STOCHASTIC = {'A2_sampled', 'A4_keyed'}


def forgery_rng(pid, pw_id):
    """A reproducible generator per (victim, password).

    Seeding per pair rather than once per run means the forgeries do not depend
    on iteration order, so M1 and M4 are guaranteed to score the identical
    synthetic samples even though they are produced by separate scripts.
    """
    h = zlib.crc32(f'{pid}|{pw_id}'.encode())
    return np.random.RandomState((RNG_SEED * 1000003 + h) % (2 ** 32))


def build_forgeries(keys, pop_pw, horizons, pid, pw_id):
    """{label: [(dwells, intervals), ...]} for one victim and password.

    Returns timings rather than feature vectors so the caller can render them
    into whichever representation its matcher needs.
    """
    rng = forgery_rng(pid, pw_id)
    forgeries = {}
    for horizon, tables in horizons.items():
        tab = tables.get(pid)
        if not tab:
            continue
        vp = victim_profile(tab, exclude_pw=pw_id)
        # the forgery must never have seen the victim type this password
        assert pw_id not in vp['used_passwords'], \
            f'leak: {pw_id} contributed to its own forgery for {pid}'
        if not np.isfinite(vp['global_dwell']):
            continue
        for mode in MODES:
            if mode in POPULATION_ONLY and horizon != 'full':
                continue              # no victim knowledge, horizon is moot
            suffix = '' if mode in POPULATION_ONLY else f'_{horizon}'
            try:
                if mode == 'A2_sampled':
                    forgeries[f'{mode}{suffix}'] = [
                        forge(keys, pop_pw, vp, mode, rng=rng)
                        for _ in range(N_STOCHASTIC)]
                elif mode in ('A4_keyed', 'A4_global'):
                    # median and sampled kept apart: if the median forgery wins
                    # and the sampled one does not, the attack is exploiting
                    # noiselessness rather than anything about the victim
                    forgeries[f'{mode}_median{suffix}'] = [
                        forge(keys, pop_pw, vp, mode)]
                    forgeries[f'{mode}_sampled{suffix}'] = [
                        forge(keys, pop_pw, vp, mode, rng=rng)
                        for _ in range(N_STOCHASTIC)]
                else:
                    forgeries[f'{mode}{suffix}'] = [forge(keys, pop_pw, vp, mode)]
            except Exception:
                continue
    return forgeries


def run(folds_wanted=None):
    samples = data_loading.load_samples(verbose=False)
    dataset = Dataset(samples)
    password_meta = {r['password_id']: r for r in build_password_metadata()}
    keys_by_pw = {pw: canonical_keys(dataset, pw) for pw in dataset.password_ids}

    rows = []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids):
        if folds_wanted is not None and fold not in folds_wanted:
            continue
        train_set, test_set = set(train_ids), set(test_ids)
        assert not (train_set & test_set), 'train/test overlap'

        feats_by_pw = {pw: [dataset.feature_vectors[i] for i in dataset.by_password[pw]
                            if dataset.samples[i].participant_id in train_set]
                       for pw in dataset.password_ids}
        clipper = IQRClipper().fit(feats_by_pw)
        m1 = M1Manhattan()
        m1.fit(dataset, train_ids, clipper)

        pop = {pw: population_profile(dataset, train_ids, pw) for pw in dataset.password_ids}
        horizons = {
            'full': per_password_tables(dataset, test_ids, weeks=None),
            'early': per_password_tables(dataset, test_ids, weeks=set(EARLY_WEEKS)),
        }

        for pid in test_ids:
            for pw_id in dataset.password_ids:
                if pop.get(pw_id) is None or pw_id not in password_meta:
                    continue
                keys = keys_by_pw[pw_id]
                n = len(keys)
                if n < 3:
                    continue

                forgeries = build_forgeries(keys, pop[pw_id], horizons, pid, pw_id)
                if not forgeries:
                    continue

                sessions_by_week = {w: dataset.sessions_at_week(pid, pw_id, w) for w in range(1, 9)}
                weeks_with_data = sorted(w for w, v in sessions_by_week.items() if v)
                for i in weeks_with_data:
                    pool_idx = dataset.sessions_for(pid, pw_id, max_week=i)
                    if not pool_idx:
                        continue
                    same_week = set(sessions_by_week[i])
                    same_in_pool = [x for x in pool_idx if x in same_week]
                    reserved = same_in_pool[-1] if len(same_in_pool) >= 2 else None
                    cand = [x for x in pool_idx if x != reserved]
                    if not cand:
                        continue
                    gallery = m1.gallery(dataset, clipper, cand[-K_ENROLLMENT:])

                    for label, variants in forgeries.items():
                        for vi, (d, iv) in enumerate(variants):
                            vec = to_feature_vector(keys, d, iv)
                            rows.append((fold, pid, pw_id, i, label, vi,
                                         score_vector(gallery, clipper, vec, pw_id)))

        print(f'fold {fold}: {len(rows):,} forgery scores so far')

    return pd.DataFrame(rows, columns=['fold', 'participant_id', 'password_id',
                                       'enrol_week', 'attack', 'variant', 'score'])


# ------------------------------------------------------------ analysis

GALLERY_KEY = ['fold', 'participant_id', 'password_id', 'enrol_week']


def analyse(forg):
    trials_dir = os.path.join(RESULT_DIR, 'exp1_trials')
    parts = sorted(f for f in os.listdir(trials_dir) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(os.path.join(trials_dir, p)) for p in parts],
                   ignore_index=True)
    gen = df[df['role'] == 'genuine']
    imp = df[df['role'] == 'impostor']

    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 82)
    emit('EXPERIMENT 2: cross-password transfer attack (A4)   M1 scaled Manhattan')
    emit('=' * 82)
    emit(f'forgery scores: {len(forg):,}  over {forg["attack"].nunique()} attack variants')
    emit()

    rows = []
    for delta in sorted(gen['delta'].unique()):
        g = gen[gen['delta'] == delta]['score'].values
        if len(g) < 100:
            continue
        galleries = gen[gen['delta'] == delta][GALLERY_KEY].drop_duplicates()
        imp_d = imp.merge(galleries, on=GALLERY_KEY, how='inner')
        forg_d = forg.merge(galleries, on=GALLERY_KEY, how='inner')

        levels = {
            'A0_unpractised': imp_d[imp_d['imp_practice'] == 0]['score'].values,
            'A1_practised': imp_d[imp_d['imp_week'] == 8]['score'].values,
        }
        for label in sorted(forg_d['attack'].unique()):
            levels[label] = forg_d[forg_d['attack'] == label]['score'].values

        for name, s in levels.items():
            if len(s) < 50:
                continue
            eer, _ = compute_eer(np.r_[np.ones(len(g)), np.zeros(len(s))], np.r_[g, s])
            row = {'delta': int(delta), 'attack': name, 'n': len(s), 'eer': eer,
                   # where the median forgery sits in the victim's own genuine
                   # score distribution; above ~50 means the forgery looks more
                   # like the template than the real user's own typing does
                   'genuine_percentile': float((g < np.median(s)).mean() * 100)}
            for b in FNMR_BUDGETS:
                thr = np.percentile(g, b * 100)
                fmr = float((s >= thr).mean())
                row[f'fmr_at_fnmr_{b}'] = fmr
                row[f'bits_at_fnmr_{b}'] = float(-np.log2(fmr)) if fmr > 0 else np.inf
            rows.append(row)
    tab = pd.DataFrame(rows)

    emit('-' * 82)
    emit('Attack success (FMR) at a 5% usability budget, by enrolment-to-query gap')
    emit('-' * 82)
    piv = tab.pivot(index='attack', columns='delta', values='fmr_at_fnmr_0.05')
    emit(piv.to_string(float_format=lambda v: f'{v:.3f}'))
    emit()
    emit('Marginal security in bits at the same budget')
    pivb = tab.pivot(index='attack', columns='delta', values='bits_at_fnmr_0.05')
    emit(pivb.to_string(float_format=lambda v: f'{v:.2f}'))
    emit()

    emit('-' * 82)
    emit('Control: is the attack capturing the victim, or exploiting noiselessness?')
    emit('-' * 82)
    emit('genuine_percentile = %% of the victim\'s own genuine samples scoring BELOW')
    emit('the typical forgery. Above 50 means the forgery matches the template more')
    emit('closely than the real user typically does, which no human attacker achieves.')
    emit()
    ctrl = tab[tab['delta'] == 0][['attack', 'fmr_at_fnmr_0.05', 'genuine_percentile']]
    emit(ctrl.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    emit()

    emit('-' * 82)
    emit('Does transfer beat a practised human impostor?')
    emit('-' * 82)
    for delta in (0, 7):
        if delta not in piv.columns:
            continue
        base = piv.loc['A1_practised', delta]
        emit(f'  delta={delta}: A1 practised human = {base:.3f}')
        for label in piv.index:
            if label.startswith('A0') or label.startswith('A1'):
                continue
            v = piv.loc[label, delta]
            verdict = 'BEATS' if v > base else 'below'
            emit(f'      {label:<26} {v:.3f}   {verdict} the human baseline')
        emit()

    os.makedirs(RESULT_DIR, exist_ok=True)
    tab.to_csv(os.path.join(RESULT_DIR, 'exp2_attack_table.csv'), index=False)
    with open(os.path.join(RESULT_DIR, 'exp2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit(f'written to {RESULT_DIR}')
    return tab


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    args = ap.parse_args()

    path = os.path.join(RESULT_DIR, 'exp2_forgery_scores.parquet')
    forg = run(folds_wanted=set(args.folds) if args.folds else None)
    forg.to_parquet(path, index=False)
    print(f'{len(forg):,} forgery scores -> {path}\n')
    analyse(forg)
