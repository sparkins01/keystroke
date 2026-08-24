"""
Load the parsed event-level CSVs (Output/week_N.csv) and reconstruct, for every
valid sample, the clean sequence of *committed* characters that make up the
password actually typed -- i.e. replay backspace corrections and drop
modifier/noise key events, so that what's left is exactly the n keystroke
events (press+release time) for the n characters of the password.

A "Sample" is one (participant, week, session, password, repetition) unit.
"""
import os
import pickle
from collections import Counter, namedtuple

import pandas as pd

from password_metadata import build_password_metadata, skeleton

OUTPUT_DIR = os.environ.get('KD_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
CACHE_PATH = os.path.join(RESULT_DIR, 'samples_cache.pkl')

TERMINATOR_KEYS = {'Return'}
DELETE_KEYS = {'Back', '\x08'}
NOISE_KEYS = {
    '', '\x00', 'LShiftKey', 'RShiftKey', 'LControlKey', 'RControlKey', 'LMenu', 'RMenu',
    'Capital', 'Down', 'Left', 'Right', 'Up', 'Tab', 'Packet', 'VolumeDown', 'VolumeUp',
    'VolumeMute', 'MediaNextTrack', 'MediaPreviousTrack', 'MediaPlayPause', 'NumLock',
    'Escape', 'Next', 'Prior', 'Insert', 'LWin', 'RWin', 'Home', 'End', 'Add', 'Subtract',
    'Decimal', 'Multiply', 'Divide', 'Scroll', 'PrintScreen', 'Pause', 'Apps', 'PageUp',
    'PageDown', 'OemBackslash', 'LaunchMail', 'ProcessKey', 'Clear',
} | {f'F{i}' for i in range(1, 25)}

CharEvent = namedtuple('CharEvent', ['key', 'press', 'release'])
Sample = namedtuple('Sample', [
    'participant_id', 'participant_name', 'week', 'session_num',
    'password_id', 'repetition', 'phrase_index', 'char_events',
])


def reconstruct_committed_chars(rows, unknown_keys_counter):
    """rows: list of (key, press_dt, release_dt) tuples in key_seq order."""
    stack = []
    for key, press, release in rows:
        if key in TERMINATOR_KEYS:
            break
        if key in DELETE_KEYS:
            if stack:
                stack.pop()
            continue
        if key in NOISE_KEYS:
            continue
        if len(key) == 1:
            stack.append(CharEvent(key, press, release))
            continue
        unknown_keys_counter[key] += 1
    return stack


def load_samples(output_dir=OUTPUT_DIR, use_cache=True, verbose=True):
    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)

    meta_rows = build_password_metadata(output_dir)
    length_by_pid = {r['password_id']: r['length'] for r in meta_rows}

    samples = []
    unknown_keys_counter = Counter()
    n_length_mismatch = 0
    n_total_groups = 0

    for week in range(1, 9):
        path = os.path.join(output_dir, f'week_{week}.csv.gz')
        if not os.path.exists(path):
            path = os.path.join(output_dir, f'week_{week}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={'participant_id': int, 'session_num': int,
                                       'phrase_index': int, 'repetition': int,
                                       'key_seq': int, 'key': str},
                          keep_default_na=False, na_values=[])
        df['press_time'] = pd.to_datetime(df['press_time'], format='ISO8601')
        df['release_time'] = pd.to_datetime(df['release_time'], format='ISO8601')
        df.sort_values(['participant_id', 'session_num', 'phrase_index', 'repetition', 'key_seq'],
                        inplace=True)

        group_cols = ['participant_id', 'session_num', 'phrase_index', 'repetition']
        for (pid, session_num, phrase_index, repetition), g in df.groupby(group_cols, sort=False):
            n_total_groups += 1
            pname = (g['participant_name'].iloc[0]
                     if 'participant_name' in g.columns else f'P{pid:03d}')
            phrase_text = g['phrase_text'].iloc[0]
            pw_id = skeleton(phrase_text)
            expected_len = length_by_pid.get(pw_id)

            rows = list(zip(g['key'], g['press_time'], g['release_time']))
            char_events = reconstruct_committed_chars(rows, unknown_keys_counter)

            if expected_len is None or len(char_events) != expected_len:
                n_length_mismatch += 1
                continue

            samples.append(Sample(
                participant_id=pid, participant_name=pname, week=week,
                session_num=session_num, password_id=pw_id, repetition=repetition,
                phrase_index=phrase_index, char_events=tuple(char_events),
            ))

        if verbose:
            print(f'week {week}: {len(df)} rows loaded, running sample total = {len(samples)}')

    if verbose:
        print(f'\n{n_total_groups} raw (participant,session,phrase,repetition) groups seen')
        print(f'{len(samples)} samples reconstructed cleanly '
              f'({n_length_mismatch} dropped for a char-count mismatch after replay)')
        print(f'top unknown/unhandled key tokens ignored as noise: {unknown_keys_counter.most_common(10)}')

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(samples, f)

    with open(os.path.join(RESULT_DIR, 'data_loading_qa.txt'), 'w', encoding='utf-8') as f:
        f.write(f'raw groups: {n_total_groups}\n')
        f.write(f'clean samples: {len(samples)}\n')
        f.write(f'dropped (length mismatch after replay): {n_length_mismatch}\n')
        f.write(f'unknown key tokens: {dict(unknown_keys_counter)}\n')

    return samples


if __name__ == '__main__':
    samples = load_samples(use_cache=False)
    print(f'\nLoaded {len(samples)} samples total.')
    print('Example sample:', samples[0])
