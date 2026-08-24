"""
Parse raw per-participant keystroke timing logs (Data/Week N/<folder>/*timings.txt,
paired with *accuracy.txt) into one tidy, event-level CSV per week.

Output (Output/):
  week_1.csv .. week_8.csv   - one row per valid (non-discarded) key event
  participants.csv           - normalized-name -> participant_id lookup
  sessions_summary.csv       - one row per session actually parsed (QA)
  skipped_sessions.csv       - one row per file/session that could not be used, with reason

See README section below (docstring) for the parsing rules.

Rules used to decide what is a "complete"/valid sample vs an "incomplete" one to ignore:

1. Each participant folder contains a "<date> timings.txt" file (raw key events, grouped
   under "**************phrase***************" header lines) and a matching
   "<date> accuracy.txt" file. The accuracy file lists, *in the order they were attempted*,
   one line per question "Qn" of the form:
       Wrong attemps for Qn: <wrong>, Attemps for Qn: <attempts>, ...
   `attempts` (= wrong + 1) tells us exactly how many consecutive header-blocks in
   timings.txt belong to that question: the first `wrong` of them are failed/incomplete
   retries (they contain typos, stray backspaces, etc.) and the LAST one is the one the
   participant actually submitted successfully. Only that last block is kept.

2. Some sessions were restarted from scratch (the Qn label sequence drops back down,
   e.g. ...,159,160,1,2,3,...). Only the *final* such restart-cycle is kept; earlier,
   abandoned full/partial run-throughs are discarded entirely.

3. A session is skipped outright (logged to skipped_sessions.csv) if it has no matching
   accuracy file (so completeness can't be determined), no timings file, or is an exact
   byte-for-byte duplicate of another session already parsed for the same participant/week
   (this happens a handful of times where the tool appears to have saved the same session
   twice).

4. Participants are matched across weeks by their normalised name (per assignment
   instructions), not by the numeric id prefix in the folder name (that id does happen to
   stay constant per person across weeks, but name is the specified join key).
"""
import csv
import hashlib
import os
import re
from collections import defaultdict
from datetime import datetime

DATA_DIR = os.environ.get('KD_RAW_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw'))
OUT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))

FOLDER_RE = re.compile(r'^(?P<id>\d+)-(?P<suffix>\d+) - (?P<name>.+?) - (?P<date>.+)$')
TIMING_FN_RE = re.compile(r'^(?P<date>\d{2}-\d{2}-\d{4})\s*timings\s*(?P<suffix>.*)\.txt$', re.IGNORECASE)
ACC_FN_RE = re.compile(r'^(?P<date>\d{2}-\d{2}-\d{4})\s*accuracy\s*(?P<suffix>.*)\.txt$', re.IGNORECASE)

MASTER_RE = re.compile(
    r'\*{3,}\s*(?P<header>[^\*\r\n]*?)\s*\*{3,}'
    r'|'
    r'(?P<t1>\d{2}[/-]\d{2}[/-]\d{4} \d{2}:\d{2}:\d{2}:\d{7}):\s*(?P<key>.*?):\s*'
    r'Elapsed_time_(?P<elapsed>\d{2}:\d{2}:\d{2}\.\d{7}):\s*'
    r'(?P<t2>\d{2}[/-]\d{2}[/-]\d{4} \d{2}:\d{2}:\d{2}:\d{7})'
)

ACC_RE = re.compile(
    r'Wrong attemps for Q(?P<q>\d+):\s*\d+,\s*Attemps for Q\d+:\s*(?P<attempts>\d+)'
)

TS_RE = re.compile(r'(\d{2})[/-](\d{2})[/-](\d{4}) (\d{2}):(\d{2}):(\d{2}):(\d{7})')
EL_RE = re.compile(r'(\d{2}):(\d{2}):(\d{2})\.(\d{7})')


def normalize_name(name):
    n = re.sub(r'\s+', ' ', name.strip().upper())
    n = re.sub(r'[^A-Z0-9 \-]', '', n)
    return n


def normalize_suffix(suffix):
    return re.sub(r'\s+', '', suffix.strip().lower())


NUMPAD_RE = re.compile(r'^NumPad(\d)$')


def normalize_key(key):
    """Collapse numpad digit keys onto their main-keyboard digit (e.g. NumPad3 -> 3);
    the study doesn't care which physical key produced a given digit."""
    m = NUMPAD_RE.match(key)
    return m.group(1) if m else key


def parse_ts(s):
    dd, mm, yyyy, hh, mi, ss, frac = TS_RE.match(s).groups()
    micros = int(frac) // 10
    return datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss), micros)


def parse_elapsed_seconds(s):
    hh, mm, ss, frac = EL_RE.match(s).groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(frac) / 1e7


def read_text(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def build_participant_registry():
    names = set()
    for wd in sorted(os.listdir(DATA_DIR)):
        wpath = os.path.join(DATA_DIR, wd)
        if not os.path.isdir(wpath) or not wd.startswith('Week'):
            continue
        for pfolder in os.listdir(wpath):
            ppath = os.path.join(wpath, pfolder)
            if not os.path.isdir(ppath):
                continue
            m = FOLDER_RE.match(pfolder)
            if not m:
                continue
            names.add(normalize_name(m.group('name')))
    ordered = sorted(names)
    return {name: i + 1 for i, name in enumerate(ordered)}


def parse_session(timings_path, accuracy_path):
    """Returns (results, meta) where results is a list of dicts:
    {phrase_index, repetition, phrase_text, events:[(t1,key,elapsed,t2), ...]}
    meta carries QA counters. Raises ValueError if accuracy file has no parseable rows.
    """
    acc_text = read_text(accuracy_path)
    acc_rows = [(int(m.group('q')), int(m.group('attempts'))) for m in ACC_RE.finditer(acc_text)]
    if not acc_rows:
        raise ValueError('accuracy file has no parseable Q lines')

    tim_text = read_text(timings_path)
    blocks = []
    current = None
    preamble_events = 0
    for m in MASTER_RE.finditer(tim_text):
        if m.group('header') is not None:
            current = [m.group('header').strip(), []]
            blocks.append(current)
        else:
            if current is None:
                preamble_events += 1
            else:
                current[1].append((m.group('t1'), m.group('key'), m.group('elapsed'), m.group('t2')))

    # split into restart-cycles based on Qn label decreasing
    cycles = []
    cur_cycle = []
    prev_label = None
    for qlabel, attempts in acc_rows:
        if prev_label is not None and qlabel < prev_label:
            cycles.append(cur_cycle)
            cur_cycle = []
        cur_cycle.append((qlabel, attempts))
        prev_label = qlabel
    cycles.append(cur_cycle)
    # a "restart" produces a new, separate cycle; pick the most complete one (most
    # entries), breaking ties in favour of the latest, since a short abandoned restart
    # tacked on after a full run-through should not shadow the complete data.
    best_cycle_pos = max(range(len(cycles)), key=lambda i: (len(cycles[i]), i))
    last_cycle_start = sum(len(c) for c in cycles[:best_cycle_pos])
    kept_cycle_len = len(cycles[best_cycle_pos])
    last_cycle_end = last_cycle_start + kept_cycle_len

    block_idx = 0
    results = []
    truncated = False
    for i, (qlabel, attempts) in enumerate(acc_rows):
        if i >= last_cycle_end:
            break
        if block_idx + attempts > len(blocks):
            truncated = True
            break
        block_idx += attempts
        if i >= last_cycle_start:
            phrase_text, events = blocks[block_idx - 1]
            results.append({
                'phrase_index': ((qlabel - 1) % 40) + 1,
                'repetition': ((qlabel - 1) // 40) + 1,
                'phrase_text': phrase_text,
                'events': events,
            })

    meta = {
        'n_cycles': len(cycles),
        'kept_cycle_index': best_cycle_pos + 1,
        'n_discarded_rows_other_cycles': len(acc_rows) - kept_cycle_len,
        'n_valid_samples': len(results),
        'truncated_accuracy_vs_blocks': truncated,
        'unused_trailing_blocks': max(0, len(blocks) - block_idx),
        'preamble_events_ignored': preamble_events,
    }
    return results, meta


def gather_week_sessions(week_num, participant_ids, skip_log, summary_log, notes_log):
    wpath = os.path.join(DATA_DIR, f'Week {week_num}')
    folders_by_name = defaultdict(list)
    for pfolder in sorted(os.listdir(wpath)):
        ppath = os.path.join(wpath, pfolder)
        if not os.path.isdir(ppath):
            continue
        m = FOLDER_RE.match(pfolder)
        if not m:
            skip_log.append([week_num, pfolder, '', '', 'folder name did not match expected pattern'])
            continue
        norm = normalize_name(m.group('name'))
        folders_by_name[norm].append((ppath, m.group('name')))

    sessions = []  # (participant_id, participant_name, session_date, timing_path, accuracy_path, session_num)

    for norm_name, folder_list in sorted(folders_by_name.items()):
        raw_name = folder_list[0][1]
        pid = participant_ids[norm_name]
        timing_files = defaultdict(list)
        acc_files = defaultdict(list)
        for folder_path, _ in folder_list:
            for f in os.listdir(folder_path):
                tm = TIMING_FN_RE.match(f)
                if tm:
                    key = (tm.group('date'), normalize_suffix(tm.group('suffix')))
                    timing_files[key].append(os.path.join(folder_path, f))
                    continue
                am = ACC_FN_RE.match(f)
                if am:
                    key = (am.group('date'), normalize_suffix(am.group('suffix')))
                    acc_files[key].append(os.path.join(folder_path, f))

        pairs = []  # (date_str, suffix_label, tpath, apath)
        exact_keys = sorted(set(timing_files) & set(acc_files))
        for key in exact_keys:
            date_str, suffix = key
            pairs.append((date_str, suffix, timing_files.pop(key)[0], acc_files.pop(key)[0]))

        # relaxed fallback: same date, exactly one leftover timing + one leftover accuracy file
        leftover_t_dates = defaultdict(list)
        for (d, s), paths in timing_files.items():
            leftover_t_dates[d].append(((d, s), paths))
        leftover_a_dates = defaultdict(list)
        for (d, s), paths in acc_files.items():
            leftover_a_dates[d].append(((d, s), paths))
        for d in sorted(set(leftover_t_dates) & set(leftover_a_dates)):
            t_candidates = leftover_t_dates[d]
            a_candidates = leftover_a_dates[d]
            if len(t_candidates) == 1 and len(a_candidates) == 1 and \
               len(t_candidates[0][1]) == 1 and len(a_candidates[0][1]) == 1:
                tkey, tpaths = t_candidates[0]
                akey, apaths = a_candidates[0]
                notes_log.append([week_num, raw_name, d, os.path.basename(tpaths[0]),
                                   os.path.basename(apaths[0]),
                                   'paired by date only; filename suffixes differed (%r vs %r)'
                                   % (tkey[1], akey[1])])
                pairs.append((d, tkey[1], tpaths[0], apaths[0]))
                del timing_files[tkey]
                del acc_files[akey]

        for key, paths in timing_files.items():
            skip_log.append([week_num, raw_name, f'{key[0]} {key[1]}', 'timing:%d,accuracy:0' % len(paths),
                              'missing timing or accuracy file for this date/suffix'])
        for key, paths in acc_files.items():
            skip_log.append([week_num, raw_name, f'{key[0]} {key[1]}', 'timing:0,accuracy:%d' % len(paths),
                              'missing timing or accuracy file for this date/suffix'])

        seen_hashes = set()
        session_num = 0
        for date_str, suffix, tpath, apath in sorted(pairs):
            content_hash = hashlib.md5(open(tpath, 'rb').read()).hexdigest()
            if content_hash in seen_hashes:
                skip_log.append([week_num, raw_name, f'{date_str} {suffix}', tpath,
                                  'duplicate content of an already-processed session'])
                continue
            seen_hashes.add(content_hash)
            session_num += 1
            sessions.append((pid, raw_name, date_str, tpath, apath, session_num))

    return sessions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    participant_ids = build_participant_registry()

    with open(os.path.join(OUT_DIR, 'participants.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['participant_id', 'participant_name'])
        for name, pid in sorted(participant_ids.items(), key=lambda x: x[1]):
            w.writerow([pid, name])

    skip_log = []
    summary_log = []
    notes_log = []

    for week_num in range(1, 9):
        sessions = gather_week_sessions(week_num, participant_ids, skip_log, summary_log, notes_log)
        out_path = os.path.join(OUT_DIR, f'week_{week_num}.csv')
        n_rows = 0
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['week', 'participant_id', 'participant_name', 'session_date', 'session_num',
                        'phrase_index', 'phrase_text', 'repetition', 'key_seq', 'key',
                        'press_time', 'elapsed_seconds', 'release_time'])
            for pid, raw_name, date_str, tpath, apath, session_num in sessions:
                dd, mm, yyyy = date_str.split('-')
                iso_date = f'{yyyy}-{mm}-{dd}'
                try:
                    results, meta = parse_session(tpath, apath)
                except ValueError as e:
                    skip_log.append([week_num, raw_name, date_str, tpath, str(e)])
                    continue

                summary_log.append([week_num, pid, raw_name, iso_date, session_num,
                                     meta['n_valid_samples'], meta['n_cycles'],
                                     meta['kept_cycle_index'],
                                     meta['n_discarded_rows_other_cycles'],
                                     meta['truncated_accuracy_vs_blocks'],
                                     meta['unused_trailing_blocks'],
                                     meta['preamble_events_ignored']])

                for sample in results:
                    for key_seq, (t1, key, elapsed, t2) in enumerate(sample['events']):
                        press_time = parse_ts(t1)
                        release_time = parse_ts(t2)
                        elapsed_seconds = parse_elapsed_seconds(elapsed)
                        w.writerow([
                            week_num, pid, raw_name, iso_date, session_num,
                            sample['phrase_index'], sample['phrase_text'], sample['repetition'],
                            key_seq, normalize_key(key),
                            press_time.isoformat(sep=' '),
                            f'{elapsed_seconds:.7f}',
                            release_time.isoformat(sep=' '),
                        ])
                        n_rows += 1
        print(f'Week {week_num}: {len(sessions)} sessions -> {n_rows} key-event rows -> {out_path}')

    with open(os.path.join(OUT_DIR, 'sessions_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['week', 'participant_id', 'participant_name', 'session_date', 'session_num',
                    'n_valid_samples', 'n_cycles', 'kept_cycle_index', 'n_discarded_rows_other_cycles',
                    'truncated_accuracy_vs_blocks', 'unused_trailing_blocks', 'preamble_events_ignored'])
        w.writerows(summary_log)

    with open(os.path.join(OUT_DIR, 'skipped_sessions.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['week', 'participant_name', 'date_or_folder', 'detail', 'reason'])
        w.writerows(skip_log)

    with open(os.path.join(OUT_DIR, 'pairing_notes.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['week', 'participant_name', 'date', 'timing_file', 'accuracy_file', 'note'])
        w.writerows(notes_log)

    print(f'\n{len(participant_ids)} unique participants total.')
    print(f'{len(skip_log)} sessions/files skipped (see Output/skipped_sessions.csv).')
    print(f'{len(notes_log)} sessions paired via relaxed (date-only) matching (see Output/pairing_notes.csv).')


if __name__ == '__main__':
    main()
