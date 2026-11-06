"""
Derive the fixed 40-password metadata table (canonical surface form, length,
substitution category) directly from the parsed event-level CSVs.

The per-participant *order* in which the 40 passwords are presented is shuffled
(each participant sees their own fixed random order, stable across their 8 weeks),
but the 40 underlying passwords themselves -- and the substitution pattern applied
to each -- are the same for the whole population. We recover the canonical set by
grouping every observed `phrase_text` by its alphabetic skeleton (letters only,
lower-cased) and taking the most frequent surface form for each skeleton (this
also silently repairs one single-participant encoding artefact, in one
participant's Week 1 data:
"gard\ufffdn", where the source file corrupted the '?' character into a UTF-8
replacement character).

Substitution category follows the paper's five buckets:
  none        - all lowercase letters, no digits/symbols
  uppercase   - contains an uppercase letter, nothing else non-letter
  numeric     - contains a digit, nothing else non-letter
  symbol      - contains a non-alnum symbol, nothing else non-letter
  combination - two or more of the above kinds present
"""
import csv
import gzip
import os
import re
from collections import Counter, defaultdict

OUTPUT_DIR = os.environ.get('KD_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'))
RESULT_DIR = os.environ.get('KD_ADVERSARY_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


def skeleton(text):
    return re.sub(r'[^a-zA-Z]', '', text).lower()


def classify_substitution(surface):
    has_upper = any(c.isupper() for c in surface)
    has_digit = any(c.isdigit() for c in surface)
    has_symbol = any(not c.isalnum() for c in surface)
    n_kinds = sum([has_upper, has_digit, has_symbol])
    if n_kinds == 0:
        return 'none'
    if n_kinds >= 2:
        return 'combination'
    if has_upper:
        return 'uppercase'
    if has_digit:
        return 'numeric'
    return 'symbol'


def build_password_metadata(output_dir=OUTPUT_DIR):
    surface_counts = defaultdict(Counter)
    for week in range(1, 9):
        path = os.path.join(output_dir, f'week_{week}.csv.gz')
        if not os.path.exists(path):
            path = os.path.join(output_dir, f'week_{week}.csv')
        if not os.path.exists(path):
            continue
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, mode='rt', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            seen_samples = set()
            for row in reader:
                key = (row['participant_id'], row['session_num'], row['phrase_index'], row['repetition'])
                if key in seen_samples:
                    continue
                seen_samples.add(key)
                surface_counts[skeleton(row['phrase_text'])][row['phrase_text']] += 1

    rows = []
    for skel, counts in sorted(surface_counts.items()):
        canonical_surface = counts.most_common(1)[0][0]
        rows.append({
            'password_id': skel,
            'surface_form': canonical_surface,
            'length': len(canonical_surface),
            'substitution_category': classify_substitution(canonical_surface),
        })
    rows.sort(key=lambda r: (r['length'], r['substitution_category'], r['password_id']))
    return rows


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = build_password_metadata()
    out_path = os.path.join(RESULT_DIR, 'password_metadata.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['password_id', 'surface_form', 'length', 'substitution_category'])
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} passwords -> {out_path}')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
