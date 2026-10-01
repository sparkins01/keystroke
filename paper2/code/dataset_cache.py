"""Disk cache for the Dataset object.

`Dataset.__init__` recomputes a feature vector and a raw sequence for all 94,894
samples, which costs minutes. M4 training touches it once per fold per seed, so
without a cache the rebuild dominates the run.
"""
import os
import pickle

import data_loading
from dataset import Dataset

CACHE = os.path.join(os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'dataset_cache.pkl')


def load_dataset(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    ds = Dataset(data_loading.load_samples(verbose=False))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(ds, f, protocol=4)
    return ds


if __name__ == '__main__':
    import time
    t0 = time.time()
    ds = load_dataset()
    print(f'{len(ds.samples):,} samples, {len(ds.password_ids)} passwords '
          f'[{time.time() - t0:.0f}s]')
