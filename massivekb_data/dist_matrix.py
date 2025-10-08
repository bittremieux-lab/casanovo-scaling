import os
from multiprocessing import Pool, cpu_count

import Levenshtein
import numpy as np
from tqdm import tqdm

# Global variable for read-only access in worker processes
_global_seqs1 = None
_global_seqs2 = None


def _square_init_pool(sequences):
    global _global_seqs1
    _global_seqs1 = sequences


def _rect_init_pool(seqs1, seqs2):
    global _global_seqs1, _global_seqs2
    _global_seqs1 = seqs1
    _global_seqs2 = seqs2


def _square_compute_row(i):
    global _global_seqs1
    n = len(_global_seqs1)
    row = np.zeros(n - i - 1, dtype=np.uint8)
    for offset, j in enumerate(range(i + 1, n)):
        d = Levenshtein.distance(_global_seqs1[i], _global_seqs1[j])
        d = int(100 * d / max(len(_global_seqs1[i]), len(_global_seqs1[j])))
        row[offset] = d
    return i, row


def _rect_compute_row(i):
    s1 = _global_seqs1[i]
    row = np.zeros(len(_global_seqs2), dtype=np.uint8)
    for j, s2 in enumerate(_global_seqs2):
        row[j] = Levenshtein.distance(s1, s2)
    return i, row


def square_dist_matrix(sequences, cache_dir=None, n_threads=cpu_count()):
    n = len(sequences)
    cache_file = os.path.join(cache_dir, f"square_dist_matrix_{n}.npy")

    if cache_dir is not None:
        if os.path.exists(cache_file):
            return np.load(cache_file)

    dist_matrix = np.zeros((n, n), dtype=np.uint8)

    with Pool(
        n_threads, initializer=_square_init_pool, initargs=(sequences,)
    ) as pool:
        for i, row in tqdm(
            pool.imap_unordered(_square_compute_row, range(n - 1)), total=n - 1
        ):
            dist_matrix[i, i + 1 :] = row
            dist_matrix[i + 1 :, i] = row

    if cache_dir is not None:
        np.save(cache_file, dist_matrix)

    return dist_matrix


def rectangle_distance_matrix(seqs1, seqs2, n_threads=cpu_count()):
    # We do not cache here, this is handled by the analysis.add_all_to_split function
    n1 = len(seqs1)
    n2 = len(seqs2)

    dist_matrix = np.zeros((n1, n2), dtype=np.uint8)

    # Create the pool and initialize globals
    with Pool(
        n_threads, initializer=_rect_init_pool, initargs=(seqs1, seqs2)
    ) as pool:
        for i, row in tqdm(
            pool.imap_unordered(_rect_compute_row, range(n1)), total=n1
        ):
            dist_matrix[i, :] = row

    return dist_matrix
