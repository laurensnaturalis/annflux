# Copyright 2025 Intel Corporation
# Copyright 2025 Naturalis Biodiversity Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import line_profiler
from pandas import DataFrame
import logging
import os
import time
from collections import defaultdict, Counter
from joblib import Parallel, delayed
from typing import List, Dict, Iterable, Any

import numpy as np
import pandas
from numpy._typing import NDArray
from sklearn.decomposition import PCA

from annflux.tools.data import canon_

logger = logging.getLogger("annflux_training")
agg_to_pca = {}  # TODO: reset when features are updated


def _fit_pca_for_label(
    agg: str, 
    indices_: List[int], 
    features: NDArray, 
    new_labeled_indices: List[int] | None,
    logger: logging.Logger
) -> tuple[str, PCA] | None:
    """
    Helper function to fit PCA for a single label (for parallelization)
    """
    if new_labeled_indices is not None:
        if len(set(new_labeled_indices).intersection(set(indices_))) == 0:
            logger.debug(f"PCA: Skipping {agg} because not in new labeled images")
            return None
    feat_ = features[sorted(indices_)]
    if len(feat_) > 10:
        logger.debug(f"PCA: Updating {agg}")
        pca = PCA(n_components=0.95)
        pca.fit(feat_)
        return (agg, pca)
    return None


def compute_fre(
    annotations: Dict[str, str],
    data: pandas.DataFrame,
    features: NDArray,
    labeled_indices: List[int],
    test_uids: Iterable[str],
    new_labeled_indices: List[int] | None = None,
    pca_cache: dict | None = None,
):
    """
    Compute feature reconstruction error
    :param pca_cache: Optional dict to cache PCA models per label. If None, uses global module cache.
    """
    time_start = time.time()
    label_agg_array = np.array(
        [
            (annotations.get(uid) if annotations.get(uid) else None)
            if uid not in test_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ]
    )
    # TODO: check if everything is canonized
    # - map from label complex to data indices
    agg_to_indices: dict[str, list[int]] = defaultdict(lambda: [])
    for index_ in labeled_indices:
        if label_agg_array[index_] is not None:
            agg_to_indices[canon_(label_agg_array[index_])].append(index_)
    logger.info(f"[TIMING] PCA data preparation took {time.time() - time_start} s")
    # Use provided cache or fall back to global
    cache = pca_cache if pca_cache is not None else agg_to_pca
    # make PCA models (parallelized)
    time_start = time.time()
    pca_jobs = []
    for agg, indices_ in agg_to_indices.items():
        pca_jobs.append(delayed(_fit_pca_for_label)(
            agg, indices_, features, new_labeled_indices, logger
        ))
    
    # Use -1 for all CPUs, or set to specific number via environment variable
    n_jobs = int(os.getenv("FRE_N_JOBS", -1))
    results = Parallel(n_jobs=n_jobs)(pca_jobs)
    
    # Update cache with results
    for result in results:
        if result is not None:
            agg, pca = result
            cache[agg] = pca
    
    logger.info(f"pca analysis took={time.time() - time_start:.2f}")
    # compute FRE values
    time_start = time.time()
    column_name = "fre"
    fre_arr = np.full(len(data), np.nan)

    label_to_indices = {
        lbl: grp.index.values
        for lbl, grp in data[data.label_predicted.notna()].groupby("label_predicted", sort=False)
        if lbl in cache
    }
    for label_predicted_, indices_ in label_to_indices.items():
        pca_: PCA = cache[label_predicted_]
        feat_ = features[indices_]
        fre_arr[indices_] = np.linalg.norm(
            pca_.inverse_transform(pca_.transform(feat_)) - feat_, axis=1
        )

    data[column_name] = fre_arr

    logger.info(f"pca application took={time.time() - time_start:.2f}")
    data[column_name] /= data[column_name].max()
    data[column_name] = 1 - data[column_name]
    # data[column_name] = data[column_name].fillna(0)
    logger.info(f"[TIMING] FRE setting in DataFrame took {time.time() - time_start} s")
    #
    # - stratify by predicted label
    # apply only to unlabeled data
    # get count per label
    stratify_by_label(data, labeled_indices)


@line_profiler.profile
def stratify_by_label(data: DataFrame, labeled_indices: list[int]):
    time_start = time.time()
    count_per_label = Counter(
        data[~pandas.isna(data.label_true)]["label_true"].values.tolist()
    ).most_common()  # TODO: assumes label_true is canonized
    label_rare_to_common = [t_[0] for t_ in reversed(count_per_label)]

    labeled_indices_set = set(labeled_indices)
    label_predicted_arr = np.array([canon_(x_) for x_ in data["label_predicted"].values], dtype=object)

    # unlabeled mask
    unlabeled_mask = np.array(
        [i not in labeled_indices_set for i in range(len(data))], dtype=bool
    )
    has_label = label_predicted_arr != None  # noqa: E711
    candidate_mask = unlabeled_mask & has_label

    # sort candidates by fre ascending
    fre_vals = data["fre"].values.astype(float)
    candidate_indices = np.where(candidate_mask)[0]
    candidate_indices = candidate_indices[np.argsort(fre_vals[candidate_indices])]

    # group candidates by predicted label (already fre-sorted within each group)
    agg_to_indices_unlabeled: dict[str, list[int]] = defaultdict(list)
    for idx in candidate_indices:
        agg_to_indices_unlabeled[label_predicted_arr[idx]].append(int(idx))

    if len(agg_to_indices_unlabeled) > 0:
        # round-robin interleave using numpy: build position array directly
        max_len = max(len(v) for v in agg_to_indices_unlabeled.values())
        # pad each label's list to max_len with -1, stack, then read column-major
        label_order = [lbl_ for lbl_ in label_rare_to_common if lbl_ in agg_to_indices_unlabeled]
        padded = np.full((len(label_order), max_len), -1, dtype=np.intp)
        for row, lbl in enumerate(label_order):
            lst = agg_to_indices_unlabeled[lbl]
            padded[row, :len(lst)] = lst
        # column-major read gives the round-robin order; filter -1 sentinels
        interleaved = padded.T.ravel()
        indices_for_fre_strat = interleaved[interleaved >= 0]

        fre_strat_arr = np.full(len(data), len(indices_for_fre_strat) + 1, dtype=np.intp)
        fre_strat_arr[indices_for_fre_strat] = np.arange(len(indices_for_fre_strat))
        data["fre_strat"] = fre_strat_arr
    logger.info(f"[TIMING] FRE stratify by label took {time.time() - time_start} s")


def compute_nn_underrepresented(
    data: DataFrame,
    features: NDArray,
    labeled_indices: List[int],
    all_indices: NDArray,
    k_neighbors: int = 10,
) -> None:
    """Compute nn_underrepresented AL score.

    For each labeled class (leaf label in label_true), find up to k_neighbors
    unlabeled nearest neighbors in feature space. Assign a score so that
    neighbors of under-represented classes come first (lowest score = highest
    priority). Classes with fewer labeled examples are considered more
    under-represented.

    Writes column ``nn_underrepresented`` into *data* in-place.
    """
    t0 = time.time()

    labeled_set = set(labeled_indices)

    # --- count labeled examples per leaf class ----------------------------
    labeled_data = data.iloc[sorted(labeled_set)]
    label_true_col = labeled_data["label_true"].dropna()
    label_count: Counter = Counter()
    label_to_labeled_indices: dict[str, list[int]] = defaultdict(list)
    for row_idx, lt in zip(label_true_col.index, label_true_col.values):
        lt_c = canon_(lt)
        if lt_c is None:
            continue
        # take the most specific (longest) label in the comma-separated list
        parts = [p.strip() for p in lt_c.split(",") if p.strip()]
        if not parts:
            continue
        leaf = max(parts, key=lambda p: len(p))  # longest string ≈ most specific
        label_count[leaf] += 1
        label_to_labeled_indices[leaf].append(row_idx)

    logger.info(f"[TIMING] nn_underrepresented: label counting took {time.time() - t0:.3f} s, {len(label_count)} classes")

    if not label_count:
        data["nn_underrepresented"] = np.nan
        return

    # sort classes: rarest first
    classes_rarest_first = [lbl for lbl, _ in sorted(label_count.items(), key=lambda kv: kv[1])]

    t1 = time.time()
    # --- for each class, find k unlabeled nearest neighbors ---------------
    # all_indices shape: (N, K_total); row i = k nearest neighbor indices of sample i
    unlabeled_array = np.array(sorted(set(range(len(data))) - labeled_set))
    unlabeled_set = set(unlabeled_array.tolist())

    # class -> ordered list of unlabeled neighbor indices (by distance rank)
    class_to_neighbors: dict[str, list[int]] = {}
    for lbl, lbl_indices in label_to_labeled_indices.items():
        seen: dict[int, int] = {}  # neighbor_idx -> min rank
        for src_idx in lbl_indices:
            for rank, nb_idx in enumerate(all_indices[src_idx]):
                nb_idx = int(nb_idx)
                if nb_idx not in unlabeled_set:
                    continue
                if nb_idx not in seen or rank < seen[nb_idx]:
                    seen[nb_idx] = rank
        # sort by best (lowest) rank, take top-k
        ordered = sorted(seen.items(), key=lambda kv: kv[1])[:k_neighbors]
        class_to_neighbors[lbl] = [idx for idx, _ in ordered]

    logger.info(f"[TIMING] nn_underrepresented: neighbor lookup took {time.time() - t1:.3f} s")

    t2 = time.time()
    # --- assign scores: interleave by class, rarest first -----------------
    # score = position in the final interleaved list
    assigned: dict[int, int] = {}  # data row index -> score
    position = 0
    queues = {lbl: list(class_to_neighbors.get(lbl, [])) for lbl in classes_rarest_first}
    while any(queues.values()):
        for lbl in classes_rarest_first:
            if queues[lbl]:
                nb_idx = queues[lbl].pop(0)
                if nb_idx not in assigned:
                    assigned[nb_idx] = position
                    position += 1

    default_score = position  # items not covered get a high score
    data["nn_underrepresented"] = default_score
    update_column_fast("nn_underrepresented", data, list(assigned.items()))

    logger.info(
        f"[TIMING] nn_underrepresented: score assignment took {time.time() - t2:.3f} s, "
        f"{len(assigned)} unlabeled items scored, total {time.time() - t0:.3f} s"
    )


def update_column_fast(column_name: str, data: DataFrame, update_for_key: list[Any]):
    if len(update_for_key) > 0:
        update_for_key = sorted(update_for_key, key=lambda t_: t_[0])
        indices, values = zip(*update_for_key)
        if not data[column_name].values.flags["OWNDATA"]:  # need in test environment
            current_values = data[column_name].values.copy()
        else:
            current_values = data[column_name].values
        current_values[np.array(indices)] = values
        data[column_name] = current_values
