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
from pandas.core.api import DataFrame
import logging
import time
from collections import defaultdict, Counter
from typing import List, Dict, Iterable, Any

import numpy as np
import pandas
from numpy._typing import NDArray
from sklearn.decomposition import PCA
from tqdm import tqdm

from annflux.tools.data import canon_

logger = logging.getLogger("annflux_server")
agg_to_pca = {}  # TODO: reset when features are updated


def compute_fre(
    annotations: Dict[str, str],
    data: pandas.DataFrame,
    features: NDArray,
    labeled_indices: List[int],
    test_uids: Iterable[str],
    new_labeled_indices: List[int] | None = None,
):
    """
    Compute feature reconstruction error
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
    # make PCA models
    time_start = time.time()
    for agg, indices_ in tqdm(agg_to_indices.items()):
        if new_labeled_indices is not None:
            if len(set(new_labeled_indices).intersection(set(indices_))) == 0:
                logger.debug(f"PCA: Skipping {agg} because not in new labeled images")
                continue
        feat_ = features[sorted(indices_)]
        if len(feat_) > 10:
            logger.debug(f"PCA: Updating {agg}")
            pca = PCA(n_components=0.95)
            pca.fit(feat_)
            agg_to_pca[agg] = pca
    logger.info(f"pca analysis took={time.time() - time_start:.2f}")
    # compute FRE values
    time_start = time.time()
    column_name = "fre"
    data[column_name] = None
    update_for_key = []
    for label_predicted_ in data.label_predicted.unique():
        if pandas.isna(label_predicted_):
            continue
        indices_ = np.where(data.label_predicted == label_predicted_)[0]

        features_ = features[indices_]

        if label_predicted_ in agg_to_pca:
            pca: PCA = agg_to_pca[label_predicted_]
            fre = np.linalg.norm(
                pca.inverse_transform(pca.transform(features_)) - features_, axis=1
            )
            logger.debug(f"{label_predicted_=}, {fre.min()=}, {fre.max()=}")
            # data.loc[indices_, column_name] = fre
            for index_, fre_val in zip(indices_, fre):
                update_for_key.append((index_, fre_val))


    # update data frame
    update_column_fast(column_name, data, update_for_key)

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
    label_rare_to_common = list(reversed([t_[0] for t_ in count_per_label]))
    low_to_high_fre_indices = data.sort_values(by="fre", ascending=True).index
    # label aggregate to FRE sorted index
    agg_to_indices_unlabeled: dict[str, list[int]] = defaultdict(lambda: [])
    label_predicted = [canon_(x_) for x_ in data["label_predicted"].values]
    labeled_indices_set = set(labeled_indices)
    for index_ in low_to_high_fre_indices:
        if index_ not in labeled_indices_set and label_predicted[index_] is not None:
            agg_to_indices_unlabeled[label_predicted[index_]].append(index_)
    #
    if len(agg_to_indices_unlabeled) > 0:
        indices_for_fre_strat: list[int] = []
        for block in range(max([len(list_) for list_ in agg_to_indices_unlabeled.values()])):
            # pick the lowest FRE value first
            for label in label_rare_to_common:
                if len(agg_to_indices_unlabeled[label]) > 0:
                    indices_for_fre_strat.append(agg_to_indices_unlabeled[label].pop(0))

        update_for_key = []
        for i, index_ in enumerate(indices_for_fre_strat):
            update_for_key.append((index_, i))
        data["fre_strat"] = len(indices_for_fre_strat) + 1
        update_column_fast("fre_strat", data, update_for_key)
        #
    logger.info(f"[TIMING] FRE stratify by label took {time.time() - time_start} s")


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
