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
import json
import logging
import os
import time
from collections import defaultdict
from typing import Set, Any, Tuple, List

import faiss
import numpy as np
import pandas
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.algorithms.feature_reconstruction_error import compute_fre
from annflux.algorithms.most_needed import compute_most_needed, compute_near_labeled
from annflux.performance.basic import (
    compute_performance,
    write_performance_key_val,
    get_performance_key_val,
)
from annflux.tools.core import AnnFluxState
from annflux.tools.data import canon_, color_and_label
from annflux.tools.io import numpy_load
from annflux.tools.mixed import remove_sys
from annflux.training.annflux.group_classifier_cnn import (
    classify_path as group_classify,
)

min_prev_near_labeled_perc = 0.99  # TODO: configurable


def group_classification(
    features,
    annflux_path_or_data: str | pandas.DataFrame,
    feature_image_out_folder,
    group_data_path_or_data: str | pandas.DataFrame,
    test_uids,
) -> tuple[NDArray, float, pandas.DataFrame]:
    """
    :return features, accuracy, out_table
    """
    return group_classify(
        features,
        annflux_path_or_data,
        feature_image_out_folder,
        group_data_path_or_data,
        test_uids,
    )


def quick_reclassification(
    state: AnnFluxState, logger: logging.Logger, knn_type="quick", group=False
):
    if not group:
        quick_reclassification_instance(knn_type, state, logger)
    else:
        quick_reclassification_group(knn_type, state, logger)


def quick_reclassification_instance(knn_type, state, logger: logging.Logger):
    """
    Trains a quick new model using kNN
    """
    start_time = time.time()
    annotations, data, result_set, test_indices, test_uids = load_data(state, logger)

    state.all_distances, state.all_indices = compute_knn(
        state.features,
        state.features,
        110,
        result_set.get_path_for("knn_results.npz"),
        state,
        logger,
    )

    has_dp_cluster = "dp_cluster" in data.columns
    dp_most_needed_idx: NDArray
    dp_distances: NDArray
    dp_indices: NDArray
    if has_dp_cluster:
        # compute 1-nearest-neighbor to density peak clusters
        dp_most_needed_idx = data[
            (data["dp_most_needed"] < data["dp_most_needed"].max())
            & (data["in_test"] == 0)
        ].index.values
        dp_distances, dp_indices = compute_knn(
            state.features[dp_most_needed_idx],
            state.features,
            1,  # nearest cluster center (cf. "mean")
            result_set.get_path_for("knn_results_dp.npz"),
            state,
            logger,
        )

    # Compute reload_features BEFORE updating state.cache_for
    reload_features = state.cache_for != result_set.entry.uid
    
    state.cache_for = result_set.entry.uid
    state.version_for_recompute = (
        result_set.entry.uid + "_" + str(state.trained_for_version_previous)
    )

    time_start = time.time()

    print("knn labeled", time.time() - start_time)
    new_labeled_nn_uids = None

    # FRE PCA cache - reset when features change
    fre_pca_cache: dict = {}

    if len(state.labeled_indices) > 0:
    #     return
        distances, indices = (
            state.all_distances[state.labeled_indices],
            state.all_indices[state.labeled_indices],
        )
        quicker_updates = int(
            os.getenv("QUICKER_UPDATES", 0)
        )  # 0 = off, 1 = display only, 2 = display + knn
        quicker_updates = (
            quicker_updates
            if (state.new_labeled_uids is not None and len(state.new_labeled_uids) > 0)
            else 0
        )

        # - figure out neighbors of updated uids
        new_labeled_nn_idx: Set[int] = set()
        if quicker_updates:
            # get indices of new_labeled_uids
            new_labeled_idx = data[data.uid.isin(state.new_labeled_uids)].index
            logger.info(f"new_labeled_uids={state.new_labeled_uids}")
            logger.info(f"new_labeled_idx={new_labeled_idx}")
            # include the nearest neighbors of the newly labeled UIDs
            new_labeled_nn_idx = set(state.all_indices[new_labeled_idx].flatten())
            logger.info(f"new_labeled_nn_idx={len(new_labeled_nn_idx)}")
            new_labeled_nn_uids = data[data.index.isin(new_labeled_nn_idx)].uid
            logger.info(f"new_labeled_nn_uids={new_labeled_nn_uids}")
        #
        if quicker_updates:
            if "score_possible" not in data.columns:
                data["label_predicted"] = None
                data["scores_predicted"] = None
                data["label_possible"] = ""
                data["score_possible"] = None
            else:
                data.loc[list(new_labeled_nn_idx), "label_predicted"] = None
                data.loc[list(new_labeled_nn_idx), "scores_predicted"] = None
                data.loc[list(new_labeled_nn_idx), "label_possible"] = ""
                data.loc[list(new_labeled_nn_idx), "score_possible"] = None
        else:
            data["label_predicted"] = None
            data["scores_predicted"] = None
            data["label_possible"] = None
            data["score_possible"] = None

        # -- make predictions for labeled indices
        logger.info(f"Using knn_rank_exponent={state.knn_rank_exponent}")
        indices_ = indices
        distances_ = distances
        labeled_indices_ = state.labeled_indices
        logger.info(f"labeled_indices_={len(labeled_indices_)}")
        new_labeled_indices_for_fre = None
        if quicker_updates > 1:
            idx_sel = [
                i_
                for i_, idx_ in enumerate(state.labeled_indices)
                if idx_ in new_labeled_nn_idx
            ]
            indices_ = indices_[idx_sel]
            distances_ = distances_[idx_sel]
            labeled_indices_ = np.array(labeled_indices_)[idx_sel].tolist()
            new_labeled_indices_for_fre = labeled_indices_
            logger.info(
                f"quicker_updates: labeled_indices_ before test_indices={len(labeled_indices_)}"
            )
            labeled_indices_.extend(test_indices)
            logger.info(f"quicker_updates: labeled_indices_={len(labeled_indices_)}")
        state.g_quick_status = "predicting labeled"
        logger.info(f"make_predictions: labeled_indices_={len(labeled_indices_)}")
        make_predictions(
            data,
            indices_,
            distances_,
            state.label_array,
            labeled_indices_,
            skip_first=True,
            knn_rank_exponent=state.knn_rank_exponent,
            tag="predicting labeled",
        )

        #
        has_density_peak = "dp_most_needed" in data.columns
        if not has_density_peak:
            prev_near_labeled_perc = get_performance_key_val(
                state.performance_path, "percentage_near_labeled", -1.0
            )
            if prev_near_labeled_perc < min_prev_near_labeled_perc:
                state.g_quick_status = "computing most needed"
                counter_of_most_need, near_labeled_indices, near_labeled_perc = (
                    compute_most_needed(
                        state.all_indices, state.labeled_indices, state.features
                    )
                )
                write_performance_key_val(
                    state.performance_path, "percentage_near_labeled", near_labeled_perc
                )
            else:
                logger.warning(
                    f"Skipping most needed because {prev_near_labeled_perc=}<{min_prev_near_labeled_perc}"
                )
                near_labeled_indices = np.arange(len(state.all_distances))
                counter_of_most_need = {}
        else:
            near_labeled_indices, near_labeled_perc = compute_near_labeled(
                state.all_indices, state.labeled_indices
            )
            counter_of_most_need = {}
        # predictions for near labeled
        time_start = time.time()
        distances, indices = (
            state.all_distances[near_labeled_indices],
            state.all_indices[near_labeled_indices],
        )
        logger.info(f"knn near_labeled={time.time() - time_start}")
        data["entropy"] = -np.inf
        data["al_measure"] = len(data) + 1
        data["most_needed"] = len(data) + 1
        # - make predictions for near labeled
        indices_ = indices
        distances_ = distances
        near_labeled_indices_ = near_labeled_indices
        if quicker_updates > 1:
            idx_sel = [
                i_
                for i_, idx_ in enumerate(near_labeled_indices)
                if idx_ in new_labeled_nn_idx
            ]
            indices_ = indices_[idx_sel]
            distances_ = distances_[idx_sel]
            near_labeled_indices_ = np.array(near_labeled_indices)[idx_sel].tolist()
            near_labeled_indices_.extend(test_indices)
        state.g_quick_status = "computing predictions"
        logger.info(f"make_predictions: near_labeled_indices_={len(near_labeled_indices_)}")
        make_predictions(
            data,
            indices_,
            distances_,
            state.label_array,
            near_labeled_indices_,
            knn_rank_exponent=state.knn_rank_exponent,
            tag="predicting near labelled",
        )
        data.label_predicted = data.label_predicted.apply(lambda x_: canon_(x_))
        if "label_true" in data.columns:
            data.label_true = data.label_true.apply(lambda x_: canon_(x_))
        # FRE
        state.g_quick_status = "computing FRE"
        # Reset PCA cache if features were reloaded
        if reload_features:
            fre_pca_cache.clear()
        compute_fre(
            annotations,
            data,
            state.features,
            state.labeled_indices,
            test_uids,
            new_labeled_indices_for_fre,
            pca_cache=fre_pca_cache,
        )
        #
        most_needed_first = sorted(counter_of_most_need.items(), key=lambda t_: -t_[1])
        for i_, (most_needed_i, _) in enumerate(most_needed_first):
            data.at[most_needed_i, "most_needed"] = i_
            if i_ > 500:  # TODO(improvement): based on actual page size
                break
        #
        if has_density_peak:
            logger.info("Using dp_most_needed for most needed")
            data["direct_most_needed"] = data["most_needed"]
            data["most_needed"] = data["dp_most_needed"]

            labeled_uids = set(annotations.keys())
            data["labeled"] = data["uid"].apply(lambda x_: int(x_ in labeled_uids))
            data_ = data[data["dp_most_needed"] < data["dp_most_needed"].max()]
            # print("bloep", data_[data["labeled"] == 0])
            logger.info(f"|data most needed| = {len(data_)}")
            near_labeled_perc = len(data_[data["labeled"] == 1]) / len(data_)
            print(f"has_density_peak: {near_labeled_perc=}")

            write_performance_key_val(
                state.performance_path, "percentage_near_labeled", near_labeled_perc
            )
        #
        # use DP cluster to predict unpredicted
        if has_dp_cluster and len(annotations) > 0:
            state.g_quick_status = "computing predictions for unpredicted using DP cluster"
            unpredicted_idx = data[
                pandas.isna(data.label_predicted) & (pandas.isna(data.label_possible))
            ].index.values.tolist()
            make_predictions(
                data,
                dp_indices[unpredicted_idx],
                dp_distances[unpredicted_idx],
                state.label_array[dp_most_needed_idx],
                unpredicted_idx,
                knn_rank_exponent=state.knn_rank_exponent,
                tag="computing predictions for unpredicted using DP cluster",
            )
            unpredicted_idx = data[
                pandas.isna(data.label_predicted) & (pandas.isna(data.label_possible))
            ].index.values
            print(f"{len(unpredicted_idx)=} after make_predictions")
        logger.info(f"make_predictions end={time.time()}")

        write_performance_key_val(
            state.performance_path,
            "percentage_labeled_possible",
            1
            - len(
                data[
                    pandas.isna(data["label_possible"])
                    & pandas.isna(data["label_predicted"])
                ]
            )
            / len(data),
        )
        #
        state.g_quick_status = "computing performance"
        labeled_predicted_test_data = data[
            (data.in_test == 1)
            & (data.labeled == 1)
            & ~pandas.isna(data.label_predicted)
            & ~pandas.isna(data.label_true)
        ]
        predicted_test = [
            canon_(x_, remove_unknown=True, remove_sys=True)
            for x_ in labeled_predicted_test_data.label_predicted
        ]

        predicted_test = [x_.split(",") if x_ is not None else [] for x_ in predicted_test]
        true_test = [
            canon_(x_, remove_unknown=True, remove_sys=True)
            for x_ in labeled_predicted_test_data.label_true
        ]
        true_test = [x_.split(",") if x_ is not None else [] for x_ in true_test]
        logger.info(f"|predicted_test|={len(predicted_test)}")
        print(f"{predicted_test=}, {true_test=}")
        compute_performance(
            predicted_test,
            true_test,
            annotations,
            data,
            len(state.labeled_indices),
            performance_graph_path=os.path.join(state.annflux_folder, "performance.json"),
            detailed_performance_path=os.path.join(
                state.annflux_folder, "detailed_performance.csv"
            )
        )
    state.g_quick_status = "coloring and labelling"

    class_to_color, class_to_count = color_and_label(
        data,
        annotations,
        json.load(open(os.path.join(state.annflux_folder, "label_defs.json")))[
            "labels"
        ],
        display_update_uids=new_labeled_nn_uids
        if new_labeled_nn_uids is not None and len(new_labeled_nn_uids) > 0
        else None,
        logger=logger,
    )

    with open(state.doublecheck_path) as f:
        double_checked = set(json.load(f)["checked"])
        data["double_checked"] = data["uid"].apply(
            lambda x_: int(x_ in set(double_checked))
        )
    data.to_csv(state.annflux_path, index=False)
    logger.info(
        f"no prediction={len(data[(data.score_predicted == 0) & (data.labeled == 0)])}"
    )
    logger.info(f"instant_reclassification done = {time.time() - start_time}")

    make_class_to_color(
        class_to_count,
        class_to_color,
        os.path.join(state.project_folder, "annflux", "class_to_color.csv"),
    )
    state.g_quick_status = "idle"


def make_class_to_color(class_cluster_to_count, class_to_color, out_path):
    pandas.DataFrame(
        data=zip(class_to_color.keys(), class_to_color.values()),
        columns=("class", "color"),  # ty: ignore
    ).merge(
        pandas.DataFrame(
            data=zip(class_cluster_to_count.keys(), class_cluster_to_count.values()),
            columns=("class", "count"),  # ty: ignore
        ),
        on="class",
        how="left",
    ).to_csv(out_path)


def load_data(state: AnnFluxState, logger: logging.Logger, no_linear_features=False):
    start_time = time.time()
    repo = Repository(os.path.join(state.annflux_folder, "datarepo"))
    result_set = None
    if no_linear_features:
        for resultset in repo.get(label=Resultset, tag="unseen")[::-1]:
            print(resultset, resultset.entry, resultset.entry.message)
            if "linear" not in resultset.entry.message:
                result_set = resultset
                break
    else:
        result_set = repo.get(label=Resultset, tag="unseen").last()
    if result_set is None:
        raise RuntimeError("result_set cannot be none")
    print(f"load data {result_set.entry.message=}")
    folder = result_set.path
    data = pandas.read_csv(
        os.path.join(state.project_folder, "annflux", "annflux.csv"),
        dtype={
            "label_predicted": str,
            "score_true": float,
            "uid": str,
            "score_possible": str,
        },
    )
    data.reset_index(drop=True, inplace=True)
    logger.info(f"instant_reclassification 2={time.time() - start_time}")
    with open(state.labels_path) as f:
        annotations = json.load(f)
    with open(os.path.join(state.annflux_folder, "split.json")) as f:
        test_uids = set(json.load(f)["test"])
    logger.info(f"instant_reclassification 3={time.time()}")
    reload_features = state.cache_for != result_set.entry.uid
    logger.info(
        f"{result_set.entry.uid=},{state.trained_for_version_previous=}"
    )
    if reload_features:
        state.g_quick_status = "Loading features"
        state.features = numpy_load(f"{folder}/last_full.npz", "lastFull")
    assert len(data) == len(state.features), f"{len(data)=}, {len(state.features)=}"
    annotated_uids = set(annotations.keys())
    most_needed_first = annotated_uids - test_uids
    state.labeled_indices = np.array(
        sorted([i for i, uid in enumerate(data.uid.values) if uid in most_needed_first])
    )
    logger.info(f"instant_reclassification 4={time.time() - start_time}")
    #
    set_data_undetermined(annotations, data)
    #
    set_state_label_array(annotations, data, state, test_uids)
    logger.debug(f"state.label_array_test={state.label_array_test}")
    logger.debug(f"|label_array_test|={len(state.label_array_test)}")
    test_indices = set([i for i, uid in enumerate(data.uid.values) if uid in test_uids])
    logger.debug(f"|test_uids|={len(test_uids)}")
    logger.debug(f"|test_indices|={len(test_indices)}")
    state.labeled_test_indices = np.array(
        sorted(
            [
                i
                for i, uid in enumerate(data.uid.values)
                if uid in test_uids and uid in annotations
            ]
        )
    )
    logger.debug(f"|labeled_test_indices|={len(state.labeled_test_indices)}")
    logger.debug(f"instant_reclassification 5={time.time() - start_time}")
    data["label_true"] = np.array(
        [canon_(annotations.get(uid)) for uid in data.uid.values]  # noqa
    )
    return annotations, data, result_set, test_indices, test_uids


def set_state_label_array(annotations, data, state, test_uids):
    state.label_array = np.array(
        [
            (annotations.get(uid).split(",") if annotations.get(uid) else None)
            if uid not in test_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ],
        dtype=object,
    )
    state.label_array = np.array(
        [remove_sys(x_) for x_ in state.label_array], dtype=object
    )
    state.label_array_test = np.array(
        [
            (annotations.get(uid).split(",") if annotations.get(uid) else None)
            if uid in test_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ],
        dtype=object,
    )


def set_data_undetermined(annotations, data):
    data["label_undetermined"] = None
    for i, uid in enumerate(data.uid.values):
        if annotations.get(uid):
            labels = annotations.get(uid).split(",")
            sure_labels = []
            for label_ in labels:
                if label_.endswith("=?"):
                    data.at[i, "label_undetermined"] = (
                        label_
                        if pandas.isna(data.at[i, "label_undetermined"])
                        else data.at[i, "label_undetermined"] + "," + label_
                    )
                else:
                    sure_labels.append(label_)
            annotations[uid] = ",".join(sorted(sure_labels))


def compute_knn(
    features_train: NDArray,
    features_test: NDArray,
    k,
    knn_results_path,
    state,
    logger,
):
    if not os.path.exists(knn_results_path):
        state.g_quick_status = "computing knn index"
        knn_index = faiss.index_factory(  # ty:ignore[possibly-missing-attribute]
            features_train.shape[1],
            "Flat",
            {"inner": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}[
                "l2"
            ],  # ty:ignore[possibly-missing-attribute]
        )
        features_train *= 1 - 1e-2 * np.random.rand(
            features_train.shape[0], features_train.shape[1]
        )
        print(features_train.shape)

        knn_index.train(features_train)
        knn_index.add(features_train)

        # state.all_distances, state.all_indices = knn_index.search(state.features, k=k)
        all_distances, all_indices = knn_index.search(features_test, k=k)

        np.savez(
            knn_results_path,
            all_distances=all_distances,
            all_indices=all_indices,
        )
    else:
        logger.info(f"Loading kNN results from {knn_results_path}")
        knn_results = np.load(knn_results_path)
        all_distances, all_indices = (
            knn_results["all_distances"],
            knn_results["all_indices"],
        )
    return all_distances, all_indices


def quick_reclassification_group(knn_type, state, logger):
    """
    Trains a quick new model using kNN
    """
    start_time = time.time()
    group_annflux_path = os.path.join(state.annflux_folder, "group0_annflux.csv")
    data = pandas.read_csv(
        group_annflux_path,
        dtype={"label_predicted": str, "score_true": float, "uid": str},
    )
    with open(state.labels_path) as f:
        annotations = json.load(f)
    # with open(os.path.join(state.working_folder, "split.json")) as f:
    #     test_uids = set(json.load(f)["test"])

    state.group_features = np.load(
        os.path.join(state.annflux_folder, "group0_features.npz")
    )["lastFull"]
    assert len(data) == len(state.group_features), (
        f"{len(data)=}, {len(state.group_features)=}"
    )
    # noinspection PyArgumentList
    uids, labels = zip(*annotations.items())
    train_uids, test_uids = train_test_split(uids, test_size=0.10, stratify=labels)
    test_uids = set(test_uids)
    train_uids = set(train_uids)

    state.group_label_array = get_label_array(annotations, data, train_uids)
    state.group_label_array_test = get_label_array(annotations, data, test_uids)
    logger.info(f"|group_label_array_test|={len(state.group_label_array_test)}")
    test_indices = set([i for i, uid in enumerate(data.uid.values) if uid in test_uids])
    logger.info(f"|test_uids|={len(test_uids)}")
    logger.info(f"|test_indices|={len(test_indices)}")
    state.group_labeled_test_indices = sorted(
        [
            i
            for i, uid in enumerate(data.uid.values)
            if uid in test_uids and uid in annotations
        ]
    )
    state.group_labeled_indices = sorted(
        [i for i, uid in enumerate(data.uid.values) if uid in annotations]
    )
    logger.info(f"|group_labeled_test_indices|={len(state.group_labeled_test_indices)}")
    k = 30  # the magic number that should be investigated

    state.g_quick_status = "computing group knn index"
    knn_index = faiss.index_factory(  # ty:ignore[possibly-missing-attribute]
        state.group_features.shape[1],
        "Flat",
        {"inner": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}[
            "l2"
        ],  # ty:ignore[possibly-missing-attribute]
    )
    state.group_features *= 1 - 1e-2 * np.random.rand(
        state.group_features.shape[0], state.group_features.shape[1]
    )

    knn_index.train(state.group_features)
    knn_index.add(state.group_features)

    state.all_distances_group, state.all_indices_group = knn_index.search(
        state.group_features, k=k
    )

    near_labeled_indices = np.array(range(len(data)))  # TODO
    data["label_predicted"] = None
    data["scores_predicted"] = None
    data["label_possible"] = None
    data["score_possible"] = None

    # - make predictions for labeled indices
    labeled_indices = state.group_labeled_indices
    distances, indices = (
        state.all_distances_group[labeled_indices],
        state.all_indices_group[labeled_indices],
    )
    logger.info(f"group labeled_indices={len(labeled_indices)}")
    state.g_quick_status = "predicting group labeled"
    make_predictions(
        data,
        indices,
        distances,
        state.group_label_array,
        labeled_indices,
        skip_first=True,
        knn_rank_exponent=state.knn_rank_exponent,
        tag="predicting group labeled",
    )

    # - make predictions for near labeled
    distances, indices = (
        state.all_distances_group[near_labeled_indices],
        state.all_indices_group[near_labeled_indices],
    )
    indices_ = indices
    distances_ = distances
    near_labeled_indices_ = near_labeled_indices

    state.g_quick_status = "computing predictions"
    make_predictions(
        data,
        indices_,
        distances_,
        state.group_label_array,
        near_labeled_indices_.tolist(),
        knn_rank_exponent=state.knn_rank_exponent,
        tag="predicting group near labelled",
    )
    logger.info(f"make_predictions end={time.time()}")
    data.label_predicted = data.label_predicted.apply(lambda x_: canon_(x_))
    data.label_true = data.label_true.apply(lambda x_: canon_(x_))
    #
    # compute_performance(out_predicted_test, out_true_test, state, annotations, data)
    state.g_quick_status = "coloring and labelling"
    class_to_color = color_and_label(
        data,
        annotations,
        json.load(open(os.path.join(state.annflux_folder, "label_defs.json")))[
            "labels"
        ],
    )

    data.to_csv(group_annflux_path, index=False)
    logger.info(
        f"no prediction={len(data[(data.score_predicted == 0) & (data.labeled == 0)])}"
    )
    logger.info(f"quick_reclassification_group done = {time.time() - start_time}")
    pandas.DataFrame(
        data=zip(class_to_color.keys(), class_to_color.values()),
        columns=("class", "color"),  # ty: ignore[invalid-argument-type]
    ).to_csv(os.path.join(state.project_folder, "annflux", "class_to_color_group.csv"))
    state.g_quick_status = "idle"


def get_label_array(annotations, data, include_uids):
    return np.array(
        [
            (
                remove_sys(annotations.get(uid).split(","))
                if annotations.get(uid)
                else None
            )
            if uid in include_uids
            else None
            for i, uid in enumerate(data.uid.values)
        ],
        dtype=object,
    )


def make_predictions(
    data: pandas.DataFrame,
    indices: NDArray,
    distances: NDArray,
    train_labels: NDArray,
    data_indices: List[int] | NDArray,
    skip_first=False,
    knn_rank_exponent=0.5,
    tag=None,
):
    """
    The predictions are made for the knn results in (`indices`, `distances`) which correspond to the indices in data defined
     by `org_map`
     Results are written in `data`
    :param data: AnnFlux data frame
    :param indices: matrix with rows corresponding to predicted samples and columns to indices of neighbors in knn
    training set
    :param distances: matrix with rows corresponding to predicted samples and columns to distances to neighbors in knn
    training set
    :param train_labels: array with labels of knn training set
    :param data_indices: maps index of (indices, distances) to original index
    :param skip_first: skip first neighbor for computing predictions, typically used when making predictions on labelled data
    :param knn_rank_exponent:
    """
    update: dict[str, list[Tuple[int, Any]]] = {}  # key -> [(data_index, value), ...]
    for key in [
        "score_possible",
        "label_possible",
        "label_predicted",
        "score_predicted",
        "scores_predicted",
        "entropy",
        "score_true",
        "num_labeled_nn",
        "min_distance",
    ]:
        update[key] = [
            (-1, None),
        ] * len(indices)
    # data["scores_predicted"] = data["scores_predicted"].astype(str)
    # data["score_possible"] = data["score_possible"].astype(str)
    org_index_to_uid = dict(zip(data.index, data.uid))
    if "num_labeled_nn" not in data.columns:
        data["num_labeled_nn"] = None
    if "min_distance" not in data.columns:
        data["min_distance"] = None
    time_probabilities = 0
    time_rest = 0
    for i, indices_for_i in tqdm(enumerate(indices), desc="making knn predictions"):
        start_time = time.time()
        org_index = data_indices[i]

        case_debug = org_index_to_uid[org_index] == "RMNH_INS_1047595"


        probabilities = defaultdict(lambda: 0)
        # knn class histogram
        max_mass = 0
        multilabel_: List[str]
        num_labeled_nn = 0
        min_distance = None
        if case_debug:
            print("CASE_DEBUG", org_index, indices_for_i, train_labels[indices_for_i])
            distance_weights = []
        for i2, multilabel_ in enumerate(train_labels[indices_for_i]): # loop through NN
            if skip_first and i2 == 0:
                continue
            if multilabel_ is not None and len(multilabel_) > 0:
                distance_weight = max(1e-8, 1. / (distances[i][i2] ** knn_rank_exponent))  # noqa
                if case_debug:
                    distance_weights.append(distance_weight)
                # if distance_weight < 0.01 * max_mass:
                #     continue
                for label_ in multilabel_:
                    probabilities[label_] += distance_weight
                max_mass += distance_weight
                num_labeled_nn += 1
                if min_distance is None:
                    min_distance = distances[i][i2]

        for label_ in probabilities:
            probabilities[label_] /= max_mass
        if case_debug:
            print("dw", np.array(distance_weights) / max_mass)
        # if num_labeled_nn > 1 and len(probabilities) > 1:
        #     print("BLAAAAT", tag, probabilities, org_index_to_uid[org_index])
        time_probabilities += time.time() - start_time
        start_time = time.time()
        update["num_labeled_nn"][i] = (org_index, num_labeled_nn)
        update["min_distance"][i] = (org_index, min_distance)
        #
        if len(probabilities) > 0:
            max_labels = [
                label_ for label_, prob_ in probabilities.items() if prob_ > 0.5
            ]
            if len(max_labels) == 0:
                max_index = np.argmax(list(probabilities.values()))
                max_labels = [list(probabilities.keys())[max_index]]
            possible_labels = [
                label_
                for label_, prob_ in sorted(
                    probabilities.items(), key=lambda t_: -t_[1]
                )
                if float(os.getenv("MIN_PROB_POSSIBLE", 0.05)) < prob_ < 0.50
            ]
            update["score_possible"][i] = (
                org_index,
                ",".join(
                    [f"{probabilities[label_]:.2f}" for label_ in possible_labels]
                ),
            )

            update["label_possible"][i] = (org_index, ",".join(possible_labels))

            if len(max_labels) > 0:
                update["label_predicted"][i] = (org_index, ",".join(max_labels))
                update["score_predicted"][i] = (
                    org_index,
                    min([probabilities[label_] for label_ in max_labels]),
                )

                update["scores_predicted"][i] = (
                    org_index,
                    ",".join([f"{probabilities[label_]:.2f}" for label_ in max_labels]),
                )
            else:
                update["label_predicted"][i] = (org_index, None)
                update["score_predicted"][i] = (org_index, 0)
                update["scores_predicted"][i] = (org_index, None)
        time_rest += time.time() - start_time

        # if i % 100 == 0:
        #     print(time_rest, time_probabilities)
    #
    for key in update:
        update_for_key = update[key]
        if len(update_for_key) == 0 or key not in data.columns:
            continue
        update_for_key = sorted(update_for_key, key=lambda t_: t_[0])
        # noinspection PyArgumentList
        indices, values = zip(*update_for_key)
        if not data[key].values.flags["OWNDATA"]:  # need in test environment
            current_values = data[key].values.copy()
        else:
            current_values = data[key].values
        current_values[np.array(indices)] = values
        data[key] = current_values
