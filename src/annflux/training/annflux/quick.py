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
import contextlib
import json
import logging
import os
import time
from collections import defaultdict
from typing import Set, Any, Tuple, List

import faiss
import numpy as np
from scipy.sparse import coo_matrix as _sparse_coo
import pandas
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.algorithms.calibration_uncertainty import compute_and_add_calibrated_uncertainty
from annflux.algorithms.cood import compute_cood
from annflux.algorithms.feature_reconstruction_error import compute_fre, compute_nn_underrepresented
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
    state: AnnFluxState, logger: logging.Logger, knn_type="quick", group=False,
    csv_write_lock=None,
):
    if not group:
        quick_reclassification_instance(knn_type, state, logger, csv_write_lock=csv_write_lock)
    else:
        quick_reclassification_group(knn_type, state, logger)


def quick_reclassification_instance(knn_type, state, logger: logging.Logger, csv_write_lock=None):
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
            os.getenv("QUICKER_UPDATES", 2)
        )  # 0 = off, 1 = display only, 2 = display + knn
        quicker_updates = (
            quicker_updates
            if (state.new_labeled_uids is not None and len(state.new_labeled_uids) > 0)
            else 0
        )
        logger.info(f"[QU] quicker_updates={quicker_updates}, new_labeled_uids={state.new_labeled_uids}")

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
                data["score_predicted"] = None
                data["scores_predicted"] = None
                data["label_possible"] = ""
                data["score_possible"] = None
            else:
                data.loc[list(new_labeled_nn_idx), "label_predicted"] = None
                data.loc[list(new_labeled_nn_idx), "score_predicted"] = None
                data.loc[list(new_labeled_nn_idx), "scores_predicted"] = None
                data.loc[list(new_labeled_nn_idx), "label_possible"] = ""
                data.loc[list(new_labeled_nn_idx), "score_possible"] = None
        else:
            data["label_predicted"] = None
            data["score_predicted"] = None
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
            labeled_arr = np.asarray(state.labeled_indices)
            idx_sel = np.where(np.isin(labeled_arr, list(new_labeled_nn_idx)))[0]
            indices_ = indices_[idx_sel]
            distances_ = distances_[idx_sel]
            labeled_indices_ = labeled_arr[idx_sel].tolist()
            new_labeled_indices_for_fre = labeled_indices_
            logger.info(f"[QU] path=2: labeled subset size={len(labeled_indices_)} of {len(state.labeled_indices)}")
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
            near_arr = np.asarray(near_labeled_indices)
            idx_sel = np.where(np.isin(near_arr, list(new_labeled_nn_idx)))[0]
            indices_ = indices_[idx_sel]
            distances_ = distances_[idx_sel]
            near_labeled_indices_ = near_arr[idx_sel].tolist()
            near_labeled_indices_.extend(test_indices)
            logger.info(f"[QU] path=2: near_labeled subset size={len(near_labeled_indices_)} of {len(near_labeled_indices)}")
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
        data["label_predicted"] = data["label_predicted"].map(canon_)
        if "label_true" in data.columns:
            data["label_true"] = data["label_true"].map(canon_)
        # FRE
        state.g_quick_status = "computing FRE"
        # Reset PCA cache if features were reloaded
        if reload_features:
            fre_pca_cache.clear()
        t_fre_start = time.time()
        compute_fre(
            annotations,
            data,
            state.features,
            state.labeled_indices,
            test_uids,
            new_labeled_indices_for_fre,
            pca_cache=fre_pca_cache,
        )
        logger.info(f"[TIMING] compute_fre total={time.time() - t_fre_start:.3f} s")
        t_nn_start = time.time()
        compute_nn_underrepresented(
            data,
            state.features,
            state.labeled_indices,
            state.all_indices,
        )
        logger.info(f"[TIMING] compute_nn_underrepresented total={time.time() - t_nn_start:.3f} s")
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
            # Use temporary column for stats calculation to avoid overwriting labeled
            data["_tmp_labeled"] = data["uid"].isin(labeled_uids).astype(int)
            data_ = data[data["dp_most_needed"] < data["dp_most_needed"].max()]
            # print("bloep", data_[data["_tmp_labeled"] == 0])
            logger.info(f"|data most needed| = {len(data_)}")
            near_labeled_perc = len(data_[data["_tmp_labeled"] == 1]) / len(data_)
            print(f"has_density_peak: {near_labeled_perc=}")

            write_performance_key_val(
                state.performance_path, "percentage_near_labeled", near_labeled_perc
            )
            # Clean up temporary column
            data.drop(columns=["_tmp_labeled"], inplace=True)
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
        
        # Calibration-based uncertainty score (computed AFTER predictions are filled)
        state.g_quick_status = "computing calibrated uncertainty"
        t_cal_start = time.time()
        compute_and_add_calibrated_uncertainty(
            data,
            labeled_indices=np.array(state.labeled_indices),
        )
        logger.info(f"[TIMING] compute_calibrated_uncertainty total={time.time() - t_cal_start:.3f} s")
        
        # COOD: Combined Out-of-Distribution Detection (disabled by default)
        if os.getenv("ENABLE_COOD", "0") == "1":
            state.g_quick_status = "computing COOD"
            t_cood_start = time.time()
            try:
                # Use pre-computed kNN from state if available
                nn_indices = getattr(state, 'all_indices', None)
                nn_distances = getattr(state, 'all_distances', None)
                if nn_indices is not None and nn_distances is not None:
                    logger.info(f"[COOD] Using pre-computed kNN: {nn_indices.shape}")
                cood_results = compute_cood(
                    data,
                    features=state.features,
                    nn_indices=nn_indices,
                    nn_distances=nn_distances,
                    train_on_labeled=len(state.labeled_indices) > 10,
                )
                # Add COOD columns to data
                for col in cood_results.columns:
                    data[col] = cood_results[col]
                logger.info(f"[TIMING] compute_cood total={time.time() - t_cood_start:.3f} s")
            except Exception as e:
                logger.warning(f"[COOD] Failed to compute: {e}")
        else:
            # Drop stale COOD columns if disabled
            cood_cols = [col for col in data.columns if col.startswith('cood_')]
            if cood_cols:
                logger.info(f"[COOD] Dropping {len(cood_cols)} stale columns: {cood_cols}")
                data.drop(columns=cood_cols, inplace=True)
        
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

    # Load certainty map for partial labeling
    certainty_map = {}
    if getattr(state, "certainty_path", None) and os.path.exists(state.certainty_path):
        with open(state.certainty_path) as f:
            certainty_map = json.load(f)

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
        certainty_map=certainty_map,
        project_folder=state.project_folder,
    )

    with open(state.doublecheck_path) as f:
        double_checked = set(json.load(f)["checked"])
        data["double_checked"] = data["uid"].isin(double_checked).astype(int)
    
    # Add certainty column if available
    if getattr(state, "certainty_path", None) and os.path.exists(state.certainty_path):
        with open(state.certainty_path) as f:
            certainty_map = json.load(f)
        data["certainty"] = data["uid"].map(lambda u: certainty_map.get(u, "certain"))
    
    with (csv_write_lock if csv_write_lock is not None else contextlib.nullcontext()):
        data.to_csv(state.annflux_path, index=False)
        if hasattr(csv_write_lock, '_bump_version'):
            csv_write_lock._bump_version()
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
            "HNSW32",
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

        print("Train/Add done")

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
    
    # Load certainty map for partial labeling
    certainty_map = {}
    if getattr(state, "certainty_path", None) and os.path.exists(state.certainty_path):
        with open(state.certainty_path) as f:
            certainty_map = json.load(f)
    
    class_to_color = color_and_label(
        data,
        annotations,
        json.load(open(os.path.join(state.annflux_folder, "label_defs.json")))[
            "labels"
        ],
        certainty_map=certainty_map,
        project_folder=state.project_folder,
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
    make_predictions_alt(data, indices, distances, train_labels, data_indices, skip_first, knn_rank_exponent, tag)

def make_predictions_alt(
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
    t_total = time.time()
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
                if distance_weight < 0.01 * max_mass:
                    break
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
                    sum([probabilities[label_] for label_ in max_labels]) / len(max_labels),
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
    logging.getLogger("annflux_training").info(
        f"[TIMING] make_predictions_alt N={len(indices)} "
        f"probabilities={time_probabilities:.3f}s rest={time_rest:.3f}s "
        f"total={time.time() - t_total:.3f}s"
    )
    for key in update:
        update_for_key = update[key]
        if len(update_for_key) == 0 or key not in data.columns:
            continue
        update_for_key = sorted(update_for_key, key=lambda t_: t_[0])
        # noinspection PyArgumentList
        indices_upd, values = zip(*update_for_key)
        if not data[key].values.flags["OWNDATA"]:
            current_values = data[key].values.copy()
        else:
            current_values = data[key].values
        current_values[np.array(indices_upd)] = values
        data[key] = current_values


def make_predictions_fast(
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
    if "num_labeled_nn" not in data.columns:
        data["num_labeled_nn"] = None
    if "min_distance" not in data.columns:
        data["min_distance"] = None
    t_total = time.time()

    N, k = indices.shape
    data_indices_arr = np.asarray(data_indices)
    min_prob_possible = float(os.getenv("MIN_PROB_POSSIBLE", 0.05))

    # --- build label vocabulary from all neighbor labels ---
    # Order must match alt: labels encountered first in row-major neighbor traversal,
    # skipping column 0 when skip_first=True (same as the alt loop's continue on i2==0)
    start_col = 1 if skip_first else 0
    all_labels: List[str] = []
    for row_idx in range(N):
        for col_idx in range(start_col, k):
            ml = train_labels[indices[row_idx, col_idx]]
            if ml is not None:
                for lbl in ml:
                    all_labels.append(lbl)
    # ensure all labels that appear in train_labels are in the vocab (needed for label_id_matrix)
    for ml in train_labels:
        if ml is not None:
            for lbl in ml:
                all_labels.append(lbl)
    unique_labels = list(dict.fromkeys(all_labels))  # order-preserving dedup
    label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    L = len(unique_labels)

    # --- expand train_labels into a (M, max_multi) integer matrix ---
    # For each neighbor position, store its label ids (-1 = no label / unlabeled)
    max_multi = max((len(ml) for ml in train_labels if ml is not None), default=1)
    label_id_matrix = np.full((len(train_labels), max_multi), -1, dtype=np.int32)
    for mi, ml in enumerate(train_labels):
        if ml is not None and len(ml) > 0:
            for li, lbl in enumerate(ml):
                label_id_matrix[mi, li] = label_to_id[lbl]

    # --- compute distance weights (N, k) ---
    weights = np.maximum(1e-8, 1.0 / (distances ** knn_rank_exponent))  # (N, k)
    if skip_first:
        weights[:, 0] = 0.0

    # mask out unlabeled neighbors (train_labels[neighbor_idx] is None or empty)
    labeled_mask = np.array(
        [ml is not None and len(ml) > 0 for ml in train_labels], dtype=bool
    )
    neighbor_labeled = labeled_mask[indices]  # (N, k)
    weights = weights * neighbor_labeled  # zero weight for unlabeled neighbors

    # num_labeled_nn per row — count only neighbors with non-zero weight (accounts for skip_first)
    num_labeled_nn_arr = (weights > 0).sum(axis=1)  # (N,)

    # min_distance: first non-zero weight column per row
    # (columns already zeroed for unlabeled/skip_first)
    nonzero_cols = weights > 0  # (N, k)
    has_any = nonzero_cols.any(axis=1)
    first_nonzero = np.where(nonzero_cols, np.arange(k), k).argmin(axis=1)
    min_distance_arr = np.where(has_any, distances[np.arange(N), first_nonzero], None)

    # --- accumulate scores into (N, L) matrix via sparse COO ---
    # Build flat arrays of (sample_row, label_col, weight) for all valid entries,
    # then let scipy.sparse sum duplicates in C when converting to CSR.
    neighbor_label_ids = label_id_matrix[indices]  # (N, k, max_multi)
    # repeat weights for each label slot
    weights_rep = np.repeat(weights[:, :, np.newaxis], max_multi, axis=2)  # (N, k, max_multi)
    valid_mask = neighbor_label_ids >= 0  # (N, k, max_multi)
    row_ids = np.broadcast_to(np.arange(N)[:, np.newaxis, np.newaxis], (N, k, max_multi))
    coo_rows = row_ids[valid_mask]
    coo_cols = neighbor_label_ids[valid_mask]
    coo_data = weights_rep[valid_mask]
    scores = np.asarray(
        _sparse_coo((coo_data, (coo_rows, coo_cols)), shape=(N, L)).tocsr().toarray()
    )

    # normalise each row by total mass (sum of weights per row, not per label slot)
    row_mass = weights.sum(axis=1, keepdims=True)  # (N, 1) — one weight per neighbor
    row_mass = np.where(row_mass == 0, 1.0, row_mass)
    scores /= row_mass  # (N, L) — probabilities per label per sample

    # --- build output columns from score matrix ---
    unique_labels_arr = np.array(unique_labels)
    has_prediction = num_labeled_nn_arr > 0  # (N,) bool

    # predicted label(s): rows where any label > 0.5 are multi-label; rest use argmax
    above_half = scores > 0.5  # (N, L)
    multi_label_rows = above_half.sum(axis=1) > 1  # (N,) rows with 2+ predicted labels
    # single-label: has neighbors, not multi-label — always predict via argmax (matching alt behaviour)
    single_label_rows = has_prediction & ~multi_label_rows

    # argmax scores for single-label fast path
    argmax_ids = scores.argmax(axis=1)  # (N,)

    # score_predicted for single-label rows
    score_predicted_arr = scores[np.arange(N), argmax_ids]  # (N,)

    # build string outputs via vectorised ops where possible
    # label_predicted for single-label rows
    label_predicted_single = np.where(single_label_rows, unique_labels_arr[argmax_ids], None)
    scores_predicted_single = np.where(
        single_label_rows,
        np.array([f"{v:.2f}" for v in score_predicted_arr]),
        None,
    )

    out_label_predicted: List = list(label_predicted_single)
    out_score_predicted: List = [float(score_predicted_arr[i]) if has_prediction[i] else 0 for i in range(N)]
    out_scores_predicted: List = list(scores_predicted_single)
    out_label_possible: List = [None] * N
    out_score_possible: List = [None] * N

    # possible labels: vectorise mask, do string join per row only for rows that have any
    poss_mask_matrix = has_prediction[:, np.newaxis] & (scores > min_prob_possible) & (scores < 0.50)  # (N, L)
    rows_with_possible = np.where(has_prediction & poss_mask_matrix.any(axis=1))[0]

    # multi-label predicted rows — small minority, handle per-row
    multi_rows = np.where(has_prediction & multi_label_rows)[0]

    for i in multi_rows:
        max_ids = np.where(above_half[i])[0]
        max_labels = [unique_labels[j] for j in max_ids]
        out_label_predicted[i] = ",".join(max_labels)
        out_score_predicted[i] = float(scores[i, max_ids].mean())
        out_scores_predicted[i] = ",".join(f"{scores[i, j]:.2f}" for j in max_ids)

    for i in rows_with_possible:
        poss_ids = np.where(poss_mask_matrix[i])[0]
        poss_ids = poss_ids[np.argsort(-scores[i, poss_ids])]
        out_label_possible[i] = ",".join(unique_labels[j] for j in poss_ids)
        out_score_possible[i] = ",".join(f"{scores[i, j]:.2f}" for j in poss_ids)

    # --- write results back to data ---
    def _write_col(col_name, values_arr):
        if col_name not in data.columns:
            return
        current = data[col_name].values
        if not current.flags["OWNDATA"]:
            current = current.copy()
        current[data_indices_arr] = values_arr
        data[col_name] = current

    _write_col("label_predicted", out_label_predicted)
    _write_col("score_predicted", out_score_predicted)
    _write_col("scores_predicted", out_scores_predicted)
    _write_col("label_possible", out_label_possible)
    _write_col("score_possible", out_score_possible)
    _write_col("num_labeled_nn", num_labeled_nn_arr)
    _write_col("min_distance", min_distance_arr)
    logging.getLogger("annflux_training").info(
        f"[TIMING] make_predictions N={N} total={time.time() - t_total:.3f}s"
    )
