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
import itertools
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


MultilabelPrediction = List[Tuple[str] | List[str]]

logger = logging.getLogger("annflux_server")


def compute_performance(
    predicted_test: MultilabelPrediction,
    true_test: MultilabelPrediction,
    annotations: Dict[str, str],
    data: pandas.DataFrame,
    num_train_val: int,
    certain_threshold=0.90,
    performance_graph_path=None,
    detailed_performance_path=None,
):
    time_start = time.time()
    if len(true_test) > 0 and len(predicted_test) > 0:
        binarizer = MultiLabelBinarizer()
        binarizer.fit(true_test)
        acc_test = accuracy_score(
            binarizer.transform(true_test), binarizer.transform(predicted_test)
        )
        if performance_graph_path is not None:
            write_performance(
                performance_graph_path, acc_test, num_train_val, len(true_test)
            )
    labels = list(set(itertools.chain(*[x_.split(",") for x_ in annotations.values()])))
    label_to_index = dict(zip(labels, range(len(labels))))
    detailed_performance_table = []

    true_matrix = np.zeros((len(true_test), len(labels)))
    predicted_matrix = np.zeros((len(true_test), len(labels)))
    for i_, (true_, predicted_) in enumerate(zip(true_test, predicted_test)):
        for label_true in true_:
            if label_true in label_to_index:
                true_matrix[i_, label_to_index[label_true]] = 1
            else:
                # TOdO
                pass
        for label_predicted in predicted_:
            predicted_matrix[i_, label_to_index[label_predicted]] = 1

    for label_ in labels:
        logger.debug(f"{label_=}")
        per_image_true = true_matrix[:, label_to_index[label_]]
        per_image_predicted = predicted_matrix[:, label_to_index[label_]]

        precision, recall, f_score, support = precision_recall_fscore_support(
            per_image_true, per_image_predicted, zero_division=0.0
        )
        # if sum(per_image_true) > 0:
        #     print(label_, per_image_true, per_image_predicted, precision, recall, f_score, support)
        if len(precision) > 0 and sum(per_image_true) > 0:
            index_ = 1 if len(precision) > 1 else 0
            logger.debug(
                f"{label_} precision={precision[index_]:.2f} recall={recall[index_]:.2f} {support[index_]}"
            )
            detailed_performance_table.append(
                (label_, precision[index_], recall[index_], support[index_])
            )
    out_table = pandas.DataFrame(
        data=detailed_performance_table,
        columns=("label", "precision", "recall", "support"),  # ty: ignore
    )
    print(f"1/2  took {time.time() - time_start} s")
    time_start = time.time()

    # compute how many are certain according to a threshold
    num_certain = defaultdict(lambda: 0)
    num_uncertain = defaultdict(lambda: 0)
    num_labeled = defaultdict(lambda: 0)
    num_unlabeled_for_label = defaultdict(lambda: 0)
    data.scores_predicted = data.scores_predicted.astype(str)
    data_predicted = data[~pandas.isna(data.label_predicted) & (~pandas.isna(data.scores_predicted))]
    #
    true_label_map = {}
    for true_label in data["label_true"].unique():
        true_label_map[true_label] = true_label.split(",") if true_label is not None else []
    predicted_label_map = {}
    for label_predicted in data["label_predicted"].unique():
        predicted_label_map[label_predicted] = label_predicted.split(",") if label_predicted is not None else []
    #
    print(f"{len(data_predicted)=}")
    list_label_true = data_predicted["label_true"]
    list_label_predicted = data_predicted["label_predicted"]
    list_num_labeled_nn = data_predicted["num_labeled_nn"]
    list_scores_predicted = data_predicted["scores_predicted"]
    # for _, row in data_predicted.iterrows():
    for label_true, label_predicted, num_labeled_nn, scores_predicted in zip(list_label_true, list_label_predicted, list_num_labeled_nn, list_scores_predicted):
        true_labels = true_label_map.get(label_true, [])

        predicted_labels = predicted_label_map.get(label_predicted, [])
        predicted_probs = map(float, scores_predicted.split(","))
        for label_, prob_ in zip(predicted_labels, predicted_probs):
            if (
                prob_ > certain_threshold
                and num_labeled_nn is not None
                and num_labeled_nn > 1
            ):
                num_certain[label_] += 1
            else:
                num_uncertain[label_] += 1
            num_unlabeled_for_label[label_] += len(true_labels) == 0

        #
    print(f"5/8  took {time.time() - time_start} s, {len(data)=}")
    time_start = time.time()
    data_true = data[~pandas.isna(data.label_true)]
    print(f"{len(data_true)=}")
    for _, row in data_true.iterrows():
        true_labels = (
            row.label_true.split(",")
            if not pandas.isna(row.label_true) and row.label_true is not None
            else []
        )

        for label_ in true_labels:
            num_labeled[label_] += 1
    print(f"3/4  took {time.time() - time_start} s, {len(data)=}")
    time_start = time.time()
    out_table["num_predicted_certain"] = [
        num_certain.get(x_, 0) for x_ in out_table.label
    ]
    out_table["num_predicted_uncertain"] = [
        num_uncertain.get(x_, 0) for x_ in out_table.label
    ]
    out_table["num_labeled"] = [num_labeled.get(x_, 0) for x_ in out_table.label]
    out_table["num_unlabeled"] = [
        num_unlabeled_for_label.get(x_, 0) for x_ in out_table.label
    ]
    if detailed_performance_path is not None:
        out_table.to_csv(detailed_performance_path, index=False)
    print(f"compute_performance took {time.time() - time_start} s")


def write_performance(performance_graph_path, acc_test, num_train_val, num_test):
    if os.path.exists(performance_graph_path):
        j_performance = json.load(open(performance_graph_path))
    else:
        j_performance = {"test_performance": []}
    j_performance["test_performance"].append([num_train_val, num_test, acc_test])
    with open(performance_graph_path, "w") as f:
        json.dump(j_performance, f, indent=2)


def write_performance_key_val(performance_graph_path, key: str, val: Any):
    if os.path.exists(performance_graph_path):
        j_performance = json.load(open(performance_graph_path))
    else:
        j_performance = {"test_performance": []}
    j_performance[key] = val
    with open(performance_graph_path, "w") as f:
        json.dump(j_performance, f, indent=2)


def get_performance_key_val(performance_graph_path, key: str, default_val=None):
    if os.path.exists(performance_graph_path):
        j_performance = json.load(open(performance_graph_path))

        return j_performance.get(key, default_val)
    return default_val
