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
import os
import sys

import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from annflux.algorithms.embeddings import compute_tsne, normalize_and_scale
from annflux.algorithms.fastdpeak import fast_density_peak_clustering
from annflux.algorithms.fastdpeak_merge import peak_merge
from annflux.performance.basic import write_performance
from annflux.shared import AnnfluxSource
from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.tools.data import color_and_label
from annflux.tools.io import read_table_pandas, numpy_load
from annflux.tools.mixed import get_basic_logger

logger = get_basic_logger("repo_results_to_embedding")


def embed_and_prepare(source: AnnfluxSource, show=False, compute_performance=False):
    working_folder = source.working_folder
    repo = Repository(os.path.join(working_folder, "datarepo"))
    resultset = repo.get(label=Resultset, tag="unseen").last()
    print(f"embed_and_prepare: {resultset.entry.uid}")
    folder = resultset.path
    labels_path = os.path.join(working_folder, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            annotations = json.load(f)
    else:
        annotations = {}
    data: pandas.DataFrame = read_table_pandas(f"{folder}/results.csv")
    extra_predictions_path = f"{folder}/predictions.csv"
    out_path = os.path.join(working_folder, "annflux.csv")

    acc_test = embed_and_prepare_func(
        annotations,
        compute_performance,
        data,
        extra_predictions_path,
        show,
        f"{folder}/last_full.npz",
        out_path,
        os.path.join(working_folder, "split.json"),
    )
    # write_performance(
    #     acc_test,
    #     len(annotations) - len(labeled_test_uids),
    #     len(true_test),
    #     working_folder,
    # ) # TODO


def embed_and_prepare_func(
    annotations,
    compute_performance,
    data,
    extra_predictions_path,
    show,
    features_path,
    out_path,
    split_path,
):
    data.label_predicted = data.label_predicted.astype(str)
    data.uid = data.uid.astype(str)
    if extra_predictions_path is not None and os.path.exists(extra_predictions_path):
        extra_predictions_ = pandas.read_csv(extra_predictions_path)
        data = pandas.merge(data, extra_predictions_, on="uid")
        data["label_predicted"] = data["prediction"]
    #
    if split_path is not None:
        with open(split_path) as f:
            test_uids = set(json.load(f)["test"])
    else:
        test_uids = set()
    labeled_test_uids = test_uids.intersection(set(annotations.keys()))
    labeled_test_data = data[data.uid.isin(labeled_test_uids)]
    true_test = [annotations[uid_].split(",") for uid_ in labeled_test_data.uid]
    binarizer = MultiLabelBinarizer()
    binarizer.fit(true_test)
    acc_test = None
    if compute_performance:
        acc_test = accuracy_score(
            binarizer.transform(true_test),
            binarizer.transform(
                [
                    x_.split(",") if not pandas.isna(x_) else []
                    for x_ in labeled_test_data.label_predicted
                ]
            ),
        )
    if "last_full" in features_path:
        npz_path = features_path.replace("last_full", "custom")
        npy_path = features_path.replace("last_full.npz", "custom.npy")
        if os.path.exists(npz_path):
            features = numpy_load(npz_path, "arr_0")
        elif os.path.exists(npy_path):
            features = np.load(npy_path)
        else:
            features = numpy_load(features_path, "lastFull")
    else:
        features = numpy_load(features_path, "lastFull")
    print("embedding", len(features))
    embedding = compute_tsne(features)
    embedding = normalize_and_scale(embedding)

    sel = np.arange(len(embedding))
    data = data.iloc[sel]
    data["e_0"] = embedding[sel, 0]
    data["e_1"] = embedding[sel, 1]
    data["in_test"] = data["uid"].apply(lambda x_: int(x_ in test_uids))
    data.to_csv(source.data_state_path, index=False)
    fast_density_peak_clustering(source) # TODO: return data and don't save in function
    peak_merge(source) # TODO: return data and don't save in function
    data = pandas.read_csv(source.data_state_path)
    color_and_label(data, annotations)
    data.to_csv(source.data_state_path, index=False)
    if show:
        import matplotlib.pyplot as plt
        plt.scatter(embedding[sel, 0], embedding[sel, 1], c=data.score_predicted)
        plt.show()
    return acc_test


if __name__ == "__main__":
    annflux_path = (
        "/home/laurens/Documents/data/ami_oh2_hour/annflux/group0_annflux.csv"
    )
    data = pandas.read_csv(annflux_path)
    data["uid"] = data.group_id
    data.score_predicted = data.score_predicted.apply(lambda x_: x_ / 100.0)
    embed_and_prepare_func(
        {},
        False,
        data,
        None,
        True,
        "/home/laurens/Documents/data/ami_oh2_hour/annflux/group0_features.npz",
        annflux_path,
        None,
    )
