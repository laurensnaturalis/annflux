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
    print(f"embed_and_prepare: {resultset}")
    folder = resultset.path
    labels_path = os.path.join(working_folder, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            annotations = json.load(f)
    else:
        annotations = {}
    extra_predictions_path = f"{folder}/predictions.csv"
    out_path = os.path.join(working_folder, "annflux.csv")

    acc_test = embed_and_prepare_func(
        source,
        annotations,
        read_table_pandas(f"{folder}/results.csv"),
        f"{folder}/last_full.npz",
        os.path.join(working_folder, "split.json"),
        extra_predictions_path,
        compute_performance,
        show=show,
    )
    # write_performance(
    #     acc_test,
    #     len(annotations) - len(labeled_test_uids),
    #     len(true_test),
    #     working_folder,
    # ) # TODO


def embed_and_prepare_func(
    source_or_data_state_path: AnnfluxSource | str,
    annotations: dict[str, str],
    data: pandas.DataFrame,
    features_path,
    split_path,
    extra_predictions_path=None,
    compute_performance: bool = False,
    show=False,
):
    source: AnnfluxSource | None = None
    if isinstance(source_or_data_state_path, AnnfluxSource):
        source = source_or_data_state_path
        data_state_path = source.data_state_path
    else:
        data_state_path = source_or_data_state_path
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
    features = numpy_load(features_path, "lastFull")
    embedding = compute_tsne(features)
    embedding = normalize_and_scale(embedding)

    sel = np.arange(len(embedding))
    data = data.iloc[sel]
    data["e_0"] = embedding[sel, 0]
    data["e_1"] = embedding[sel, 1]
    data["in_test"] = data["uid"].apply(lambda x_: int(x_ in test_uids))
    data.to_csv(data_state_path, index=False)
    if source is not None:
        fast_density_peak_clustering(
            source
        )  # TODO: return data and don't save in function
        peak_merge(source)  # TODO: return data and don't save in function
    data = pandas.read_csv(data_state_path)
    color_and_label(data, annotations, json.load(open(source.label_definitions_path))["labels"])
    data.to_csv(data_state_path, index=False)
    if show:
        import matplotlib.pyplot as plt

        plt.scatter(embedding[sel, 0], embedding[sel, 1], c=data.score_predicted)
        plt.show()
    return acc_test


def group_embedding(project_folder):
    source_ = AnnfluxSource(project_folder)
    data_ = pandas.read_csv(source_.group_flux_data_path())
    if "group_id" in data_:  # TODO: let group classifier output the right format
        data_["uid"] = data_.group_id
        data_.score_predicted = data_.score_predicted.apply(lambda x_: x_ / 100.0)
        del data_["group_id"]
    embed_and_prepare_func(
        source_.group_flux_data_path(),
        json.load(open(source_.labels_path)),
        data_,
        source_.group_features_path(),
        None,
        None,
        False,
    )


if __name__ == "__main__":
    group_embedding(sys.argv[1])
