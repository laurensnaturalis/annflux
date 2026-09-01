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

"""
Server background job functions for AnnFlux
"""

import json
import os
import pandas
import threading
import time
import logging
from datetime import datetime

import numpy as np

from annflux.algorithms.embeddings import compute_tsne
from annflux.algorithms.fastdpeak import fast_density_peak_clustering
from annflux.algorithms.fastdpeak_merge import peak_merge
from annflux.repo_results_to_embedding import group_embedding
from annflux.repository.dataset import Dataset
from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.shared import AnnfluxSource
from annflux.tools.core import AnnFluxState
from annflux.tools.data import create_group_flux_data
from annflux.tools.progress_learn import estimate_duration
from annflux.training.annflux.feature_extractor import make_resultset
from annflux.training.annflux.quick import quick_reclassification, group_classification, load_data
from annflux.training.pytorch.torch_backend import linear_retraining


class StatusUpdate:
    """Callback for updating training status during linear retraining."""
    
    def __init__(self, state: AnnFluxState):
        self.state = state

    def __call__(self, epoch, val_loss=None):
        self.state.linear_status_epoch = epoch


def retrain_job(state: AnnFluxState, logger):
    """Background job for retraining the model."""
    state.g_quick_status = "training"
    load_data(state, logger, no_linear_features=True)
    weights_path = linear_retraining(state, StatusUpdate(state))
    
    repo: Repository = AnnfluxSource(state.project_folder).repository
    # TODO: store linear model
    make_resultset(
        repo.get(label=Dataset).last(),
        state.features,
        repo,
        message=f"linear features from label state={len(state.labeled_indices)}",
    )
    logger.info(
        f"Stored Resultset for linear trained features in {repo.get(label=Resultset).first()}"
    )
    
    state.g_quick_status = "computing embedding"
    embedding = compute_tsne(state.features)
    state.g_quick_status = "computing embedding done"
    data = pandas.read_csv(
        state.annflux_path,
        dtype={"label_predicted": str, "score_true": float},
    )
    data["e_0"] = embedding[:, 0]
    data["e_1"] = embedding[:, 1]
    data.to_csv(state.annflux_path, index=False)
    state.trained_for_version_previous = len(state.labeled_indices)
    
    state.g_quick_status = "computing density peak"
    fast_density_peak_clustering(state.project_folder)
    peak_merge(state.project_folder)
    state.g_quick_status = "quicker classification"
    
    quick_reclassification(state, logger)

    logger.info(f"retrain_job: done - {state.trained_for_version}")
    state.trained_for_version = len(state.labeled_indices)  # TODO: replace by hash?


def group_train_job(state: AnnFluxState, logger):
    """Background job for group training."""
    state.g_quick_status = "group training"
    source = AnnfluxSource(state.project_folder)

    if state.features is None or state.labeled_indices is None:
        load_data(state, logger)

    create_group_flux_data(source)

    split_path = os.path.join(state.annflux_folder, "split_group.json")
    group_data = pandas.read_csv(source.group_flux_data_path())

    if not os.path.exists(split_path):
        test_uids = np.random.choice(
            group_data.uid.values, int(0.10 * len(group_data)), replace=False
        ).tolist()
        with open(split_path, "w") as f:
            json.dump({"test": test_uids}, f)
    else:
        test_uids = json.load(open(split_path))["test"]

    record_features, accuracy_group, record_table = group_classification(
        g_state.features,
        pandas.read_csv(g_state.annflux_path),
        os.path.join(state.annflux_folder, "group_feature_images"),
        group_data,
        test_uids,
    )
    print(record_features.shape, accuracy_group, len(record_table))
    record_table.to_csv(
        os.path.join(g_state.annflux_folder, "group0_annflux.csv"), index=False
    )
    
    np.savez(
        os.path.join(g_state.annflux_folder, "group0_features.npz"),
        lastFull=record_features,
    )
    state.g_quick_status = "group embedding"
    group_embedding(g_state.project_folder)
    logger.info(f"group_train_job: done - {state.trained_for_version}")
    state.trained_for_version = len(state.labeled_indices)  # TODO(crit): for group


def do_quick_reclassification(g_state: AnnFluxState, is_group: bool):
    """Trigger quick reclassification in a background thread."""
    logger = logging.getLogger("annflux_training")
    
    if g_state.train_thread is None or not g_state.train_thread.is_alive():
        g_state.train_thread = threading.Thread(
            target=quick_reclassification, args=(g_state, logger, "quick", is_group)
        )
        g_state.train_thread.start()
        g_state.train_thread.join()
