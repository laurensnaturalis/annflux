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
import time
from pathlib import Path

from numpy._typing import NDArray
import numpy as np


class AnnFluxState(object):
    """
    Stores the core state of the AnnFlux state
    """

    def __init__(self, working_folder):
        super().__init__()
        self.time_new_status_time = None
        self.working_folder_ = Path(working_folder)

    @property
    def project_folder(self) -> str:
        return str((self.working_folder_ / "..").resolve())

    @property
    def annflux_folder(self) -> str:
        return str(self.working_folder_)

    @property
    def timings_path(self) -> str:
        return str(self.working_folder_ / "timings.csv")

    @property
    def annflux_path(self) -> str:
        return str(self.working_folder_ / "annflux.csv")

    @property
    def g_quick_status(self):
        return self.g_quick_status_

    @g_quick_status.setter
    def g_quick_status(self, val):
        if val != self.g_quick_status_:
            self.time_new_status_time = time.time()
        self.g_quick_status_ = val
        timings_path = self.working_folder_ / "timings.csv"
        if not timings_path.exists():
            self.write_timing(
                timings_path, "status", "timestamp", "num_total", "num_labeled", "w"
            )
        self.write_timing(
            timings_path,
            self.g_quick_status_,
            time.time(),
            len(self.features) if self.is_initialized() else None,
            len(self.labeled_indices) if self.labeled_indices is not None else None,
        )

    @staticmethod
    def write_timing(timings_path, key, val, num_total, num_labeled, mode="a"):
        with open(timings_path, mode) as f:
            f.write(f"{key},{val},{num_total or ''},{num_labeled or ''}\n")

    @property
    def labeled_indices(self) -> NDArray[np.uint64]:
        return (
            self.labeled_indices_
            if hasattr(self, "labeled_indices_")
            else np.zeros((0,), dtype=np.uint64)
        )

    @labeled_indices.setter
    def labeled_indices(self, val: NDArray[np.uint64]):
        self.labeled_indices_ = val

    cache_for: str | None = None
    features_: NDArray

    @property
    def features(self) -> NDArray:
        return self.features_ if hasattr(self, "features_") else np.zeros((0,), dtype=np.uint64)

    @features.setter
    def features(self, val):
        self.features_ = val

    knn_index = None
    all_distances = None
    all_indices: NDArray = None # ty:ignore
    opt_knn_rank_exponent = None
    knn_rank_exponent = 6
    label_array: NDArray
    label_array_test: NDArray
    labeled_indices_: NDArray

    version_for_recompute_: str

    @property
    def version_for_recompute(self):
        return self.version_for_recompute_ if hasattr(self, "version_for_recompute_") else "N/A"

    @version_for_recompute.setter
    def version_for_recompute(self, val):
        self.version_for_recompute_ = val

    g_quick_status_ = None
    new_labeled_uids = None
    train_thread = None
    trained_for_version_previous = None
    linear_status_epoch: int = -1
    trained_for_version: int = -1
    optimize_weight_exponent: bool = False
    labels_path: str
    doublecheck_path: str
    performance_path: str
    labeled_test_indices: NDArray[np.uint64]

    def is_initialized(self):
        return hasattr(self, "features_")
