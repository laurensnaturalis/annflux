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
import os
from pathlib import Path



base_data_path = os.path.expanduser(os.path.join("~", "annflux", "data"))


class AnnfluxSource(object):
    start_labels = []
    exclusivity = []
    id_column = "image_id"
    data_path_ = None
    images_path_ = None
    annflux_folder_ = None
    label_column_for_unseen = "label"

    def __init__(self, folder: Path | str | None = None):
        if folder is not None:
            folder = Path(folder)
            self.data_path_ = str(folder / "images.csv")
            self.images_path_ = str(folder / "images")
            self.annflux_folder_ = str(folder / "annflux")
            self.folder = folder

    @property
    def data_path(self):
        """
        Path containing input data (images.csv)
        """
        return (
            os.path.join(base_data_path, self.data_path_) # ty: ignore
            if base_data_path is not None
            else self.data_path_
        )

    @property
    def flux_data_path(self):
        return os.path.join(self.annflux_folder_, "annflux.csv") # ty: ignore

    def group_flux_data_path(self, group=0):
        return os.path.join(self.annflux_folder_, f"group{group}_annflux.csv")  # ty: ignore

    def group_features_path(self, group=0):
        return os.path.join(self.annflux_folder_, f"group{group}_features.npz") # ty: ignore


    @property
    def stream_data_path(self):
        return os.path.join(self.annflux_folder_, "stream_process.csv") # ty: ignore

    @property
    def images_folder(self):
        """
        Folder where images are located
        """
        return (
            os.path.join(base_data_path, self.images_path_) # ty: ignore
            if base_data_path is not None
            else self.images_path_
        )

    @property
    def original_images_path(self):
        return str(self.folder / "original")

    def named_path(self, name: str):
        return str(self.folder / name)

    @property
    def working_folder(self):
        """
        'annflux' folder in the project folder
        """
        return (
            os.path.join(base_data_path, self.annflux_folder_) # ty: ignore
            if base_data_path is not None
            else self.annflux_folder_
        )

    @property
    def feature_cache_folder(self):
        return (
            os.path.join(self.working_folder, "feature_cache") # ty: ignore

        )

    @property
    def labels_path(self):
        return os.path.join(self.working_folder, "labels.json") # ty: ignore

    @property
    def label_definitions_path(self):
        return os.path.join(self.working_folder, "label_defs.json") # ty: ignore

    @property
    def exclusivity_path(self):
        return os.path.join(self.working_folder, "exclusivity.csv") # ty: ignore

    @property
    def data_state_path(self):
        """
        Data state (annflux.csv)
        """
        return os.path.join(self.working_folder, "annflux.csv") # ty: ignore

    @property
    def split_path(self):
        return os.path.join(self.working_folder, "split.json") # ty: ignore

    @property
    def repository(self):
        from annflux.repository.repository import Repository
        return Repository(os.path.join(self.working_folder, "datarepo"))

    @property
    def dataset(self):
        from annflux.repository.dataset import Dataset
        return self.repository.get(label=Dataset, tag="unseen").first()

