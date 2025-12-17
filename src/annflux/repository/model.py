# Copyright 2025 Intel Corporation
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
# coding=utf-8
from __future__ import annotations

import json
import os
import shutil
from abc import abstractmethod, ABCMeta

import matplotlib
import pandas

from .dataset import Dataset
from ..tools.io import file_hash

try:
    import _tkinter  # noqa
except ImportError:
    matplotlib.use("Agg")

from .repository import Repository, RepositoryEntry


class Model(object, metaclass=ABCMeta):
    label = "model"
    class_to_label_path: str | None
    def __init__(
        self,
        path_or_entry: str | RepositoryEntry,
        class_to_label_path: str | None = None,
    ):
        """

        :type path_or_entry: RepositoryEntry
        :param path_or_entry:
        :param class_to_label_path:
        """
        self.index_to_class_name: dict[int, str] | None = None
        if isinstance(path_or_entry, str):
            self._path = path_or_entry
            self.source_path = self._path
            self.class_to_label_path = class_to_label_path
        else:
            self.entry = path_or_entry
            self._path = path_or_entry.path
            self.class_to_label_path = os.path.join(self._path, "labels.csv")

    @property
    def size(self) -> int:
        return 96  # MB

    @property
    def path(self) -> str:
        return self._path

    @property
    def model(self):
        return Repository.get_ancestors(self.entry, "model", Model)  # ty: ignore # TODO

    @abstractmethod
    def get_uid(self):
        pass

    @abstractmethod
    def store_contents(self, directory, mode):
        pass

    @property
    def dataset(self) -> Dataset:
        return Repository.get_ancestors(
            self.entry, "dataset", Dataset
        )  # ty: ignore # TODO

    @property
    def index_to_class(self) -> dict[int, str]:
        if self.index_to_class_name is None and isinstance(self.class_to_label_path, str):
            tmp_ = pandas.read_csv(self.class_to_label_path, dtype={"class_name": str})
            self.index_to_class_name = dict(zip(tmp_["index"], tmp_["class_name"]))
        else:
            raise RuntimeError("index_to_class not available")
        return self.index_to_class_name


class ClipModel(Model):
    class_to_label_path: str

    def __init__(self, path_or_entry: RepositoryEntry | str, class_to_label_path=None):
        super().__init__(path_or_entry, class_to_label_path)
        self.folder = (
            path_or_entry if isinstance(path_or_entry, str) else path_or_entry.path
        )
        self.adapter_folder = os.path.join(self.folder, "adapter")
        self.weights_path = os.path.join(
            self.adapter_folder, "adapter_model.safetensors"
        )
        self.class_to_label_path = os.path.join(self.folder, "labels.csv")
        self.model_configuration_path = os.path.join(self.folder, "model.json")

    @property
    def size(self) -> int:
        return 150

    def get_uid(self):
        return (
            file_hash(self.weights_path)
            if os.path.exists(self.weights_path)
            else file_hash(os.path.join(self.folder, "model.json"))
        )

    def store_contents(self, directory, mode):
        shutil.copy(self.class_to_label_path, directory)
        shutil.copy(self.model_configuration_path, directory)
        if os.path.exists(self.adapter_folder):
            shutil.copytree(self.adapter_folder, os.path.join(directory, "adapter"))

    def export_model_package(self, out_folder: str):
        os.makedirs(out_folder, exist_ok=False)
        shutil.copy(self.class_to_label_path, out_folder)
        shutil.copytree(self.adapter_folder, os.path.join(out_folder, "adapter"))
        with open(os.path.join(out_folder, "description.json"), "w") as f:
            json.dump({"entry": self.entry.to_json()}, f)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"


class KerasModel(Model):
    def __init__(
        self,
        model_folder_path_or_entry: str | RepositoryEntry,
        class_id_to_class_path=None,
        model_configuration_path=None,
    ):
        """
        A multiclass classification model which can be stored in a Repository
        :param model_folder_path_or_entry: model folder where temporary results are stored OR RepositoryEntry object
        :param class_id_to_class_path: obsolete, for backwards compatibility
        """
        super().__init__(model_folder_path_or_entry)
        self.class_to_label_path: str
        if isinstance(model_folder_path_or_entry, str):
            self.entry = None
            self._path = model_folder_path_or_entry
            self.source_path = self._path
            self.weights_path = os.path.join(self.source_path, "weights_stage{n}.h5")
            self.protobuf_path = os.path.join(self.source_path, "model_stage{n}/")
            for stage in [3, 2, 1]:
                path = self.weights_path.format(n=stage)
                if os.path.exists(path):
                    print("Found weights path", path)
                    self.weights_path = path
                    break

                if stage == 1:
                    self.weights_path = None

            if class_id_to_class_path is None:
                class_id_to_class_path = os.path.join(self.source_path, "labels.txt")
            self.class_to_label_path = class_id_to_class_path

        elif hasattr(model_folder_path_or_entry, "path"):
            self.entry = model_folder_path_or_entry
            self._path = model_folder_path_or_entry.path
            self.weights_path = os.path.join(self._path, "weights.h5")
            self.protobuf_path = os.path.join(self._path, "model.pb")
            self.class_to_label_path = os.path.join(self._path, "labels.txt")
        else:
            self._path = "tmp"
            self.class_to_label_path = os.path.join(self._path, "labels.txt")
            pass

        self.model_configuration_path: str = (
            os.path.join(self._path, "model.json")
            if hasattr(self, "path")
            else model_configuration_path
        )

        if os.path.exists(self.model_configuration_path):
            self.model_configuration = json.load(open(self.model_configuration_path))
        else:  # backwards compatibility
            self.model_configuration = {
                "architecture": "inception_v3",
                "num_fully_connected_nodes": 1024,
                "squaring_method": "crop",
            }

        # backwards compatibility
        if "squaring_method" not in self.model_configuration:
            self.model_configuration["squaring_method"] = "crop"

        if self.model_configuration["architecture"] in [
            "efficientnetb0",
        ]:
            self.output_node_names = ["dense/Softmax"]
        else:
            raise ValueError(
                "Unknown output_node_name for architecture {}".format(
                    self.model_configuration["architecture"]
                )
            )

    @property
    def architecture(self):
        return self.model_configuration["architecture"]

    def get_uid(self):
        return (
            file_hash(self.weights_path)
            if self.weights_path is not None
            else file_hash(self.model_configuration_path)
        )

    def store_contents(self, directory: str, mode):
        if self.class_to_label_path is not None:
            shutil.copy(self.class_to_label_path, os.path.join(directory, "labels.txt"))
        if self.weights_path is not None:
            shutil.copy(self.weights_path, os.path.join(directory, "weights.h5"))
        shutil.copy(self.model_configuration_path, directory)

    def __repr__(self):
        return f"Model(uid={self.entry.uid},tag=uid={self.entry.tag})"  # ty:ignore[possibly-missing-attribute]
