from __future__ import annotations

import json
import os.path
import shutil
from os import PathLike
import zipfile

import pandas
import requests
from io import BytesIO

from annflux.tools.mixed import get_basic_logger

logger = get_basic_logger("bombus-plant-test")


class DataSource:
    url = None
    name = None
    hash = None  # TODO

    def __init__(self):
        self.out_folder = os.path.expanduser(f"~/annflux/datasources/{self.name}")
        if not os.path.isdir(self.out_folder):
            self.download()

    def download(self):
        if not os.path.isdir(self.out_folder):
            req = requests.get(self.url)

            zipfile_ = zipfile.ZipFile(BytesIO(req.content))
            zipfile_.extractall(self.out_folder)
            logger.warning(f"Extracted zip to {self.out_folder}")
            os.rename(self.out_folder + "/s_256", self.out_folder + "/images")
        else:
            logger.warning(f"{self.out_folder} already exists")

    @property
    def folder(self):
        return self.out_folder

    def copy_to(self, folder: str | PathLike):
        shutil.copytree(self.out_folder, folder)
        logger.warning(f"Copied to {self.out_folder}")

    @property
    def true_labels_path(self):
        return os.path.join(os.path.dirname(__file__), "true_labels.json")


class BombusPlantTest(DataSource):
    url = "https://zenodo.org/records/15049184/files/images.zip?download=1"
    name = "bombus-plant-test"


class StreetSurfaceVis(DataSource):
    url = "https://zenodo.org/records/11449977/files/s_256.zip?download=1"
    labels_url = (
        "https://zenodo.org/records/11449977/files/streetSurfaceVis_v1_0.csv?download=1"
    )
    name = "streetsurfacevis"

    def __init__(self):
        super().__init__()
        self.true_labels_path_ = os.path.join(self.out_folder, "labels.json")

    def download(self):
        super().download()
        self.true_labels_path_ = os.path.join(self.out_folder, "labels.json")
        t = pandas.read_csv(self.labels_url, dtype={"mapillary_image_id": str})
        with open(self.true_labels_path, "w") as f:
            json.dump(dict(zip(t.mapillary_image_id, t.surface_type)), f, indent=2)

        print(len(t))

    @property
    def true_labels_path(self):
        return self.true_labels_path_

class DiopsisPublic(DataSource):
    url = "TODO"
    labels_url = (
        "TODO"
    )
    name = "diopsis-coco"

    def __init__(self):
        super().__init__()
        self.true_labels_path_ = os.path.join(self.out_folder, "labels.json")

    def download(self):
        pass

    @property
    def true_labels_path(self):
        return self.true_labels_path_
