from __future__ import annotations

import os.path
import shutil
from os import PathLike
import zipfile
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
        req = requests.get(self.url)

        zipfile_ = zipfile.ZipFile(BytesIO(req.content))
        zipfile_.extractall(self.out_folder)
        logger.warning(f"Extracted zip to {self.out_folder}")

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
