from typing import Tuple

import numpy as np
import open_clip
import pandas
import torch
from PIL import Image
from numpy._typing import NDArray
from torch.multiprocessing import Pool
from tqdm import tqdm

from annflux.training.annflux.feature_extractor import BaseFeatureExtractor


def compute_feature(image_path_, model, preprocess):
    # return np.random.random(512)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        try:
            image = Image.open(image_path_)
            print(f"{image_path_}")
        except:  # noqa # TODO
            print(f"Failed to read {image_path_}")
            return np.random.random(512)
        image = preprocess(image).unsqueeze(0)
        print("Done")
        return model.encode_image(image)


def batch(iterable, n=1):
    for ndx in range(0, len(iterable), n):
        yield iterable[ndx : min(ndx + n, len(iterable))]


def compute_batch(image_paths, model, preprocess):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        batch_ = []
        for image_path_ in image_paths:
            try:
                image = Image.open(image_path_)
            except:  # noqa
                raise
                print(f"Failed to read {image_path_}")
                image = Image.new("RGB", (299, 299))
            image = preprocess(image).unsqueeze(0)
            batch_.append(image)
        return model.encode_image(torch.vstack(batch_).to("cuda"))


class BioClip2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, repo_model, logger):
        self.repo_model = repo_model
        super().__init__()

    def load_model(self):
        print(torch.cuda.is_available())
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        )
        self.model.to("cuda")
        # tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

    def get_feature_size(self) -> int:
        self.load_model()
        image_size = (
            self.model.encode_image(
                self.preprocess_train(Image.new("RGB", (299, 299))).unsqueeze(0).to("cuda")
            )
            .detach()
            .cpu()
            .numpy()
            .shape
        )
        return image_size[1]

    def compute_features(
        self,
        dataset: pandas.DataFrame,
        classes_: list[str],
        multi=False,
        batch_size=32,
        feature_cache_path: str | None = None,
        flush=True,
        other_feature_cache_path: str | None = None,
    ) -> Tuple[NDArray, NDArray | None]:
        if multi:
            with Pool(4) as pool:
                image_features = pool.starmap(
                    compute_feature,
                    zip(
                        dataset.filename,
                        [
                            self.model,
                        ]
                        * len(dataset),
                        [
                            self.preprocess_val,
                        ]
                        * len(dataset),
                    ),
                )
        else:
            image_features = []
            for batch_paths in tqdm(
                batch(dataset["filename"], n=batch_size),
                desc="computing features",
                total=int(len(dataset) / batch_size),
            ):
                image_features.extend(
                    compute_batch(batch_paths, self.model, self.preprocess_val).cpu()
                )
        return np.vstack(image_features), None

class BioCapFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, repo_model, logger):
        self.repo_model = repo_model
        super().__init__()

    def load_model(self):
        print(torch.cuda.is_available())
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms("hf-hub:imageomics/biocap")
        )
        self.model.to("cuda")
        # tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

    def get_feature_size(self) -> int:
        self.load_model()
        image_size = (
            self.model.encode_image(
                self.preprocess_train(Image.new("RGB", (224, 224))).unsqueeze(0).to("cuda")
            )
            .detach()
            .cpu()
            .numpy()
            .shape
        )
        return image_size[1]

    def compute_features(
        self,
        dataset: pandas.DataFrame,
        classes_: list[str],
        multi=False,
        batch_size=32,
        feature_cache_path: str | None = None,
        flush=True,
        other_feature_cache_path: str | None = None,
    ) -> Tuple[NDArray, NDArray | None]:

        image_features = []
        for batch_paths in tqdm(
            batch(dataset["filename"], n=batch_size),
            desc="computing features",
            total=int(len(dataset) / batch_size),
        ):
            image_features.extend(
                compute_batch(batch_paths, self.model, self.preprocess_val).cpu()
            )
        return np.vstack(image_features), None
