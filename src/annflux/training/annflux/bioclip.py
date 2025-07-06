import numpy as np
import open_clip
import torch
from PIL import Image
from torch.multiprocessing import Pool
from tqdm import tqdm

from annflux.repository.dataset import Dataset
from annflux.training.annflux.feature_extractor import BaseFeatureExtractor


def compute_feature(image_path_, model, preprocess):
    # return np.random.random(512)
    with torch.no_grad(), torch.cuda.amp.autocast():
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
        yield iterable[ndx: min(ndx + n, len(iterable))]


def compute_batch(image_paths, model, preprocess):
    with torch.no_grad(), torch.cuda.amp.autocast():
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
        return model.encode_image(torch.vstack(batch_))


class BioClipFeatureExtractor(BaseFeatureExtractor):
    def load_model(self):
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        # tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")



    def compute_features(
            self, dataset: Dataset, multi=False, batch_size=32
    ) -> np.array:
        if multi:
            df = dataset.as_dataframe()
            with Pool(4) as pool:
                image_features = pool.starmap(
                    compute_feature,
                    zip(
                        df.filename,
                        [
                            self.model,
                        ]
                        * len(df),
                        [
                            self.preprocess_val,
                        ]
                        * len(df),
                    ),
                )
        else:
            image_features = []
            df = dataset.as_dataframe()
            for batch_paths in tqdm(
                    batch(dataset.as_dataframe().filename, n=batch_size),
                    desc="computing features",
                    total=int(len(df) / batch_size),
            ):
                image_features.extend(compute_batch(batch_paths, self.model, self.preprocess_val))
        return np.vstack(image_features)