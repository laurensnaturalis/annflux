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
import json
import logging
import os
import random
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import torch
import zarr
from numpy._typing import NDArray
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from zarr import Group

from annflux.repository.model import Model
from annflux.tools.data import canon_
from annflux.tools.io import basename_no_extension
from annflux.tools.mixed import get_basic_logger
from annflux.training.annflux.clip_shared import Image_dataset
from annflux.training.annflux.feature_extractor import (
    BaseFeatureExtractor,
    OpenVinoMixin,
    PeftTrainableMixin,
    TrainParameters,
    batched,
)
import warnings

# disable UserWarning: The codec `vlen-utf8` is currently not part in the Zarr format 3 specification
warnings.simplefilter("ignore", UserWarning, 44)
# from annflux.training.annflux.gem_model import batched

pandas.options.mode.copy_on_write = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_batches(data_train):
    train_img = data_train.filename.values
    train_caption = data_train.caption.values
    train_counts = Counter(train_caption.tolist())
    names, counts = zip(*train_counts.items())
    weights = np.array(counts, dtype=float)
    weights /= weights.sum()
    train_caption_to_idx = defaultdict(lambda: [])
    for i, val in enumerate(train_caption):
        train_caption_to_idx[val].append(i)
    unique_captions = list(set(train_caption))
    batch_size = len(unique_captions)  # TODO: based on number of classes
    print(batch_size)
    batch_size = min(batch_size, 64)
    num_batches = 1 * (len(train_img) // batch_size)
    new_captions = []
    new_images = []
    for _ in range(num_batches):
        names_for_batch = np.random.choice(
            list(names), size=batch_size, p=weights, replace=False
        )
        for caption in names_for_batch:
            index_ = np.random.choice(train_caption_to_idx[caption])
            assert caption == train_caption[index_]
            new_captions.append(train_caption[index_])
            new_images.append(train_img[index_])

    data_train = pandas.DataFrame(
        data=zip(new_images, new_captions),
        columns=["filename", "caption"],  # ty: ignore
    )
    return batch_size, data_train


def train_model(
    model,
    criterion,
    optimizer,
    data_train,
    val_loader,
    processor,
    custom_batch_builder,
    num_epochs=10,
    checkp_epoch=0,
    scheduler=None,
    log=True,
    plot_file=__file__ + ".log",
    device="cuda",
):
    since = time.time()
    print("len(data_train)", len(data_train))

    my_file = None
    if log:
        my_file = open(plot_file, "a")

    pbar = tqdm(range(checkp_epoch, num_epochs))
    for epoch in pbar:
        batch_size, data_train_batch = make_batches(data_train)

        train_set = Image_dataset(
            root_dir="Images", data_frame=data_train_batch, processor=processor
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_batch_builder,
            drop_last=True,
        )

        model.train()

        running_loss = 0.0
        num_batch_labels = None

        for sample in tqdm(train_loader, f"epoch {epoch}"):
            input_ids, attention_mask, pixel_values, caption = (
                sample["input_ids"],
                sample["attention_mask"],
                sample["pixel_values"],
                sample["caption"],
            )
            assert len(set(caption)) == num_batch_labels or num_batch_labels is None
            num_batch_labels = len(set(caption))
            batch_size = input_ids.size(0)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(input_ids, pixel_values, attention_mask)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text

                targets = torch.arange(logits_per_image.size(0)).long().to(device)

                texts_loss = criterion(logits_per_text, targets)
                images_loss = criterion(logits_per_image, targets)
                loss = (images_loss + texts_loss) / 2.0

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * batch_size

        train_loss = running_loss / len(train_loader)

        model.eval()

        running_loss = 0.0

        with torch.no_grad():
            for sample in val_loader:
                input_ids, attention_mask, pixel_values = (
                    sample["input_ids"],
                    sample["attention_mask"],
                    sample["pixel_values"],
                )
                batch_size = input_ids.size(0)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                pixel_values = pixel_values.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(input_ids, pixel_values, attention_mask)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text

                    targets = torch.arange(logits_per_image.size(0)).long().to(device)

                    texts_loss = criterion(logits_per_text, targets)
                    images_loss = criterion(logits_per_image, targets)
                    loss = (images_loss + texts_loss) / 2.0

                # statistics
                running_loss += loss.item() * batch_size

        val_loss = running_loss / len(val_loader)
        if log:
            data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            df = pandas.DataFrame(data, index=[0])  # ty: ignore
            df.to_csv(my_file, header=False, index=False)
        # print()

        pbar.set_description(
            "train loss {:.4} val loss {:.4}".format(train_loss, val_loss)
        )
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def pad_to_square(image, fill_color=(0, 0, 0)):
    # Get original dimensions
    width, height = image.size

    # Calculate target size (max of width/height)
    target_size = max(width, height)

    # Calculate padding (left, top, right, bottom)
    left = (target_size - width) // 2
    top = (target_size - height) // 2
    right = target_size - width - left
    bottom = target_size - height - top

    # Pad the image (default: black background)
    padded_image = ImageOps.expand(
        image, border=(left, top, right, bottom), fill=fill_color
    )
    return padded_image


class ClipFeatureExtractor(BaseFeatureExtractor, PeftTrainableMixin, OpenVinoMixin):
    def __init__(
        self,
        repo_model: Model,
        logger: logging.Logger,
    ):
        print(f"f{torch.cuda.is_available()=}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.repo_model = repo_model
        self.folder = repo_model.path
        self.labels_path = os.path.join(self.folder, "labels.csv")
        self.repository_entry = (
            json.load(open(os.path.join(self.folder, "description.json")))
            if repo_model is None
            else repo_model.entry
        )
        self.adapter_folder = os.path.join(self.folder, "adapter")
        if not os.path.exists(self.adapter_folder):
            self.adapter_folder = None
        self.configuration = json.load(open(os.path.join(self.folder, "model.json")))
        self.clip_variant = self.configuration["model_variant"]
        self.model: CLIPModel = None
        self.processor: CLIPProcessor = None
        self.index_to_label: dict[int, str] | None = None
        self.label_to_index: dict[str, int] | None = None
        self.load_model()

    def get_feature_size(self) -> int:
        self.load_model()
        return (
            self.compute_outputs([Image.new("RGB", (299, 299))], ["foo", "bar"])[3]
            .detach()
            .cpu()
            .numpy()
            .shape[1]
        )

    def load_model(self):
        print(f"{self.clip_variant=}")
        if self.model is None:
            torch_dtype = torch.float16
            self.model = CLIPModel.from_pretrained(
                self.clip_variant,
                # attn_implementation="flash_attention_2",
                device_map=self.device,
                torch_dtype=torch_dtype,
            )
            self.processor = CLIPProcessor.from_pretrained(self.clip_variant)

            if self.adapter_folder is not None:
                self.logger.info(f"Loading adapter from {self.adapter_folder}")
                label_map = pandas.read_csv(os.path.join(self.folder, "labels.csv"))
                self.index_to_label = dict(
                    zip(label_map["index"], label_map["class_name"])
                )
                self.label_to_index = dict(
                    zip(label_map["class_name"], label_map["index"])
                )
                self.model.load_adapter(self.adapter_folder)

    def compute_features(
        self,
        dataset: pandas.DataFrame,
        classes_: list[str],
        multi=False,
        batch_size=512,
        feature_cache_path: str | None = None,
        flush=True,
        other_feature_cache_path: str | None = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute features, probability tensors for `dataset`
        """
        #
        print(f"{feature_cache_path=}")
        ov_model = None
        # try:
        #     ov_model = self.convert_model()
        # except RuntimeError:
        #     print("Failed to convert model using OpenVino")
        #     # traceback.print_exc()
        #     ov_model = None
        # TODO: implement skip_positions with batching
        other_filenames: NDArray | None = None
        other_features: NDArray | None = None
        other_probs: NDArray
        print("other_feature_cache_path", other_feature_cache_path)
        if other_feature_cache_path is not None:
            other_feature_cache = zarr.open_group(other_feature_cache_path)
            other_filenames = np.array(
                [
                    basename_no_extension(x_)  # ty:ignore[invalid-argument-type]
                    for x_ in other_feature_cache.get(
                        "filenames"
                    )[  # ty:ignore[invalid-argument-type]
                        :
                    ]  # ty:ignore[invalid-argument-type, non-subscriptable, not-iterable]
                ]
            )
            other_features = other_feature_cache.get(
                "features"
            )  # ty:ignore[invalid-assignment]
            other_probs = other_feature_cache.get(
                "probs"
            )  # ty:ignore[invalid-assignment]

        do_batched = True
        if do_batched:
            filenames = dataset.filename
            num_batches = len(filenames) // batch_size + 1
            features_per_batch: list[NDArray] | list[None] = [
                None,
            ] * num_batches
            probs_per_batch: list[NDArray] | list[None] = [
                None,
            ] * num_batches

            print(f"{len(filenames)=}")
            batch_i = 0
            feature_cache = None

            def get_sliced_array(
                feature_cache_: Group, name_: str, start_: int, end_: int
            ) -> NDArray:
                return feature_cache_.get(name_)[start_:end_]  # ty: ignore

            for batch in tqdm(
                batched(filenames, batch_size),
                total=num_batches,
                desc="Computing features",
            ):
                if feature_cache_path is not None:
                    start: int = batch_i * batch_size
                    end: int = start + batch_size
                    if feature_cache is None:
                        feature_cache = zarr.open_group(feature_cache_path)
                    cache_filenames = get_sliced_array(
                        feature_cache, "filenames", start, end
                    )

                    if cache_filenames[0] != "0":
                        if len(cache_filenames) == len(filenames[start:end]) and np.all(
                            cache_filenames == filenames[start:end]
                        ):
                            cache_batch_features = get_sliced_array(
                                feature_cache, "features", start, end
                            )
                            # cache_batch_features = feature_cache.get(
                            #     "features"
                            # )[  # ty: ignore
                            #     start:end
                            # ]
                            feature_sum = np.all(
                                np.sum(cache_batch_features, axis=1) != 0
                            )
                            if (
                                feature_sum
                            ):  # TODO: replace by explicit column for computed or not
                                print(
                                    f"Features already in cache, skipping batch, {cache_batch_features.shape=}, {feature_sum=}, {batch_i=}"
                                )
                                features_per_batch[batch_i] = cache_batch_features
                                probs_per_batch[batch_i] = get_sliced_array(
                                    feature_cache, "probs", start, end
                                )[:, :2]

                                batch_i += 1
                                continue
                        else:
                            print(list(zip(cache_filenames, filenames[start:end]))[:10])
                            print("Filename mismatch with cache, aborting")
                            exit(1)
                    elif other_features is not None:
                        indices = np.array(
                            [
                                int(np.where(other_filenames == item)[0][0])
                                if np.any(other_filenames == item)
                                else -1
                                for item in [
                                    basename_no_extension(x_)
                                    for x_ in filenames[start:end]
                                ]
                            ]
                        )
                        if np.all(indices > -1):
                            print(
                                f"Features already in other cache, skipping batch {batch_i=}, {other_features[indices].shape=}, {other_probs[indices].shape=}"
                            )
                            features_per_batch[batch_i] = other_features[indices]
                            probs_per_batch[batch_i] = other_probs[indices]
                            batch_i += 1
                            continue

                    else:
                        # no cache yet
                        pass
                images_ = []
                for filename in batch:
                    try:
                        image = Image.open(filename)
                        image.load()
                        pad_to_square(image)
                    except:  # noqa
                        print(f"Failed to read {filename}")
                        image = Image.new("RGB", (299, 299))
                    images_.append(image)

                with torch.no_grad():
                    with torch.autocast(self.device):
                        print(f"{classes_=}")
                        outputs = self.compute_outputs(images_, classes_)
                features_for_batch = outputs[3].cpu().numpy()

                features_per_batch[batch_i] = features_for_batch
                #
                logits_per_image = outputs[0]  # .cpu().numpy()
                # print(f"{logits_per_image.shape=}, {torch.max(logits_per_image)=}")
                # _, predictions = torch.max(logits_per_image, 1)
                probs_for_batch = torch.softmax(logits_per_image, 1).cpu().numpy()
                probs_per_batch[batch_i] = probs_for_batch
                print(f"{logits_per_image.shape=}, {probs_for_batch.shape=}")
                #
                if feature_cache_path is not None:
                    start = batch_i * batch_size
                    end = start + batch_size
                    if feature_cache is None:
                        feature_cache = zarr.open_group(feature_cache_path)
                    feature_cache.get("features")[
                        start:end, :
                    ] = (  # ty:ignore[invalid-assignment]
                        features_for_batch  # ty:ignore[invalid-assignment]
                    )
                    feature_cache.get("probs")[
                        start:end, :
                    ] = (  # ty:ignore[invalid-assignment]
                        probs_for_batch  # ty:ignore[invalid-assignment]
                    )
                    feature_cache.get("filenames")[
                        start:end
                    ] = (  # ty:ignore[invalid-assignment]
                        batch  # ty:ignore[invalid-assignment]
                    )
                    print(
                        f"writing {len(features_for_batch)=} to {feature_cache_path=} at {(start, end)}"
                    )
                    if flush:
                        # zarr 3.0 has no flush yet, this is a workaround
                        del feature_cache
                        feature_cache = None
                batch_i += 1
            # end - batch loop
            features = np.vstack(features_per_batch)  # ty: ignore[no-matching-overload]
            probs = np.vstack(probs_per_batch)  # ty: ignore[no-matching-overload]

        else:
            raise NotImplementedError("probs")  # TODO
        return features, probs

    def compute_outputs(self, images_, text: list[str]):
        if self.processor is not None and self.model is not None:
            inputs = self.processor(
                text=text,
                images=images_,
                return_tensors="pt",
                padding=True,
            )
            inputs.to(self.device)
            outputs = self.model(**inputs)

            return outputs
        raise RuntimeError("compute_outputs called on uninitialized object")

    def train_peft(
        self,
        data: pandas.DataFrame,
        out_folder: Path | str,
        train_parameters: TrainParameters,
        logger=get_basic_logger("clip:train_peft"),
    ):
        """
        Assumes columns 'filename', 'label_true'
        """
        if self.processor is None:
            raise RuntimeError("train_peft called on uninitialized object")
        if isinstance(out_folder, str):
            out_folder = Path(out_folder)
        print(f"{Counter(data['label_true'])=}")
        data["caption"] = data["label_true"].apply(
            lambda x_: canon_(
                x_,
                remove_unknown=True,
                output_separator=" ",
                replace_space=True,
                remove_sys=True,
            )
        )
        print(f"{data.caption=}")
        counts = (
            data[data.subset != "test"]
            .groupby("caption")
            .size()
            .to_frame(name="count")
            .reset_index()
        )
        print(counts)
        sufficient_data_classes = set(counts[counts["count"] >= 3]["caption"]) - {""}
        print(f"{sufficient_data_classes=}")
        data = data[data["caption"].isin(sufficient_data_classes)]
        #
        unique_labels = data.caption.unique()
        class_to_label_path = out_folder / "labels.csv"
        pandas.DataFrame(
            data={"class_name": unique_labels, "index": list(range(len(unique_labels)))}
        ).to_csv(class_to_label_path)
        #
        data_train = data[data.subset != "test"]
        data_test = data[data.subset == "test"]
        logger.info(f"{data_train.size=}")
        logger.info(f"{data_test.size=}")

        print(Counter(data_train["caption"]))
        data_train, data_val = train_test_split(
            data_train,
            test_size=max(int(0.1 * len(data_train)), len(sufficient_data_classes)),
            stratify=data_train["caption"],
            random_state=42,
        )

        model = self.model

        np.random.seed(42)
        torch.manual_seed(42)

        val_set = Image_dataset(
            root_dir="Images", data_frame=data_test, processor=self.processor
        )
        test_set = Image_dataset(
            root_dir="Images", data_frame=data_test, processor=self.processor
        )
        unique_labels = list(data_test.caption.unique())

        def custom_batch_builder(samples):
            img, caption = zip(*samples)

            # noinspection PyCallingNonCallable
            inputs_ = self.processor(
                text=caption, images=list(img), return_tensors="pt", padding=True
            )
            inputs_["caption"] = np.array(caption, dtype=object)
            return inputs_

        def test_batch_builder(samples):
            img, caption = zip(*samples)

            # noinspection PyCallingNonCallable
            inputs_ = self.processor(
                text=unique_labels, images=list(img), return_tensors="pt", padding=True
            )
            inputs_["caption"] = np.array(caption, dtype=object)
            return inputs_

        train_set_size = len(data_train)
        print("Train set size:", train_set_size)
        val_set_size = len(val_set)
        print("Val set size:", val_set_size)

        criterion = nn.CrossEntropyLoss()

        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        val_loader = DataLoader(
            val_set,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_batch_builder,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            collate_fn=test_batch_builder,
        )

        model = model.to(self.device)  # ty: ignore
        inputs = next(iter(val_loader))
        for key in inputs.keys():
            print("Sample {} shape ".format(key), inputs[key].shape)

        model.eval()
        # test(
        #     model, test_loader, test_set, unique_labels, self.device
        # )

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules="all-linear",  # ["q_proj", "v_proj"],  # "all-linear",
            lora_dropout=0.1,
            bias="none",
        )

        lora_model = get_peft_model(model, config)
        print_trainable_parameters(lora_model)
        #

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        model = train_model(
            lora_model,
            criterion,
            optimizer,
            data_train,
            val_loader,
            self.processor,
            custom_batch_builder,
            num_epochs=train_parameters.num_epochs,
            scheduler=None,
            device=self.device,
        )
        model.save_pretrained(out_folder / "adapter")
        model.eval()
        # acc, predictions, probs, hier_probs = test(
        #     model, test_loader, test_set, unique_labels
        # )
        # print("perf", acc)
        # print(accuracy_score(predictions, test_true_vals))
        # data_test["predictions"] = predictions
        # data_test["probability"] = probs
        # # for level in range(6):
        #     data_test[f"level_{level}"] = [
        #         hier_prob.get(level)[0] if hier_prob.get(level) else None
        #         for hier_prob in hier_probs
        #     ]
        #     data_test[f"level_{level}_probability"] = [
        #         hier_prob.get(level)[1] if hier_prob.get(level) else None
        #         for hier_prob in hier_probs
        #     ]

        # data_test.to_csv(out_folder / "predicted.csv")

        self.model.config.torchscript = True

        return class_to_label_path
