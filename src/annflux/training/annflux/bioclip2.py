from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import open_clip
import pandas
import torch
from PIL import Image
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from annflux.tools.data import canon_
from annflux.tools.mixed import get_basic_logger
from annflux.training.annflux.feature_extractor import BaseFeatureExtractor, PeftTrainableMixin, TrainParameters


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


class OpenClipImageDataset(Dataset):
    def __init__(self, data_frame: pandas.DataFrame, preprocess, tokenizer):
        self.data_frame = data_frame.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        try:
            img = self.preprocess(Image.open(row["filename"]))
        except Exception:
            print(f"Failed to read {row['filename']} using black image")
            img = self.preprocess(Image.new("RGB", (224, 224)))
        text = self.tokenizer([row["caption"]])[0]
        return img, text


class BioClip2FeatureExtractor(BaseFeatureExtractor, PeftTrainableMixin):
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

    def train_peft(
        self,
        data: pandas.DataFrame,
        out_folder: Path | str,
        train_parameters: TrainParameters,
        logger=None,
    ):
        """
        Fine-tunes the BioClip2 vision encoder with LoRA using a contrastive
        (CLIP-style) loss.  Expects columns 'filename', 'label_true', 'subset'.
        """
        if logger is None:
            logger = get_basic_logger("bioclip2:train_peft")
        if isinstance(out_folder, str):
            out_folder = Path(out_folder)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        data = data.copy()
        data["caption"] = data["label_true"].apply(
            lambda x_: canon_(
                x_,
                remove_unknown=True,
                output_separator=" ",
                replace_space=True,
                remove_sys=True,
            )
        )
        logger.info(f"{Counter(data['caption'])=}")

        counts = (
            data[data.subset != "test"]
            .groupby("caption")
            .size()
            .to_frame(name="count")
            .reset_index()
        )
        sufficient_data_classes = set(counts[counts["count"] >= 3]["caption"]) - {""}
        logger.info(f"{sufficient_data_classes=}")
        data = data[data["caption"].isin(sufficient_data_classes)]

        unique_labels = data.caption.unique().tolist()
        class_to_label_path = out_folder / "labels.csv"
        pandas.DataFrame(
            data={"class_name": unique_labels, "index": list(range(len(unique_labels)))}
        ).to_csv(class_to_label_path)

        data_train = data[data.subset != "test"]
        data_test = data[data.subset == "test"]
        logger.info(f"{len(data_train)=}, {len(data_test)=}")

        data_train, data_val = train_test_split(
            data_train,
            test_size=max(int(0.1 * len(data_train)), len(sufficient_data_classes)),
            stratify=data_train["caption"],
            random_state=42,
        )

        tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

        train_dataset = OpenClipImageDataset(data_train, self.preprocess_train, tokenizer)
        val_dataset = OpenClipImageDataset(data_val, self.preprocess_val, tokenizer)

        def dedup_collate(batch):
            seen = {}
            for img, text in batch:
                key = tuple(text.tolist())
                if key not in seen:
                    seen[key] = (img, text)
            imgs, texts = zip(*seen.values())
            return torch.stack(imgs), torch.stack(texts)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True,
            collate_fn=dedup_collate,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0,
            collate_fn=dedup_collate,
        )

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        lora_model = get_peft_model(self.model, config)
        lora_model.logit_scale.requires_grad_(True)
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        logger.info(f"trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            (p for p in lora_model.parameters() if p.requires_grad), lr=1e-4
        )

        lora_model.to(device)

        for epoch in range(train_parameters.num_epochs):
            lora_model.train()
            running_loss = 0.0
            for images, texts in tqdm(train_loader, desc=f"epoch {epoch}"):
                images = images.to(device)
                texts = texts.to(device)
                optimizer.zero_grad()
                with torch.enable_grad():
                    image_features = lora_model.encode_image(images)
                    text_features = lora_model.encode_text(texts)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = lora_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logits_per_image.T
                targets = torch.arange(logits_per_image.size(0), device=device)
                loss = (criterion(logits_per_image, targets) + criterion(logits_per_text, targets)) / 2.0
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            lora_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, texts in val_loader:
                    images = images.to(device)
                    texts = texts.to(device)
                    image_features = lora_model.encode_image(images)
                    text_features = lora_model.encode_text(texts)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logit_scale = lora_model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.T
                    logits_per_text = logits_per_image.T
                    targets = torch.arange(logits_per_image.size(0), device=device)
                    val_loss += (
                        (criterion(logits_per_image, targets) + criterion(logits_per_text, targets)) / 2.0
                    ).item() * images.size(0)
            val_loss /= len(val_loader.dataset)
            logger.info(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        adapter_path = out_folder / "adapter"
        lora_model.save_pretrained(adapter_path)
        logger.info(f"Saved LoRA adapter to {adapter_path}")

        return class_to_label_path

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
