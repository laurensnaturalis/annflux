import glob
import itertools
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Set, Callable, Tuple

import numpy as np
import pandas
import zarr
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split

from annflux.repository.dataset import Dataset
from annflux.repository.model import KerasModel, Model, ClipModel
from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.shared import AnnfluxSource
from annflux.training.annflux.feature_extractor import (
    make_resultset,
    TrainParameters,
)


def get_repo_model(
    repo: Repository, architecture, model_variant, model_type=ClipModel
) -> Model:
    tag = f"{architecture}:{model_variant}"
    print(f"Looking for {tag=}, {model_type=}")
    model = repo.get(
        label=model_type,  # TODO: generalize
        tag=tag,
    ).last()
    print(f"{model=}")
    if model is None:
        tmp_dir = Path("tmp")
        if os.path.isdir("tmp"):
            shutil.rmtree("tmp")
        tmp_dir.mkdir(exist_ok=False)
        with open(tmp_dir / "model.json", "w") as f:
            json.dump(
                {
                    "architecture": architecture,
                    "model_variant": model_variant,
                },
                f,  # noqa, pycharm bug
                indent=2,
            )
        model = model_type(str(tmp_dir))
        pandas.DataFrame(
            data=list(zip(range(2), ["foo", "bar"])),
            columns=["index", "class_name"],  # ty: ignore
        ).to_csv(model.class_to_label_path, index=False)
        model = repo.commit(model, tag=tag)
        shutil.rmtree(tmp_dir)
    return model


def train_then_features(
    source: AnnfluxSource,
    dataset: Dataset | pandas.DataFrame | None = None,
    architecture="clip",
    backend="annflux",
    train_method: str | None = None,
    train_parameters=TrainParameters(num_epochs=15),
    train_model=True,
    model_variant="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
    cache_name: str | None = None,
    existing_feature_cache_path=None,
) -> Tuple[NDArray, NDArray, Model]:
    """
    Returns (features, probs) of `dataset`
    """
    if dataset is None:
        dataset: Dataset = source.repository.get(label=Dataset, tag="unseen").last()
    if isinstance(dataset, Dataset):
        data = dataset.as_dataframe()
    else:
        data = dataset
    print(f"{type(data)=}, {type(dataset)=}")
    train_folder = os.path.join(source.working_folder, "train_job")
    repo = source.repository
    labels_path = source.labels_path
    if backend == "naturalis-ai":
        raise NotImplementedError
        if not os.path.exists(labels_path):
            dataset, model = get_untrained_model(repo, architecture)
        else:
            # train(source, architecture, repo, train_folder)
            model = repo.get(label=KerasModel, tag="seen").last()
            shutil.rmtree(train_folder)
        extract_and_store_features(dataset, model, repo)
    elif backend == "annflux":
        if architecture == "bioclip":
            from annflux.training.annflux.bioclip import (
                BioClipFeatureExtractor as Extractor,
            )
        elif architecture == "bioclip2":
            from annflux.training.annflux.bioclip2 import (
                BioClip2FeatureExtractor as Extractor,
            )
        elif architecture == "biocap":
            from annflux.training.annflux.bioclip2 import (
                BioCapFeatureExtractor as Extractor,
            )
        elif architecture == "clip":
            from annflux.training.annflux.clip import ClipFeatureExtractor as Extractor
        else:
            raise ValueError(
                f"{architecture} unknown. Choices are {['clip', 'bioclip']}"
            )
        #
        print(f"{model_variant=}")
        model = get_repo_model(source.repository, architecture, model_variant)
        extractor = Extractor(model, logging.getLogger("train_features"))
        extractor.load_model()
        if train_model:
            if train_method == "train_peft":
                data = add_annotations_and_set(data, source)
                model_out_folder = tempfile.mkdtemp()
                class_to_label_path = extractor.train_peft(
                    data, model_out_folder, train_parameters,
                    label_defs_path=source.label_definitions_path
                )
                with open(Path(model_out_folder) / "model.json", "w") as f:
                    json.dump(
                        {
                            "architecture": architecture,
                            "model_variant": model_variant,
                        },
                        f,
                        indent=2,
                    )

                print(model_out_folder)
                repo_model = ClipModel(model_out_folder, class_to_label_path)
                repo.commit(
                    repo_model,
                    ancestors=[dataset],
                    tag=f"clip:{model_variant}",  # TODO: other architectures than CLIP
                    allow_mixed_tags=True,
                )
                shutil.rmtree(model_out_folder)
                model = get_repo_model(source.repository, architecture, model_variant)
                extractor = Extractor(model, logging.getLogger("train_features"))
                extractor.load_model()
            else:
                raise ValueError  # TODO
        #
        if False:  # and isinstance(extractor, AttentionMapMixin):
            output_folder = "/mnt/big/indeed/diopsis-hazehorst-apr5-6-gem/attention"  # TODO(generalize)
            os.makedirs(output_folder, exist_ok=True)
            features = extractor.compute_features_and_attention_map(
                dataset, output_folder
            )
            # TODO(cache)
        else:
            feature_size = extractor.get_feature_size()
            os.makedirs(source.feature_cache_folder, exist_ok=True)
            # print(type(data), dataset)
            cache_name = (
                dataset.entry.uid if isinstance(dataset, Dataset) else cache_name
            )
            if cache_name is None:
                raise ValueError(
                    "Either a Dataset should be passed or cache_name should be non-empty"
                )
            feature_cache_path = os.path.join(
                source.feature_cache_folder,
                f"model_{extractor.repo_model.entry.uid}_dataset_{cache_name}.zarr",
            )
            # TODO: not fool-proof
            existing_feature_cache_path = sorted(
                glob.glob(
                    os.path.join(
                        source.feature_cache_folder,
                        f"model_{extractor.repo_model.entry.uid}_dataset_stream_*.zarr",
                    )
                )
            ) if existing_feature_cache_path is None else [existing_feature_cache_path]
            if len(existing_feature_cache_path) > 0:
                existing_feature_cache_path = existing_feature_cache_path[0]
            else:
                existing_feature_cache_path = None
            print(f"{existing_feature_cache_path=}")
            num_classes = len(model.index_to_class)

            if not os.path.exists(feature_cache_path):
                feature_cache = zarr.create_group(store=feature_cache_path)
                feature_cache.create_array(
                    shape=(len(dataset), feature_size),
                    chunks=(1000, feature_size),
                    dtype="float",
                    name="features",
                )
                feature_cache.create_array(
                    shape=(len(dataset), num_classes),
                    chunks=(1000, num_classes),
                    dtype="float",
                    name="probs",
                )
                feature_cache.create_array(
                    shape=(len(dataset),), chunks=(1000,), dtype=str, name="filenames"
                )
                # TODO: load existing zarr with possibly different size
            features, probs = extractor.compute_features(
                data,
                list(model.index_to_class.values()),
                feature_cache_path=feature_cache_path,
                other_feature_cache_path=existing_feature_cache_path,
            )
            if probs is None:
                probs = np.zeros((len(dataset), num_classes))
        if isinstance(dataset, Dataset):
            make_resultset(dataset, features, repo)
        else:
            print(f"{type(dataset)=} is not a Dataset, skipping make_resultset")
    return features, probs, model


def add_annotations_and_set(data: pandas.DataFrame, source):
    print(f"add_annotations and set {data.shape=}")
    annotations = json.load(open(source.labels_path))
    subset_split = json.load(open(source.split_path))
    test_uids = set(subset_split["test"])
    data["label_true"] = [annotations.get(uid_) for uid_ in data.uid]
    data["subset"] = ["test" if uid_ in test_uids else None for uid_ in data.uid]
    data.dropna(subset=["label_true"], inplace=True)
    print(f"add_annotations and set 2 {data.shape=}")
    return data


def get_untrained_model(repo: Repository, architecture="efficientnetb0"):
    dataset = repo.get(label=Dataset, tag="unseen").last()
    #
    model = repo.get(label=KerasModel, tag="untrained").first()
    if model is None:
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=False)
        with open(tmp_dir / "model.json", "w") as f:
            json.dump(
                {
                    "architecture": architecture,
                    "num_fully_connected_nodes": 0,
                    "squaring_method": "crop",
                },
                f,
                indent=2,
            )
        model = KerasModel(str(tmp_dir))
        pandas.DataFrame(
            data=list(zip(range(2), ["foo", "bar"])),
            columns=["index", "class_name"],  # ty: ignore
        ).to_csv(model.class_to_label_path, index=False)
        repo.commit(model, tag="untrained")
        shutil.rmtree(tmp_dir)
    model = repo.get(label=KerasModel, tag="untrained").first()
    return dataset, model


def extract_and_store_features(dataset, model, repo):
    results_folder = model.validate(
        set_name=None,
        split_size=1024,
        dataset=dataset,
        compute_saliency=False,
        batch_size=256,
    )
    result_set = Resultset(results_folder)
    repo.commit(
        result_set, ancestors=[model, dataset], tag="unseen", allow_mixed_tags=True
    )


def train(
    source: AnnfluxSource,
    architecture,
    repo,
    train_folder,
    deep_backend_func: Callable[[Dataset, Model, str, str], str],
):
    working_folder = source.working_folder
    id_column = source.id_column
    labels_path = source.labels_path
    data_path = source.data_path
    seen_dataset_path = os.path.join(working_folder, "seen.csv")
    seen_taxon_mapping_path = os.path.join(working_folder, "seen_taxon_mapping.csv")
    with open(labels_path) as f:
        annotations = json.load(f)
    with open(os.path.join(working_folder, "split.json")) as f:
        test_uids = set(json.load(f)["test"])
    annotations = {
        k: ",".join([x_ for x_ in v.split(",") if "?" not in x_])
        for k, v in annotations.items()
        if k not in test_uids
    }
    annotated_labels = list(
        itertools.chain(*[labels_.split(",") for labels_ in annotations.values()])
    )
    annotated_counts = Counter(annotated_labels)
    unique_labels = set(
        [
            x_[0]
            for x_ in annotated_counts.most_common()
            if (x_[1] >= 5 and x_[0] != "Animal")  # HACK
        ]
    )
    print("unique_labels", unique_labels)
    pandas.DataFrame(
        data=list(zip(unique_labels, unique_labels)),
        columns=["label", "taxon"],  # ty: ignore
    ).to_csv(seen_taxon_mapping_path)
    labeled_ids = [
        x_
        for x_ in annotations.keys()
        if len(set(annotations[x_].split(",")).intersection(unique_labels)) > 0
    ]
    seen_data = pandas.read_csv(data_path, dtype={id_column: str})
    seen_data[id_column] = seen_data[id_column].apply(
        lambda x_: x_.replace(":", "_")
    )  # TODO
    seen_data = seen_data[seen_data.image_id.isin(set(labeled_ids))]
    seen_labels = [annotations[x_] for x_ in seen_data.image_id]
    seen_data["filename"] = seen_data[id_column].apply(
        lambda x_: os.path.join(source.images_folder, x_ + ".jpg")
    )
    seen_annotated_labels = list(
        itertools.chain(*[labels_.split(",") for labels_ in seen_labels])
    )
    seen_annotated_counts = Counter(seen_annotated_labels)

    def min_count_label(l_, l2_: Set):
        min_count_ = np.inf
        min_label_ = None
        for label_ in l_:
            if annotated_counts[label_] < min_count_ and label_ in l2_:
                min_count_ = seen_annotated_counts[label_]
                min_label_ = label_
        return min_label_

    labels_for_strat = [
        min_count_label(labels_.split(","), set(unique_labels))
        for labels_ in seen_labels
    ]
    print(seen_annotated_counts, Counter(labels_for_strat))
    #
    id_train, id_test = train_test_split(
        seen_data.image_id.values.tolist(),
        test_size=0.20,
        random_state=42,
        stratify=labels_for_strat,
    )
    print(id_train)
    train_val = []
    for id_ in seen_data.image_id:
        if id_ in id_train:
            train_val.append("train")
        else:
            train_val.append("validation")
    seen_data["set"] = train_val
    #
    seen_data["uid"] = seen_data[id_column]
    seen_data["label"] = seen_labels
    seen_data["record_id"] = seen_data[id_column].apply(lambda x_: x_ + "R")
    seen_data.to_csv(seen_dataset_path, index=False)
    dataset = Dataset(seen_dataset_path, taxon_mapping_path=seen_taxon_mapping_path)
    repo.commit(dataset, tag="seen")
    dataset = repo.get(label=Dataset, tag="seen").last()
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    previous_model = repo.get(label=KerasModel, tag="seen").last()
    # TODO: actual implementation of deep training
    deep_backend_func(dataset, previous_model, architecture, train_folder)
    model = Model(train_folder)
    repo.commit(model, tag="seen")
