import glob
import itertools
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Set, Callable, List

import numpy as np
import pandas
import zarr
from PIL import Image
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


def png_to_jpg(path: str):
    im = Image.open(path)
    rgb_im = im.convert("RGB")
    rgb_im.save(path.replace(".png", ".jpg"))


def make_images(images_path_):
    images_path_ = Path(images_path_)
    jpgs = glob.glob(str(images_path_ / "*.jpg"))
    pngs = glob.glob(str(images_path_ / "*.png"))
    all_files = glob.glob(str(images_path_ / "*"))

    if len(pngs) > 0:
        with Pool(32) as pool:
            pool.map(png_to_jpg, pngs)

    if len(all_files) != len(jpgs):
        print(f"Non JPGs in {images_path_}")


def execute(source: AnnfluxSource, architecture="efficientnetb0"):
    source, repo = init_folder(source)

    train_then_features(
        source,
        architecture=architecture,
    )


def get_model_folder(
    repo: Repository, architecture, model_variant, model_type=ClipModel
):
    tag = f"{architecture}:{model_variant}"
    print(f"Looking for {tag=}, {model_type=}")
    model = repo.get(
        label=model_type,  # TODO: generalize
        tag=tag,
    ).last()
    print(f"{model=}")
    if model is None:
        tmp_dir = Path("tmp")
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
            columns=["index", "class_name"],
        ).to_csv(model.class_to_label_path, index=False)
        repo.commit(model, tag="untrained")
        shutil.rmtree(tmp_dir)
        model = repo.get(label=model_type, tag="untrained").first()
    return model


def train_then_features(
    source: AnnfluxSource,
    dataset: Dataset | pandas.DataFrame = None,
    architecture="clip",
    backend="annflux",
    train_method: str = None,
    train_parameters=TrainParameters(num_epochs=15),
    train_model=True,
    model_variant="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
    cache_name: str = None,
    feature_cache: str = None,  # TODO(implement)
) -> (np.array, np.array, Model):
    """
    Returns (features, probs) of `dataset`
    """
    if dataset is None:
        dataset = source.repository.get(label=Dataset, tag="unseen").last()
    if isinstance(dataset, Dataset):
        data = dataset.as_dataframe()
    else:
        data = dataset
    print(f"{type(data)=}, {type(dataset)=}")
    train_folder = os.path.join(source.working_folder, "train_job")
    repo = source.repository
    labels_path = source.labels_path
    if backend == "naturalis-ai":
        if not os.path.exists(labels_path):
            dataset, model = get_untrained_model(repo, architecture)
        else:
            train(source, architecture, repo, train_folder)
            model = repo.get(label=KerasModel, tag="seen").last()
            shutil.rmtree(train_folder)
        extract_and_store_features(dataset, model, repo)
    elif backend == "annflux":
        if architecture == "bioclip":
            from annflux.training.annflux.bioclip import (
                BioClipFeatureExtractor as Extractor,
            )
        elif architecture == "clip":
            from annflux.training.annflux.clip import ClipFeatureExtractor as Extractor
        else:
            raise ValueError(
                f"{architecture} unknown. Choices are {['clip', 'bioclip']}"
            )
        #
        print(f"{model_variant=}")
        model = get_model_folder(source.repository, architecture, model_variant)
        extractor = Extractor(model, logging.getLogger("train_features"))
        extractor.load_model()
        if train_model:
            if train_method == "train_peft":
                data = add_annotations_and_set(data, source)
                model_out_folder = tempfile.mkdtemp()
                class_to_label_path = extractor.train_peft(
                    data, model_out_folder, train_parameters
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
                model = get_model_folder(source.repository, architecture, model_variant)
                extractor = Extractor(model, logging.getLogger("train_features"))
                extractor.load_model()
            else:
                raise ValueError  # TODO
        #
        if False and isinstance(extractor, AttentionMapMixin):
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
            )
            if len(existing_feature_cache_path) > 0:
                existing_feature_cache_path = existing_feature_cache_path[0]
            else:
                existing_feature_cache_path = None
            print(f"{existing_feature_cache_path=}")
            if not os.path.exists(feature_cache_path):
                feature_cache = zarr.create_group(store=feature_cache_path)
                feature_cache.create_array(
                    shape=(len(dataset), feature_size),
                    chunks=(1000, feature_size),
                    dtype="float",
                    name="features",
                )
                num_classes = len(model.index_to_class)
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
        if isinstance(dataset, Dataset):
            make_resultset(dataset, features, repo)
        else:
            print(f"{type(dataset)=} is not a Dataset, skipping make_resultset")
    return features, probs, model


def add_annotations_and_set(data: pandas.DataFrame, source):
    annotations = json.load(open(source.labels_path))
    subset_split = json.load(open(source.split_path))
    test_uids = set(subset_split["test"])
    data["label_true"] = [annotations.get(uid_) for uid_ in data.uid]
    data["subset"] = ["test" if uid_ in test_uids else None for uid_ in data.uid]
    data.dropna(subset=["label_true"], inplace=True)
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
            columns=["index", "class_name"],
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
        data=list(zip(unique_labels, unique_labels)), columns=["label", "taxon"]
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


def init_folder(
    source: AnnfluxSource,
    label_column_name=None,
    start_labels=None,
    exclusivity_groups: List[List[str]] = None,
    refresh_media=False,
    import_stream_metadata=False,
) -> AnnfluxSource:
    if start_labels is None:
        start_labels = []
    if exclusivity_groups is None:
        exclusivity_groups = []
    working_folder = source.working_folder
    data_path = source.data_path
    label_column_for_unseen = (
        source.label_column_for_unseen
        if label_column_name is None
        else label_column_name
    )
    start_labels = [(x_, "null") for x_ in start_labels]  # TODO

    images_path = source.images_folder
    start_labels = source.start_labels if start_labels is None else start_labels
    if len(exclusivity_groups) == 0:
        exclusivity = source.exclusivity
    else:
        exclusivity = []
        for group_children in exclusivity_groups:
            exclusivity.extend(itertools.combinations(group_children, 2))
    id_column = source.id_column

    annflux_folder_exists = os.path.isdir(working_folder)
    if not annflux_folder_exists:
        os.makedirs(working_folder)
        with open(os.path.join(working_folder, "label_defs.json"), "w") as f:
            json.dump({"labels": start_labels}, f)
        pandas.DataFrame(data=exclusivity, columns=["left", "right"]).to_csv(
            os.path.join(working_folder, "exclusivity.csv"), index=False
        )

    unseen_dataset_path = os.path.join(working_folder, "unseen_annflux_data.csv")
    unseen_data = None
    if not os.path.exists(unseen_dataset_path) or refresh_media:
        if not os.path.exists(data_path) or refresh_media:
            make_images(images_path)

            clean_filenames(images_path)
            image_ids = [
                os.path.splitext(x_)[0]
                for x_ in os.listdir(images_path)
                if x_.endswith(".jpg")
            ]
            images_table = pandas.DataFrame(
                data=zip(
                    image_ids,
                    [
                        "foo,bar",
                    ]
                    * len(image_ids),
                ),
                columns=[id_column, label_column_for_unseen],
            )
            images_table.to_csv(data_path, index=False)

        unseen_data = pandas.read_csv(data_path, dtype={id_column: str})
        unseen_data[id_column] = unseen_data[id_column].str.replace("-", "_")
        unseen_data[id_column] = unseen_data[id_column].apply(
            lambda x_: x_.replace(":", "_").replace(".", "_")
        )
        unseen_data["filename"] = unseen_data[id_column].apply(
            lambda x_: os.path.join(images_path, x_ + ".jpg")
        )
        unseen_data["set"] = None
        unseen_data["uid"] = unseen_data[id_column]
        if label_column_for_unseen not in unseen_data.columns:
            print(
                f"'{label_column_for_unseen}' not in {unseen_data.columns}, "
                f"consider to use --label_column_name {{label}}"
            )

        unseen_data["label"] = unseen_data[label_column_for_unseen]
        unseen_data["record_id"] = unseen_data[id_column].apply(lambda x_: x_ + "R")
        #
        if import_stream_metadata:
            stream_metadata = pandas.read_csv(
                os.path.join(source.working_folder, "stream_process.csv")
            )

            unseen_data = pandas.merge(
                unseen_data, stream_metadata, left_on=id_column, right_on="image_id"
            )  # TODO: image_id
        #
        unseen_data.to_csv(unseen_dataset_path, index=False)
    taxon_mapping_path = os.path.join(source.working_folder, "taxon_mapping.csv")
    if not os.path.exists(taxon_mapping_path):
        ids = [str(x_) for x_ in range(1000)]
        pandas.DataFrame(data=list(zip(ids, ids)), columns=["label", "taxon"]).to_csv(
            taxon_mapping_path
        )

    #
    test_path = os.path.join(working_folder, "split.json")
    if not os.path.exists(test_path):
        test_uids = np.random.choice(
            unseen_data.uid.values, int(0.10 * len(unseen_data)), replace=False
        ).tolist()
        with open(test_path, "w") as f:
            json.dump({"test": test_uids}, f)
    repo = source.repository
    if len(repo.get(label=Dataset, tag="unseen")) == 0 or refresh_media:
        dataset = Dataset(unseen_dataset_path, taxon_mapping_path=taxon_mapping_path)
        repo.commit(dataset, tag="unseen")

    return source


def clean_filenames(images_path):
    for fn in os.listdir(images_path):
        if ":" in fn or "." in fn:
            os.rename(
                os.path.join(images_path, fn),
                os.path.join(
                    images_path,
                    fn.replace(":", "_")
                    .replace(".", "_")
                    .replace("=", "_")
                    .replace("_jpg", ".jpg"),
                ),
            )
