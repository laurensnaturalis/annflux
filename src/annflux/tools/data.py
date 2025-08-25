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
import copy
import glob
import itertools
import json
import logging
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import pandas
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex

from annflux.repository.dataset import Dataset
from annflux.shared import AnnfluxSource
from annflux.tools.io import basename_no_extension

from annflux.tools.mixed import get_basic_logger
from annflux.training.annflux.group_classifier_cnn import get_labels


@lru_cache
def canon_(
    multilabel_string: str,
    remove_unknown=False,
    output_separator=",",
    replace_space=False,
) -> str | None:
    """
    Canonizes a multilabel string separated by comma's
    """
    result = (
        output_separator.join(
            sorted(
                [
                    x_.replace(" ", "_") if replace_space else x_
                    for x_ in multilabel_string.split(",")
                    if "?" not in x_ or not remove_unknown
                ]
            )
        )
        if multilabel_string is not None and not pandas.isna(multilabel_string)
        else None
    )
    return result


def get_full_labeling(tuples, leaf):
    # TODO: outside
    parent_map = {}
    for t_ in tuples:
        child, parent = t_[0], t_[1]
        if parent != "null" and parent is not None:
            parent_map[child] = parent

    # Traverse from leaf to root
    full_label = []
    current = leaf
    while current in parent_map:
        full_label.append(current)
        current = parent_map[current]
    full_label.append(current)  # Add the root

    # Reverse to get root-to-leaf order
    full_label.reverse()
    return full_label


def color_and_label(
    data: pandas.DataFrame,
    annotations: dict[str, str],
    label_definitions: list[tuple[str, str]],
    display_update_uids: list[str] = None,
    logger: logging.Logger = get_basic_logger("color_and_label"),
):
    individual_labels = list(
        itertools.chain(
            *[canon_(x_, remove_unknown=True).split(",") for x_ in annotations.values()]
        )
    )

    class_to_count = Counter(individual_labels)
    class_to_count.update(
        Counter([canon_(x_, remove_unknown=True) for x_ in annotations.values()])
    )
    start_time = time.time()
    logger.info(f"color_and_label start={time.time()}")
    #
    data_to_update = (
        data[data.uid.isin(display_update_uids)]
        if display_update_uids is not None
        else data
    )
    logger.info(f"color_and_label: updating {len(data_to_update)} media")
    #
    label_array = np.array(
        [annotations.get(uid) for i, uid in enumerate(data.uid.values)]  # noqa
    )
    cmap = plt.get_cmap("viridis")
    cmap_distinct = plt.get_cmap("tab20b")
    individual_labels_unique = set(individual_labels)
    data_to_update["label_predicted"] = data_to_update["label_predicted"].apply(
        lambda x_: x_
        if len(
            set(
                x_.split(",") if x_ is not None and not pandas.isna(x_) else []
            ).intersection(individual_labels_unique)
        )
        > 0
        else "n/a"
    )

    data["label_true"] = label_array
    labeled_uids = set(annotations.keys())
    data["labeled"] = data["uid"].apply(lambda x_: int(x_ in labeled_uids))

    data_to_update.label_predicted = data_to_update.label_predicted.apply(
        lambda x_: canon_(x_)
    )
    data.label_true = data.label_true.apply(lambda x_: canon_(x_))

    # list of multi-labels (multiple labels in a single string; canonized) and single labels
    unique_predicted_multilabels = set(
        [
            canon_(x_, output_separator=",")
            for x_ in data["label_predicted"].unique()
            if x_ is not None and x_ != "n/a"
        ]
    )
    unique_annotated_multilabels = set(
        [
            canon_(x_, output_separator=",", remove_unknown=True)
            for x_ in annotations.values()
        ]
    )
    unique_labels = sorted(
        list(unique_predicted_multilabels.union(unique_annotated_multilabels)),
        key=lambda x_: class_to_count.get(x_, 0),
        reverse=True,
    )
    for multilabel_ in individual_labels_unique:
        try:
            canon_full = canon_(",".join(get_full_labeling(label_definitions, multilabel_)))
        except TypeError:
            print(multilabel_, get_full_labeling(label_definitions, multilabel_))
            raise
        if multilabel_ not in unique_labels and canon_full not in unique_labels:
            unique_labels.append(multilabel_)

    unique_labels.append("n/a")
    logger.info(f"unique_labels={unique_labels}")
    logger.info(f"unique_labels end={time.time() - start_time}")

    time_start = time.time()
    cluster_to_color = {k: rgb2hex(cmap(k / 100)) for k in range(101)}
    # TODO(critical)
    # data_to_update["color_prob"] = data_to_update.apply(
    #     lambda x_: cluster_to_color[int(x_.score_predicted * 100)]
    #     if not x_.labeled
    #     else "#800080",
    #     axis=1,
    # )
    logger.info(f"color_prob took={time.time() - start_time}")
    #
    if "dp_cluster" in data_to_update.columns:
        time_start = time.time()
        num_clusters = int(data_to_update["dp_cluster"].max())
        cluster_to_color = {
            k: rgb2hex(cmap(float(k / num_clusters))) for k in range(num_clusters + 1)
        }
        data_to_update["dp_cluster_color"] = data_to_update.apply(
            lambda x_: cluster_to_color[int(x_["dp_cluster"])]
            if not pandas.isna(x_["dp_cluster"])
            else "#800080",
            axis=1,
        )
        logger.info(f"coloring dp_cluster took={time.time() - start_time}")
    #
    if "fre" in data_to_update.columns:
        q5, q95 = np.percentile(data_to_update.fre, [1, 99])
        logger.info(f"fre={q5, q95}")
        if q5 != q95:
            data_to_update["fre_for_color"] = data_to_update.fre.apply(
                lambda x_: (np.clip(x_, q5, q95) - q5) / (q95 - q5)
            )
            cluster_to_color = {k: rgb2hex(cmap(k / 100)) for k in range(101)}
            data_to_update["color_fre"] = data_to_update.apply(
                lambda x_: cluster_to_color[int(x_.fre_for_color * 100)]
                if not pandas.isna(x_.fre_for_color)
                else "",
                axis=1,
            )
            del data_to_update["fre_for_color"]
        logger.info(f"fre_for_color took={time.time() - start_time}")
    #
    label_to_float = dict(
        list(
            zip(
                unique_labels[:20],
                np.arange(len(unique_labels[:20])) / len(unique_labels[:20]),
            )
        )
    )

    multilabel_to_color = {
        class_: cmap_distinct(label_to_float[class_])
        for class_ in list(unique_labels[:20])
        if class_ is not None
    }
    colors = multilabel_to_color.values()

    unassigned_list = []
    for multilabel_ in unique_labels[20:]:
        labels_ = multilabel_.split(",")
        if len(labels_) == 1:
            full_labels = get_full_labeling(label_definitions, labels_[0])
        else:
            deepest_label = None
            deepest_level = -1
            for label2_ in labels_:
                full_labels = get_full_labeling(label_definitions, label2_)
                if len(full_labels) > deepest_level:
                    deepest_label = label2_
                    deepest_level = len(full_labels)
            full_labels = get_full_labeling(label_definitions, deepest_label)
        assigned = False
        if len(full_labels) > 1:
            for i_ in range(0, len(full_labels)):
                ancestor_ = full_labels[:-i_] if i_ > 0 else full_labels
                ancestor_canon = canon_(",".join(ancestor_))
                if ancestor_canon in multilabel_to_color:
                    # print(f"Using {ancestor_} for label {label_}")
                    ancestor_color = multilabel_to_color[ancestor_canon]
                    multilabel_to_color[multilabel_] = np.clip(
                        np.array(ancestor_color)
                        + np.random.randn(len(ancestor_color)) * 0.05,
                        0,
                        1,
                    )
                    assigned = True
                    break
        elif full_labels[0] in multilabel_to_color:
            ancestor_color = multilabel_to_color[full_labels[0]]
            multilabel_to_color[multilabel_] = np.clip(
                np.array(ancestor_color) + np.random.randn(len(ancestor_color)) * 0.05,
                0,
                1,
            )
            assigned = True
        if not assigned:
            print(f"{multilabel_} not assigned")
            unassigned_list.append(multilabel_)

    # if the full label tree of an individual label use its same color
    for label_ in individual_labels_unique:
        if label_ not in multilabel_to_color:
            canon_full = canon_(",".join(get_full_labeling(label_definitions, label_)))
            if canon_full in multilabel_to_color:
                multilabel_to_color[label_] = multilabel_to_color[canon_full]

    # print(colors)
    for label_ in unassigned_list:
        if label_ not in multilabel_to_color:
            multilabel_to_color[label_] = list(colors)[np.random.choice(list(range(20)))]

    for key in multilabel_to_color:
        multilabel_to_color[key] = rgb2hex(multilabel_to_color[key])

    def multilabel_to_color_func(x_):
        return (
            multilabel_to_color[canon_(x_, remove_unknown=True)]
            if x_ not in ["n/a", None]
            else "#AAAAAA"
        )

    data["color_class"] = data.apply(
        lambda x_: multilabel_to_color_func(
            x_.label_predicted if not x_.labeled else x_.label_true
        ),
        axis=1,
    )
    logger.info(f"color_class took={time.time() - start_time}")
    #
    if (
        "label_possible" in data_to_update.columns
        and "score_possible" in data_to_update.columns
    ):
        data_to_update["incorrect_score"] = 0.0
        # TODO: use vector update
        for r, row in data_to_update.iterrows():
            label_possible = row.label_possible
            score_possible = row.score_possible
            label_predicted = row.label_predicted
            scores_predicted = row.scores_predicted
            label_true = row.label_true
            if (
                row.labeled == 0
                or pandas.isna(label_possible)
                or pandas.isna(score_possible)
                or len(label_possible) == 0
            ):
                continue
            score = compute_incorrect_score(
                label_possible,
                label_true,
                score_possible,
                label_predicted,
                scores_predicted,
            )
            data_to_update.at[r, "incorrect_score"] = score
        data_to_update.incorrect_score = (
            data_to_update.incorrect_score.max() - data_to_update.incorrect_score
        )
    #
    if "record_id" in data.columns and False:  # TODO(fix): use group data?
        record_ids = data.record_id.unique()
        has_records = len(record_ids) < len(data)
        if has_records:
            data["predicted_x"] = None
            data["true_x"] = None
            for record_id in record_ids:
                sel = data[data.record_id == record_id]
                _, predicted_set = get_labels(sel, "label_predicted")
                _, true_set = get_labels(sel, "label_true")
                all_labels = predicted_set | true_set
                label_to_index = dict(zip(list(all_labels), range(len(all_labels))))

                # within the record assign coordinates & jitter
                for r, row in sel.iterrows():
                    data.at[r, "predicted_x"] = (
                        label_to_index[row.label_predicted] * 2 + np.random.rand()
                    )
                    data.at[r, "true_x"] = (
                        label_to_index[row.label_true] * 2 + np.random.rand()
                    )
    #
    logger.info(f"label_possible took={time.time() - start_time}")
    logger.info(f"coloring took={time.time() - time_start}")

    return multilabel_to_color, class_to_count


def compute_incorrect_score(
    label_possible: str | float,
    label_true: str,
    score_possible: str | float,
    label_predicted: str | float,
    scores_predicted: str | float,
):
    predicted_map = name_to_probability(label_possible, score_possible)
    predicted_map.update(name_to_probability(label_predicted, scores_predicted))
    labels_true = set(label_true.split(","))
    for key in copy.copy(list(predicted_map.keys())):
        if predicted_map[key] < 0.5 and key not in labels_true:
            del predicted_map[key]
    score = 0
    for label_ in set(predicted_map.keys()).union(labels_true):
        score += abs(((label_ in labels_true) * 1) - predicted_map.get(label_, 0.0))
    return score


def name_to_probability(label_possible, score_possible):
    predicted_map = dict(
        list(
            zip(
                label_possible.split(",") if isinstance(label_possible, str) else [],
                map(float, score_possible.split(","))
                if isinstance(score_possible, str)
                else [score_possible],
            )
        )
    )
    return predicted_map


def remove_uids_from_double_check(
    uids: List[str],
    doublecheck_path,
    logger=get_basic_logger("remove_uids_from_double_check"),
):
    if not os.path.exists(doublecheck_path):
        return
    j_doublecheck = json.load(open(doublecheck_path))
    doublecheck_uids = set(j_doublecheck["checked"])
    num_before = len(doublecheck_uids)
    doublecheck_uids -= set(uids)
    num_after = len(doublecheck_uids)
    logger.info(f"remove_uids_from_double_check removed {num_before - num_after}")
    j_doublecheck["checked"] = list(doublecheck_uids)
    with open(doublecheck_path, "w") as f:
        json.dump(j_doublecheck, f, indent=2)


def get_images_path() -> str:
    """
    Get default path for images
    :return:
    """
    return os.path.join(get_project_root(), "images")


def get_thumb_path() -> str:
    """
    Get default path for thumbnails
    :return:
    """
    return os.path.join(get_project_root(), "thumbnails")


def get_group_images_path() -> str:
    """
    Get default path for images
    :return:
    """
    return os.path.join(get_project_root(), "images_group")


def get_failed_images_path() -> str:
    return os.path.join(get_project_root(), "images_failed")


def get_project_root() -> str:
    """
    Get PROJECT_ROOT environment variable, raises ValueError if not set
    :return: PROJECT_ROOT
    """
    project_root = os.getenv("PROJECT_ROOT", None)
    if project_root is None:
        raise ValueError("PROJECT_ROOT not set")
    return project_root


def add_group_to_exclusivity(group_children: List[str], exclusivity_path: str):
    """
    Add an exclusivity group to the exclusivity database
    """
    exclusivity_relations = itertools.combinations(group_children, 2)
    update = pandas.DataFrame(exclusivity_relations, columns=["left", "right"])
    if os.path.exists(exclusivity_path):
        exclusivity_table = pandas.read_csv(exclusivity_path)
        exclusivity_table = pandas.concat(
            [
                exclusivity_table,
                update,
            ],
            ignore_index=True,
        )
        exclusivity_table.drop_duplicates(["left", "right"])
    else:
        exclusivity_table = update
    print(f"Adding {group_children} to {exclusivity_path}")
    exclusivity_table.to_csv(exclusivity_path, index=False)


def create_group_flux_data(source):
    if not os.path.exists(source.group_flux_data_path()):
        uids = [
            basename_no_extension(x_)
            for x_ in os.listdir(source.named_path("images_group"))
        ]
        label_true = [
            None,
        ] * len(uids)
        t = pandas.DataFrame({"uid": uids, "label_true": label_true})
        t.to_csv(source.group_flux_data_path(), index=False)


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


def png_to_jpg(path: str):
    im = Image.open(path)
    rgb_im = im.convert("RGB")
    rgb_im.save(path.replace(".png", ".jpg"))


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
    if not os.path.exists(taxon_mapping_path):  # TODO: check if this is still necessary
        ids = [str(x_) for x_ in range(1000)]
        pandas.DataFrame(data=list(zip(ids, ids)), columns=["label", "taxon"]).to_csv(
            taxon_mapping_path
        )

    #
    split_path = os.path.join(working_folder, "split.json")
    if not os.path.exists(split_path):
        test_uids = np.random.choice(
            unseen_data.uid.values, int(0.10 * len(unseen_data)), replace=False
        ).tolist()
        with open(split_path, "w") as f:
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


def make_backup(source_path, backup_dir="backups"):
    """
    Creates a backup of the source_path in the backup_dir with a timestamp.

    Args:
        source_path (str): Path to the file or directory to back up.
        backup_dir (str): Directory where backups will be stored. Defaults to "backups".
    """
    # Create the backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)

    # Get the base name of the source path
    base_name = os.path.basename(source_path)

    # Create a timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the backup path with timestamp
    backup_path = os.path.join(backup_dir, f"{base_name}_{timestamp}")

    # Copy the file or directory to the backup location
    if os.path.isfile(source_path):
        shutil.copy2(source_path, backup_path)
        print(f"File backed up to: {backup_path}")
    elif os.path.isdir(source_path):
        shutil.copytree(source_path, backup_path)
        print(f"Directory backed up to: {backup_path}")
    else:
        print(f"Error: {source_path} does not exist or is not a file/directory.")
