# Copyright 2025 Intel Corporation
# Copyright 2025 Naturalis Biodiversity Center
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
from __future__ import annotations

import argparse
import copy
import datetime
import glob
import json
import os
import shutil
import subprocess
from argparse import ArgumentParser, _HelpAction
from collections import defaultdict
from http.client import RemoteDisconnected
from time import sleep
from typing import Optional

import duckdb
import pandas
import requests
from tqdm import tqdm
from urllib3.exceptions import ProtocolError

from annflux.repo_results_to_embedding import embed_and_prepare
from annflux.repository.model import ClipModel
from annflux.scripts.extract_video_frames import frame_capture
from annflux.scripts.tile_images import tile_and_save, link_files, download_files
from annflux.shared import AnnfluxSource
from annflux.tools.api_sdk import is_port_open, call_predict
from annflux.tools.mixed import get_logger
from annflux.train_features import train_then_features
from annflux.tools.data import init_folder
from annflux.training.annflux.feature_extractor import TrainParameters
import annflux

logger = get_logger("cli.log", mode="w")


def export_model_package(source: AnnfluxSource, out_folder: Optional[str]):
    repo = source.repository
    model: ClipModel = repo.get(label=ClipModel).last()  # TODO: generalize
    if model is None:
        raise RuntimeError(f"No model found in {repo}")
    if out_folder is None:
        out_folder = os.path.join(
            source.folder, "model_package", f"{model.entry.date}-{model.entry.uid}"
        )
    model.export_model_package(out_folder)
    shutil.copy(
        os.path.join(annflux.__path__[0], "training/annflux/clip_server.py"),
        out_folder,
    )
    shutil.copy(
        os.path.join(annflux.__path__[0], "training/annflux/clip_shared.py"),
        out_folder,
    )

    first_image_path = glob.glob(source.images_folder + "/*.jpg")[0]
    shutil.copy(first_image_path, out_folder)

    usage = f"""Usage 
    source activate annflux_whl
    cd {out_folder}
    python clip_server.py .
    # in another shell tab
    curl -X POST -F "image=@{os.path.basename(first_image_path)}" http://127.0.0.1:8008/v1/predict | jq .
    """
    print(usage)
    with open(os.path.join(out_folder, "readme.md"), "w") as f:
        f.write(usage)


def execute(arg_list: list[str] | None = None):
    # Create the main parser
    parser = argparse.ArgumentParser(description="AnnFlux command", add_help=False)
    subparsers = parser.add_subparsers(dest="command")

    data_csv_path_help, label_column_name_help, start_labels_help = make_init_parser(
        subparsers, parser
    )

    architectures = ["clip", "bioclip", "bioclip2"]
    clip_variants = [
        "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "openai/clip-vit-base-patch32",
    ]

    features_parser = make_features_parser(subparsers, architectures, clip_variants)

    make_train_then_features_parser(subparsers, [features_parser])
    #
    embedding_parser = subparsers.add_parser(
        "embed", help="Embed the features in the 2D space for display"
    )
    embedding_parser.add_argument("folder", type=str, help="Data folder")

    make_go_parser(
        subparsers,
        [features_parser],
        "data_csv_path_help",
        "label_column_name_help",
        "start_labels_help",
    )
    make_data_parser(subparsers, [features_parser])
    #
    export_parser = subparsers.add_parser("export", help="TODO")
    export_parser.add_argument("folder", type=str, help="Data folder")
    export_parser.add_argument(
        "--out_folder", type=str, help="Model package folder", default=None
    )

    args = parser.parse_args(arg_list)
    folder = os.path.expanduser(args.folder)
    source = AnnfluxSource(folder)
    if args.command == "init":
        source = init_folder(
            source,
            label_column_name=args.label_column_name,
            start_labels=args.start_labels,
            refresh_media=args.refresh_media,
            import_stream_metadata=args.import_stream_metadata,
        )
        print(f"Initialized AnnFlux in folder {source.working_folder}")
    elif args.command == "train_then_features":
        train_then_features(
            source,
            architecture=args.architecture,
            train_method=args.train_peft,
            train_parameters=TrainParameters(num_epochs=args.num_epochs),
        )
    elif args.command == "features":
        train_then_features(
            source,
            train_model=False,
            architecture=args.architecture,
            model_variant=args.model_variant,
        )
    elif args.command == "embed":
        embed_and_prepare(source)
    elif args.command == "go":
        label_column_name = args.label_column_name
        start_labels = args.start_labels
        exclusivity_groups = [group_.split(",") for group_ in args.exclusivity]
        # import_stream_metadata = args.import_stream_metadata # TODO
        architecture = args.architecture
        feature_cache = args.feature_cache
        go_command(
            source,
            architecture,
            exclusivity_groups,
            feature_cache,
            False,
            label_column_name,
            start_labels,
        )
    elif args.command == "export":
        source = source
        export_model_package(
            source,
            os.path.join(source.working_folder, args.out_folder)
            if args.out_folder is not None
            else None,
        )
    elif args.command == "data":
        source = source
        if args.subcommand == "stream":
            stream(
                args.percentage_to_add,
                args.data_input,
                source,
                subsample=args.subsample,
            )


def go_command(
    source,
    architecture="clip",
    exclusivity_groups=None,
    feature_cache=None,
    import_stream_metadata=False,
    label_column_name="label",
    start_labels=None,
    random_seed=None,
):
    source = init_folder(
        source,
        label_column_name=label_column_name,
        start_labels=start_labels,
        exclusivity_groups=exclusivity_groups,
        import_stream_metadata=import_stream_metadata,
        random_seed=random_seed,
    )
    print(f"Initialized AnnFlux in folder {source.working_folder}")
    train_then_features(
        source,
        architecture=architecture,
        train_model=False,
        # feature_cache=feature_cache
    )
    embed_and_prepare(source)


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def write_table(table: pandas.DataFrame, out_path: str):
    tmp_file = out_path + ".tmp"
    # TODO: write to Parquet if env variable is set
    table.to_csv(tmp_file, index=False)

    old_path = out_path + ".old"
    if os.path.exists(out_path):
        shutil.move(out_path, old_path)
    shutil.move(tmp_file, out_path)
    if os.path.exists(old_path):
        os.remove(old_path)


def read_table(path: str, dtype=None):
    if path.endswith(".csv"):
        pq_path = rreplace(path, ".csv", ".pq", 1)
        if not os.path.exists(pq_path):
            table = pandas.read_csv(path, dtype=dtype)
            print(f"Converting {path} to parquet {pq_path}")
            table.to_parquet(pq_path)
        else:
            raise RuntimeError(f"Both {pq_path} and {path} exist, cannot continue")
        path = pq_path
    return pandas.read_parquet(path)


stream_pipeline_steps = {
    "extract_video_frames": frame_capture,
    "tile": tile_and_save,
    "link_files": link_files,
    "download_files": download_files,
}


def stream(
    al_selection_fraction: float,
    data_input: str,
    source: AnnfluxSource,
    update_when_model_changed=False,
    use_webserver=False,
    subsample=None,
    sort_stream=True,
):
    stream_logger = get_logger("stream_process.log", "a")
    stream_process_config_path = os.path.join(source.working_folder, "stream.json")
    if os.path.exists(stream_process_config_path):
        stream_process_config = json.load(open(stream_process_config_path))
    else:
        stream_process_config = {"pipeline": []}

    stream_process_path = os.path.join(source.working_folder, "stream_process.csv")

    # preprocess raw data
    previous_output_folder = None
    num_steps = len(stream_process_config["pipeline"])
    for s, step in enumerate(stream_process_config["pipeline"]):
        is_last_step = s == num_steps - 1
        action = step["action"]
        existing_original_paths = set()
        # TODO: this means there can be only one step with the same action
        stream_step_path = os.path.join(
            source.working_folder, f"stream_process_{action}.csv"
        )
        stream_step_table = None
        if os.path.exists(stream_step_path):
            stream_step_table = pandas.read_csv(
                stream_step_path, dtype={"original_image_id": str}
            )
            existing_original_paths = set(stream_step_table.original_path)
            print(f"{len(stream_step_table)=}")
        input_folder = step.get(
            "input_folder", data_input if s == 0 else previous_output_folder
        )
        output_folder = os.path.join(source.folder, step.get("output_folder"))
        os.makedirs(output_folder, exist_ok=True)
        kwargs = copy.deepcopy(step)
        del kwargs["output_folder"]
        del kwargs["action"]
        if "input_folder" in kwargs:
            del kwargs["input_folder"]
        new_table = stream_pipeline_steps[action](
            input_folder, output_folder, existing_original_paths, **kwargs
        )
        #
        if stream_step_table is not None:
            if len(new_table) > 0:
                stream_step_table = pandas.merge(
                    stream_step_table,
                    new_table,
                    how="outer",
                    on=list(
                        set(stream_step_table.columns)
                        - {"label_possible", "label_probability", "model_version"}
                    ),
                    suffixes=("", "_new"),
                )
            # drop duplicate columns
            stream_step_table = stream_step_table.loc[
                :, ~stream_step_table.columns.duplicated()
            ].copy()
            for column in stream_step_table.columns:
                if "_new" in column:
                    del stream_step_table[column]
            del new_table
            logger.info(f"{len(stream_step_table)=}")
        else:
            stream_step_table = new_table
            if is_last_step:
                stream_step_table["label_possible"] = None
                stream_step_table["label_probability"] = None
                if "model_version" not in stream_step_table:
                    stream_step_table["model_version"] = None

                if sort_stream:
                    stream_step_table.sort_values(by="image_id", inplace=True)
                    stream_step_table.reset_index(drop=True, inplace=True)
        write_table(stream_step_table, stream_step_path)
        #

        previous_output_folder = output_folder

    if use_webserver:
        model_version, port = start_webservice(source)

    #
    stream_process_table = stream_step_table
    print(f"{len(stream_process_table)=}")
    table_to_predict = stream_process_table[
        pandas.isna(stream_process_table.label_possible)
        | (
            update_when_model_changed
            and (stream_process_table.model_version != model_version)
        )
    ]
    #
    print(f"{len(table_to_predict)=}")
    tmp_path = stream_process_path + ".tmp.csv"
    if os.path.exists(tmp_path):
        print(f"Found {tmp_path=}, consider re-using it")

    if use_webserver:
        inference_edge_server(
            model_version, port, stream_process_table, table_to_predict, tmp_path
        )
    else:
        table_to_predict["filename"] = table_to_predict[
            "path"
        ]  # TODO: patch_path --> path
        features, probs, model = train_then_features(
            source,
            table_to_predict,
            "clip",
            train_model=False,
            cache_name=f"stream_{len(table_to_predict)}",  # TODO: not fail-safe
        )
        import numpy as np

        max_probs = np.max(probs, axis=1)
        max_class = np.argmax(probs, axis=1)
        i_ = 0
        model_uid = model.entry.uid
        index_to_class_map = model.index_to_class
        for r, row_ in tqdm(
            table_to_predict.iterrows(),
            desc="writing predictions",
            total=len(table_to_predict),
        ):
            stream_process_table.at[r, "label_possible"] = index_to_class_map[
                max_class[i_]
            ]  # TODO: this actually does multiclass right now, not multilabel
            stream_process_table.at[r, "label_probability"] = max_probs[i_]
            stream_process_table.at[r, "model_version"] = model_uid
            i_ += 1

    # stream_process_table.to_csv(stream_process_path, index=False)
    write_table(stream_process_table, stream_process_path)
    # - Select data using AL
    if (
        al_selection_fraction < 1.0
        and "label_probability" in stream_process_table.columns
    ):
        image_level = (
            stream_process_table[
                ~pandas.isna(stream_process_table["label_probability"])
            ]
            .groupby(by="original_id")
            .mean("label_probability")  # noqa
            .reset_index()
        )
        print(image_level)
        print(image_level.columns)
        import numpy as np

        weights = np.array(1 - image_level["label_probability"] ** 3).copy()
        weights /= weights.sum()
        to_include = np.random.choice(
            image_level["original_id"],
            int(al_selection_fraction * len(image_level)),
            p=weights,
            replace=False,
        )
        al_selection = image_level[image_level["original_id"].isin(set(to_include))][
            "original_id"
        ]
        print(al_selection)
    else:
        image_level = (
            stream_process_table.groupby(by="original_id").size().reset_index()
        )

        al_selection = image_level["original_id"]

    data_to_add = stream_process_table[
        stream_process_table["original_id"].isin(al_selection)
    ]
    # - Run 'data add'
    # TODO: make atomic operation
    os.makedirs(source.images_folder, exist_ok=True)
    stream_process_table["date_to_project"] = None
    for r, row in data_to_add.iterrows():
        shutil.copy(row.path, source.images_folder)
        stream_logger.info(f"Adding {row.path} to project")
        stream_process_table.loc[r, "date_to_project"] = (
            datetime.datetime.now().isoformat()
        )
    write_table(stream_process_table, stream_process_path)
    print("run init with --refresh_media --import_stream_metadata")
    # if os.path.exists(source.data_path):
    #     images = pandas.read_csv(source.data_path)
    #     images = pandas.concat([images, data_to_add], ignore_index=True)
    # else:
    #     images = data_to_add
    # images.to_csv(source.data_path, index=False)
    # TODO: modify stream_process_path when image has been added
    # - Run 'train_then_features'
    # - Run 'embed'


def subsample_minute(subsample, table_to_predict):
    # minutes
    # assume uid in format `white_20250410032936_x2048_y512 # TODO: gen
    time_part_length = len("202504100329")
    # print(table_to_predict.columns)
    time_unit = "minute"
    table_to_predict[time_unit] = table_to_predict.image_id.apply(
        lambda x_: int(x_.split("_")[1][:time_part_length])
    )
    table_to_predict = subsample_time_unit(subsample, table_to_predict, time_unit)
    return table_to_predict


def subsample_hour(subsample, table_to_predict):
    # assume uid in format `white_20250410032936_x2048_y512 # TODO: gen

    time_part_length = len("2025041003")
    time_unit = "hour"
    table_to_predict[time_unit] = table_to_predict.image_id.apply(
        lambda x_: int(x_.split("_")[1][:time_part_length])
    )
    table_to_predict = subsample_time_unit(subsample, table_to_predict, time_unit)
    return table_to_predict


def subsample_time_unit(subsample, table_to_predict, time_unit_name):
    import numpy as np

    indices_to_use = []
    seen_time_units = set()
    if not np.all(sorted(table_to_predict.image_id) == table_to_predict.image_id):
        # TODO: assumes data is sorted
        raise NotImplementedError
    original_to_instance_indices = get_original_to_instance_indices(table_to_predict)
    # TODO: screen should not be added here but during data import
    table_to_predict["screen"] = table_to_predict.original_image_path.apply(
        lambda x_: "white" if "white" in x_ else "yellow"
    )
    for r_, row_ in tqdm(
        table_to_predict.iterrows(),
        desc=f"subsampling using {subsample}",
        total=len(table_to_predict),
    ):
        time_unit = row_[time_unit_name]
        screen = row_["screen"]
        if (time_unit, screen) not in seen_time_units:
            seen_time_units.add((time_unit, screen))
            # make sure to add all instances
            indices_to_use.extend(original_to_instance_indices[row_.original_image_id])
    del table_to_predict[time_unit_name]
    print(
        f"{len(table_to_predict)=}, {indices_to_use[:10]}, {table_to_predict.index.values[:10]=}"
    )
    table_to_predict = table_to_predict.loc[indices_to_use]
    print(f"{len(table_to_predict)=}")
    return table_to_predict


def get_original_to_instance_indices(table_to_predict: pandas.DataFrame):
    original_to_instance_indices: dict[str, list[int]] = defaultdict(lambda: [])
    # for r_, row_ in tqdm(table_to_predict.iterrows(), desc="mapping original to instances", total=len(table_to_predict)):
    #     original_to_instance_indices[row_.original_image_id].append(r_)
    group_result = duckdb.sql("""WITH indexed_images AS (
                SELECT
                    original_image_id,
                    ROW_NUMBER() OVER () AS row_index
                FROM
                    table_to_predict
            )
            SELECT
                original_image_id,
                ARRAY_AGG(row_index) AS row_indices,
                COUNT(*) AS group_count
            FROM
                indexed_images
            GROUP BY
                original_image_id;

            """).fetchall()
    table_indices = table_to_predict.index.values
    for row_ in group_result:
        original_to_instance_indices[row_[0]] = table_indices[
            [x_ - 1 for x_ in row_[1]]  # duckdb uses 1-based indices
        ]
    return original_to_instance_indices


def start_webservice(source):
    # columns: ['original_image_path', 'original_image_id', 'image_id', 'patch_x',
    #        'patch_y', 'datetime', 'label', 'patch_path']
    # - Run table against edge server
    port = 8008
    # TODO: check model version on running server is same as requested model
    webservice_running = is_port_open("localhost", port)  # TODO
    if not webservice_running:
        logfile_path = "edge_service.log"
        model_package_root_folder = os.path.join(source.folder, "model_package")
        # TODO: use object repo
        # TODO: make model version CLI configurable
        model_package_folder = sorted(glob.glob(model_package_root_folder + "/*"))[-1]
        model_version = os.path.split(model_package_folder)[-1].split("-")[-1]
        logger.info(f"Using {model_package_folder}")
        with open(logfile_path, "w") as log_file:
            process = subprocess.Popen(
                "python clip_server.py .",
                stdout=log_file,
                stderr=log_file,
                cwd=model_package_folder,  # TODO(other locations)
                shell=True,
            )

        print(f"Web service started with PID {process.pid}")
        print("waiting for service to start")
        sleep(10)
    else:
        # TODO: get version from running server
        raise NotImplementedError("get version from running server")
        print(f"(some) service already running at {port}")
    return model_version, port


def inference_edge_server(
    model_version, port, stream_process_table, table_to_predict, tmp_path
):
    num_updated = 0
    for r, row in tqdm(
        table_to_predict.iterrows(),
        total=len(table_to_predict),
        desc="predicting",
    ):
        try:
            j_out, _ = call_predict(
                [row.patch_path], f"http://0.0.0.0:{port}/v1/predict"
            )
        except (
            RemoteDisconnected,
            requests.exceptions.ConnectionError,
            ProtocolError,
        ):
            print(f"RemoteDisconnected for {row.patch_path}")
            continue
        prediction = j_out["predictions"][0]["classes"]["items"][0] # ty: ignore
        # print(prediction)
        stream_process_table.loc[r, "label_possible"] = prediction["name"]
        stream_process_table.loc[r, "label_probability"] = prediction["probability"]
        stream_process_table.loc[r, "model_version"] = model_version
        num_updated += 1

        if (num_updated % 100) == 0:
            stream_process_table.to_csv(tmp_path, index=False)


def make_init_parser(subparsers, parent_parser):
    init_parser = subparsers.add_parser(
        "init", add_help=False
    )  # , help="Initialize AnnFlux")
    init_parser.add_argument("folder", type=str, help="Data folder")
    data_csv_path_help = "The filename of images CSV"
    init_parser.add_argument(
        "--data_csv_path",
        type=str,
        help=data_csv_path_help,
        default="images.csv",
    )
    label_column_name_help = "The name for the true label if externally provided"
    init_parser.add_argument(
        "--label_column_name",
        type=str,
        help=label_column_name_help,
        default="label",
    )
    init_parser.add_argument(
        "--refresh_media",
        help="TODO",
        action="store_true",
    )
    init_parser.add_argument(
        "--import_stream_metadata",
        help="TODO",
        action="store_true",
    )
    start_labels_help = "Annotation labels"
    init_parser.add_argument(
        "--start_labels", nargs="+", type=str, help=start_labels_help, default=[]
    )
    return data_csv_path_help, label_column_name_help, start_labels_help


def make_train_then_features_parser(subparsers, parents):
    parser = subparsers.add_parser(
        "train_then_features",
        help="Train & then compute features",
    )
    add_parent_actions(parents, parser)
    parser.add_argument(
        "--train_peft",
        type=str,
        help="TODO",
        default="train_peft",
        choices=["train_peft", "train_deep"],
    )
    parser.add_argument("--num_epochs", type=int, help="TODO", default=15)

    return parser


def make_features_parser(subparsers, architectures, clip_variants):
    parser = subparsers.add_parser(
        "features",
        help="Compute features",
    )
    parser.add_argument("folder", type=str, help="Data folder")
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend used for training models.",
        choices=["otx", "naturalis-ai", "annflux"],
        default="annflux",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Neural architecture to use to compute features. NB not all backends support all architectures",
        default=architectures[0],
        choices=architectures,
    )

    parser.add_argument(
        "--model_variant",
        type=str,
        help="TODO",
        default=clip_variants[0],
        choices=clip_variants,
    )

    return parser


def make_go_parser(
    subparsers,
    parents: list[ArgumentParser],
    data_csv_path_help,
    label_column_name_help,
    start_labels_help,
):
    parser = subparsers.add_parser(
        "go",
        help="Init project, compute features, and embed in one go",
    )
    add_parent_actions(parents, parser, ignore_dest=["folder"])
    parser.add_argument("folder", type=str, help="Data folder")
    parser.add_argument(
        "--data_csv_path", type=str, help=data_csv_path_help, default="images.csv"
    )
    parser.add_argument(
        "--label_column_name", type=str, help=label_column_name_help, default="label"
    )
    parser.add_argument(
        "--start_labels", nargs="+", type=str, help=start_labels_help, default=[]
    )
    parser.add_argument(
        "--exclusivity", nargs="+", type=str, help="Exclusivity", default=[]
    )
    parser.add_argument("--feature_cache", type=str, help="TODO", default=None)


def make_data_parser(subparsers, parents: list[ArgumentParser]):
    parser = subparsers.add_parser("data", help="TODO")
    add_parent_actions(parents, parser, ignore_dest=["folder"])
    parser.add_argument("subcommand", type=str, help="TODO", choices=["stream"])
    parser.add_argument("folder", type=str, help="Data folder")
    parser.add_argument("data_input", type=str, help="Folder or CSV/PQ")
    parser.add_argument(
        "--model_package_folder", type=str, help="TODO", default="model_package"
    )

    parser.add_argument(
        "--al_method",
        type=str,
        help="TODO",
        default="uncertainty",
        choices=["uncertainty"],
    )
    parser.add_argument(
        "--percentage_to_add",
        type=float,
        help="TODO",
        default=1.0,  # TODO: min, max
    )
    parser.add_argument("--subsample", type=str, help="TODO", default=None)


def add_parent_actions(parents, parser, ignore_dest=None):
    for parser_ in parents:
        for action_ in parser_._actions:
            if not isinstance(action_, _HelpAction) and (
                ignore_dest is None or action_.dest not in ignore_dest
            ):
                parser._add_action(action_)


if __name__ == "__main__":
    execute()
