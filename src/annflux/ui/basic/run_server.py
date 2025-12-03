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
import math
import os
import shutil
import tempfile

from PIL import Image
from pandas.errors import EmptyDataError

from annflux.repo_results_to_embedding import group_embedding
from annflux.repository.dataset import Dataset
from annflux.repository.repository import Repository
from annflux.repository.resultset import Resultset
from annflux.shared import AnnfluxSource
from annflux.tools.io import (
    generate_missing_thumbnail,
    to_js_arrow,
    compute_hash,
    sql_to_pandas_query,
)
from annflux.tools.progress_learn import estimate_duration
from annflux.tools.visualization import most_contrasting_gray, brighten_hex_color
from annflux.training.annflux.feature_extractor import make_resultset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from functools import update_wrapper, wraps
from typing import Dict, List, Optional, Tuple

import flask
import pandas
from flask import make_response, render_template, request, send_file, Response
from flask_httpauth import HTTPBasicAuth
from tensorflow.python.keras.callbacks import Callback
from werkzeug.security import check_password_hash

from annflux.algorithms.embeddings import compute_tsne
from annflux.algorithms.fastdpeak import fast_density_peak_clustering
from annflux.algorithms.fastdpeak_merge import peak_merge
from annflux.tools.core import AnnFluxState
from annflux.tools.data import (
    add_group_to_exclusivity,
    get_images_path,
    remove_uids_from_double_check,
    get_group_images_path,
    get_failed_images_path,
    get_thumb_path,
    create_group_flux_data,
    make_backup,
)
from annflux.tools.mixed import get_logger, str2bool, get_version
from annflux.tools.io import file_hash
from annflux.training.annflux.quick import (
    quick_reclassification,
    group_classification,
    load_data,
)
from annflux.training.tensorflow.tf_backend import linear_retraining
# from annflux.training.tensorflow.torch_backend import linear_retraining

project_root: Optional[str] = None
images_path: Optional[str] = None
thumb_path_: Optional[str] = None
failed_images_path: Optional[str] = None
group_images_path: Optional[str] = None
working_folder: Optional[str] = None
exclusivity_path: Optional[str] = None
label_definitions_path: Optional[str] = None
label_provider_path: Optional[str] = None
g_state: Optional[AnnFluxState] = None
g_layout: Optional[str] = None
logger: Optional[logging.Logger] = None


class NoStatus(logging.Filter):
    def filter(self, record):
        return "POST /status" not in record.getMessage()


def _init():
    global \
        project_root, \
        images_path, \
        working_folder, \
        g_state, \
        exclusivity_path, \
        label_provider_path, \
        label_definitions_path, \
        g_layout, \
        group_images_path, \
        failed_images_path, \
        thumb_path_

    global logger
    project_root = os.getenv("PROJECT_ROOT", None)
    if project_root is None:
        raise RuntimeError("You should set PROJECT_ROOT environment variable")
    else:
        images_path = get_images_path()
        thumb_path_ = get_thumb_path()
        group_images_path = get_group_images_path()
        failed_images_path = get_failed_images_path()
        working_folder = os.path.join(project_root, "annflux")

    g_state = AnnFluxState(working_folder)

    g_state.doublecheck_path = os.path.join(
        g_state.project_folder, "annflux", "doublecheck.json"
    )
    exclusivity_path = os.path.join(
        g_state.project_folder, "annflux", "exclusivity.csv"
    )
    label_provider_path = os.path.join(
        g_state.project_folder, "annflux", "label_provider.csv"
    )
    g_state.labels_path = os.path.join(g_state.project_folder, "annflux", "labels.json")
    label_definitions_path = os.path.join(
        g_state.project_folder, "annflux", "label_defs.json"
    )
    g_state.performance_path = os.path.join(
        g_state.project_folder, "annflux", "performance.json"
    )

    print(os.getenv("LOGGING_LEVEL", "INFO"))
    log_level: int = logging.getLevelName(os.getenv("LOGGING_LEVEL", "INFO"))
    log_path = os.path.join(g_state.annflux_folder, "annflux.log")
    os.makedirs(g_state.annflux_folder, exist_ok=True)
    logger = get_logger(log_path, level=log_level, name="annflux_server")
    logging.getLogger("werkzeug").addFilter(NoStatus())
    logger.warning(
        f"Logging to {log_path} with level {logging.getLevelName(log_level)}"
    )

    g_layout = "label"
    t = pandas.read_csv(g_state.annflux_path)
    columns = t.columns
    has_records = "record_id" in t and len(t.record_id.unique()) < len(t)
    if "patch_x" in columns or has_records:
        g_layout = "tileLabel"
    else:
        g_layout = "originalImageLabel"  # TODO(ENV)

    print(f"Using project_root={project_root}, images_path ={images_path}")


app = flask.Flask(
    __name__,
    static_url_path=os.getenv("STATIC_URL", "/static"),
)
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app, profile_dir="prof") #, restrictions="^(?!quick\.py$).*$")


def get_app():
    return app


knn_type = "quick"
dump_linear_features = False
optimize_weight_exponent = False

num_unlabeled_certain = None

auth = HTTPBasicAuth()

# space separated list of entries generated by make_password_entry.py
users = dict([x_.split("|") for x_ in os.getenv("USERS", "").split()])


@auth.verify_password
def verify_password(username, password):
    if len(users) == 0 or (
        username in users and check_password_hash(users.get(username), password)
    ):
        return username
    return None


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Last-Modified"] = datetime.now()
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "-1"
        return response

    return update_wrapper(no_cache, view)


@app.route("/annflux")
@app.route("/")
@auth.login_required
def annflux_endpoint():
    """ """
    return render_template(
        {"annflux": "html", "golden": "annflux_golden.html"}[
            os.getenv("INDEED_TEMPLATE", "golden")
        ],
        layout=g_layout,
        auto_linear_train_idle_time=int(os.getenv("AUTO_LINEAR_TRAIN_IDLE_TIME", 1800)),
    )


@app.route("/data")
@app.route("/v1/data")
@nocache
def data_get():
    """
    http://127.0.0.1:8006/data?filter_query=Papi%20in%20row.label_predicted
    blurry-or-low-res NOT IN row.label_possible AND blurry-or-low-res NOT IN row.label_predicted
    """
    annflux_data_path = os.path.join(g_state.project_folder, "annflux", "annflux.csv")
    time_start = time.time()
    filter_query = flask.request.args.get("filter_query")
    print(f"data_get: {filter_query=}")
    hash_ = file_hash(annflux_data_path)
    if filter_query is not None:
        hash_ += compute_hash(filter_query)
    print(f"{annflux_data_path} took {(time.time() - time_start) * 1000} ms")
    annflux_pq_cache_path = os.path.join(
        g_state.project_folder, "annflux", f"annflux_{hash_}.parquet"
    )

    if not os.path.exists(annflux_pq_cache_path):
        if filter_query is not None:
            t = sql_to_pandas_query(
                filter_query,
                pandas.read_csv(
                    annflux_data_path, dtype={"label_predicted": str, "label_true": str}
                ),
            )
            print(f"{len(t)=}")
            with tempfile.NamedTemporaryFile() as fn:
                t.to_csv(fn, index=False)
                annflux_data_path = fn.name

                to_js_arrow(annflux_data_path, annflux_pq_cache_path)
        else:
            to_js_arrow(annflux_data_path, annflux_pq_cache_path)

    logger.info(f"annflux_data_path = {annflux_data_path}")
    return send_file(
        annflux_pq_cache_path,
        mimetype="application/x-binary",
        as_attachment=False,
    )


@app.route("/data/group")
@nocache
def data_group_get():
    """
    TODO
    """
    group_data_path = os.path.join(
        g_state.project_folder, "annflux", "group0_annflux.csv"
    )
    logger.info(f"annflux_data_path = {group_data_path}")
    return (
        send_file(
            group_data_path,
            mimetype="text_csv",
            as_attachment=False,
        )
        if os.path.exists(group_data_path)
        else {}
    )


group_uids = set()


def get_group_uids() -> set[str]:
    global group_uids
    group_data_path = os.path.join(
        g_state.project_folder, "annflux", "group0_annflux.csv"
    )

    if os.path.exists(group_data_path):
        group_uids = set(pandas.read_csv(group_data_path)["uid"])
    return group_uids


@app.route("/images/thumbnail/<uid>")
def thumbnail(uid):
    """ """
    if uid in get_group_uids():
        image_path = os.path.join(group_images_path, f"{uid}.jpg")
    else:
        image_path = os.path.join(images_path, f"{uid}.jpg")
    # TODO: this puts both instance and group thumbs in one folder
    os.makedirs(thumb_path_, exist_ok=True)
    thumb_path = os.path.join(thumb_path_, f"{uid}.jpg")
    if not os.path.exists(image_path):
        failed_thumb_path = os.path.join(failed_images_path, f"{uid}.jpg")
        if not os.path.exists(failed_thumb_path):
            generate_missing_thumbnail(uid).save(failed_thumb_path)
        thumb_path = failed_thumb_path
    else:
        with Image.open(image_path) as img:
            img.thumbnail((256, 256))  # TODO: configurable
            img.convert("RGB").save(thumb_path)

    return send_file(thumb_path, mimetype="image/jpg", as_attachment=False)


@app.route("/images/original/thumbnail/<uid>")
def thumbnail_original(uid):
    """ """
    thumb_path = os.path.join(images_path.replace("images", "original"), f"{uid}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="image/jpg", as_attachment=False)
    return None


@app.route("/images/mask/thumbnail/<uid>")
def thumbnail_mask(uid):
    """ """
    thumb_path = os.path.join(images_path.replace("images", "mask"), f"{uid}.png")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="image/png", as_attachment=False)
    return None


@app.route("/images/full/<uid>")
def images_full(uid):
    """ """
    file_path = os.path.join(images_path, f"{uid}.jpg")
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/jpg", as_attachment=False)
    else:
        print(f"{file_path} not found")


@app.route("/sounds/<uid>")
def sound(uid):
    """
    Identification from image(s) (with authentication)
    :return: json server response
    """
    thumb_path = os.path.join(images_path.replace("images", "wav"), f"{uid}.wav")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="audio/wav", as_attachment=False)


class StatusUpdate(Callback):
    def __init__(self, state: AnnFluxState):
        super().__init__()
        self.state = state

    def __call__(self, epoch, logs=None):
        self.state.linear_status_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.state.linear_status_epoch = epoch

@app.route("/v1/label_definitions/sort", methods=["GET"])
def label_defs_sort():
    if not os.path.exists(label_definitions_path):
        label_definitions = {"labels": []}
    else:
        label_definitions = json.load(open(label_definitions_path))
    #
    class_to_color = pandas.read_csv(
        os.path.join(g_state.annflux_folder, "class_to_color.csv")
    )
    class_to_count = dict(zip(class_to_color["class"], class_to_color["count"]))
    label_definitions["labels"] = sorted(
        label_definitions["labels"],
        key=lambda t: class_to_count.get(t[0], 0),
        reverse=True,
    )
    make_backup(
        label_definitions_path,
        backup_dir=os.path.join(os.path.dirname(label_definitions_path), "backups"),
    )
    with open(label_definitions_path, "w") as f:
        json.dump(label_definitions, f, indent=2)
    return {"result": "ok"}


@app.route("/label_defs", methods=["PUT"])
@app.route("/v1/label_definitions", methods=["PUT"])
def label_defs_add():
    """
    Stores label in label definitions file if it does not already exist

    Expects as input a tuple of the form [new_label, parent_label] or [new_label, parent_label, exclusive_under_parent]

    If exclusive_under_parent is True all the children of parent_label will be made mutually exclusive
    @return:
    """
    label_def = request.get_json(force=True)
    logger.info(f"label_def={label_def}")
    label_definitions: Dict[str, List[Tuple[str, str]]]  # Tuple = (label, parent)
    if not os.path.exists(label_definitions_path):
        label_definitions = {"labels": []}
    else:
        label_definitions = json.load(open(label_definitions_path))
    if label_def not in label_definitions["labels"]:
        label_definitions["labels"].append(label_def)
        # -- undetermined labels
        if os.path.exists(g_state.labels_path):
            modified_uids = []
            annotations = json.load(open(g_state.labels_path))
            exclusive_under_parent = False
            if len(label_def) == 2:
                new_label, parent_label = label_def
            elif len(label_def) == 3:
                new_label, parent_label, exclusive_under_parent = label_def
            else:
                raise RuntimeError()
            has_parent = parent_label not in ["null", "", None]
            # - assign undetermined state appropriate images
            for image_id, annotation in annotations.items():
                labels = annotation.split(",")
                if has_parent:
                    if parent_label in labels:
                        annotation += f",{new_label}=?"
                        annotations[image_id] = annotation
                        modified_uids.append(image_id)
                else:
                    annotation += f",{new_label}=?"
                    annotations[image_id] = annotation
                    modified_uids.append(image_id)
            json.dump(annotations, open(g_state.labels_path, "w"), indent=2)
            # remove uids from doublecheck list so that they appear for the viewer to check
            remove_uids_from_double_check(modified_uids, g_state.doublecheck_path)
            # exclusive_under_parent
            if exclusive_under_parent and has_parent:
                group_children = [
                    t_[0] for t_ in label_definitions["labels"] if t_[1] == parent_label
                ]
                add_group_to_exclusivity(group_children, exclusivity_path)

        # add new label to label definitions
        with open(label_definitions_path, "w") as f:
            json.dump(label_definitions, f)
    return {"result": "ok"}


@app.route("/version")
def version_endpoint():
    return g_version


g_version = get_version()


@app.route("/label", methods=["POST"])
def label():
    g_state.new_labeled_uids = set()
    if os.path.exists(g_state.labels_path):
        with open(g_state.labels_path, "r") as f:
            j_labels = json.load(f)
    else:
        j_labels = {}
    label_update = request.get_json(force=True)
    logger.info(f"label_update={label_update}")
    # double check
    if os.path.exists(g_state.doublecheck_path):
        with open(g_state.doublecheck_path, "r") as f:
            j_doublecheck = json.load(f)
    else:
        j_doublecheck = {"checked": []}
    is_group = False
    for uid in label_update:
        # is_group |= uid.startswith("R")  # TODO
        if uid in j_labels:
            print(f"Adding {uid} to double check")
            j_doublecheck["checked"].append(uid)
        g_state.new_labeled_uids.add(uid)
    with open(g_state.doublecheck_path, "w") as f:
        json.dump(j_doublecheck, f, indent=2)  # noqa
    #
    # TODO: check that not incidentally undetermined labels are removed
    j_labels.update({k: v for k, v in label_update.items() if v != "n/a" and v != ""})
    with open(g_state.labels_path, "w") as f:
        json.dump(j_labels, f, indent=2)

    do_quick_reclassification(is_group)

    return {
        "success": True,
    }


def do_quick_reclassification(is_group: bool):
    if g_state.train_thread is None or not g_state.train_thread.is_alive():
        g_state.train_thread = threading.Thread(
            target=quick_reclassification, args=(g_state, logger, "quick", is_group)
        )
        g_state.train_thread.start()
        g_state.train_thread.join()


@app.route("/performance")
@app.route("/v1/performance")
def performance():
    return (
        json.load(open(g_state.performance_path))
        if os.path.exists(g_state.performance_path)
        else {}
    )


@app.route("/detailed_performance/data")
def detailed_performance_data():
    return send_file(
        os.path.join(g_state.annflux_folder, "detailed_performance.csv"),
        mimetype="text_csv",
        as_attachment=False,
    )


@app.route("/exclusivity/data")
def exclusivity_data():
    if not os.path.exists(exclusivity_path):
        pandas.DataFrame(data={}, columns=["left", "right"]).to_csv(
            exclusivity_path, index=False
        )

    return send_file(exclusivity_path, mimetype="text_csv", as_attachment=False)


@app.route("/label_provider/data")
def label_provider_data():
    return send_file(label_provider_path, mimetype="text_csv", as_attachment=False)


@app.route("/labels/css")
def labels_css():
    label_to_color = pandas.read_csv(
        os.path.join(g_state.annflux_folder, "class_to_color.csv")
    )
    css_str = []
    count_max = label_to_color["count"].max()
    for _, row in label_to_color.iterrows():
        if not pandas.isna(row["class"]) and "," not in row["class"]:
            background_color = brighten_hex_color(row.color)
            font_color = most_contrasting_gray(row.color)
            css_str.append(
                f".label_{row['class'].replace(' ', '_')} {{ background-color: {background_color}; "
                f"border-color:{row.color}; border-width:2px; color: {font_color}; font-size: {max(1.0, 2 * math.sqrt(row['count'] / count_max))}em }}"
            )

    return Response("\n".join(css_str), mimetype="text/css")


@app.route("/exclusivity/data", methods=["POST"])
def exclusivity_data_post():
    pandas.DataFrame(data=request.get_json(), columns=["left", "right"]).to_csv(
        exclusivity_path, index=False
    )
    return {"success": True}


@app.route("/exclusivity")
def exclusivity_ui():
    return render_template("exclusivity.html")


@app.route("/class_examples")
def class_examples_ui():
    return render_template("class_examples.html")


@app.route("/label_provider")
def label_provider_ui():
    return render_template("label_provider.html")


@app.route("/label_defs")
def label_defs_list():
    return json.load(open(label_definitions_path))


@app.route("/detailed_performance")
@app.route("/ui/detailed_performance")
def detailed_performance():
    return render_template("detailed_performance.html")


def retrain_job(state: AnnFluxState):
    state.g_quick_status = "training"
    load_data(state, logger, no_linear_features=True)
    weights_path = linear_retraining(state, StatusUpdate(state))
    shutil.copy(
        weights_path,
        os.path.join(state.annflux_folder, state.version_for_recompute + ".weights.h5"),
    )
    #
    repo: Repository = AnnfluxSource(state.project_folder).repository
    # TODO: store linear model
    make_resultset(
        repo.get(label=Dataset).first(),
        state.features,
        repo,
        message=f"linear features from label state={len(state.labeled_indices)}",
    )
    logger.info(
        f"Stored Resultset for linear trained features in {repo.get(label=Resultset).first()}"
    )
    #
    state.g_quick_status = "computing embedding"
    embedding = compute_tsne(state.features)
    state.g_quick_status = "computing embedding done"
    data = pandas.read_csv(
        state.annflux_path,
        dtype={"label_predicted": str, "score_true": float},
    )
    data["e_0"] = embedding[:, 0]
    data["e_1"] = embedding[:, 1]
    data.to_csv(state.annflux_path, index=False)
    state.trained_for_version_pre = len(state.labeled_indices)
    #
    state.g_quick_status = "computing density peak"
    fast_density_peak_clustering(state.project_folder)
    peak_merge(state.project_folder)
    state.g_quick_status = "quicker classification"
    #
    quick_reclassification(state, logger)

    logger.info(f"retrain_job: done - {state.trained_for_version}")
    state.trained_for_version = len(state.labeled_indices)  # TODO: replace by hash?


def group_train_job(state: AnnFluxState):
    state.g_quick_status = "group training"
    source = AnnfluxSource(state.project_folder)

    if g_state.features is None:
        load_data(state, logger)

    create_group_flux_data(source)

    split_path = os.path.join(state.annflux_folder, "split_group.json")
    group_data = pandas.read_csv(source.group_flux_data_path())
    import numpy as np

    if not os.path.exists(split_path):
        test_uids = np.random.choice(
            group_data.uid.values, int(0.10 * len(group_data)), replace=False
        ).tolist()
        with open(split_path, "w") as f:
            json.dump({"test": test_uids}, f)
    else:
        test_uids = json.load(open(split_path))["test"]

    record_features, accuracy_group, record_table = group_classification(
        g_state.features,
        pandas.read_csv(g_state.annflux_path),
        os.path.join(state.annflux_folder, "group_feature_images"),
        group_data,
        test_uids,
    )
    print(record_features.shape, accuracy_group, len(record_table))
    record_table.to_csv(
        os.path.join(g_state.annflux_folder, "group0_annflux.csv"), index=False
    )
    import numpy as np

    np.savez(
        os.path.join(g_state.annflux_folder, "group0_features.npz"),
        lastFull=record_features,
    )
    state.g_quick_status = "group embedding"
    group_embedding(g_state.project_folder)
    logger.info(f"group_train_job: done - {state.trained_for_version}")
    state.trained_for_version = len(state.labeled_indices)  # TODO(crit): for group


@app.route("/status", methods=["POST"])
def status():
    label_update = request.get_json(force=True)
    # print(f"{label_update=}")
    auto_linear_train_idle_time = int(os.getenv("AUTO_LINEAR_TRAIN_IDLE_TIME", 1800))
    # print(label_update["idleTime"], auto_linear_train_idle_time, g_state.labeled_indices)
    group_train = label_update["groupTrain"] if "groupTrain" in label_update else False
    if label_update["idleTime"] > auto_linear_train_idle_time:
        if g_state.train_thread is None or not g_state.train_thread.is_alive():
            if g_state.labeled_indices is not None:
                if g_state.trained_for_version != len(g_state.labeled_indices):
                    logger.info(
                        f"Training from status: {g_state.trained_for_version=}"
                        f", {len(g_state.labeled_indices)=}, {label_update['idleTime']}"
                    )
                    g_state.train_thread = threading.Thread(
                        target=retrain_job, args=(g_state,)
                    )
                    g_state.train_thread.start()
    #
    if group_train:
        if g_state.train_thread is None or not g_state.train_thread.is_alive():
            # TODO(opt): consider multiple train threads
            g_state.train_thread = threading.Thread(
                target=group_train_job, args=(g_state,)
            )
            g_state.train_thread.start()
    #
    estimated_duration_s = 0
    duration_std_s = 0
    if g_state.features is not None and g_state.labeled_indices is not None:
        try:
            estimated_duration_s, duration_std_s = estimate_duration(
                g_state,
                (
                    g_state.g_quick_status,
                    len(g_state.features),
                    len(g_state.labeled_indices),
                ),
                logger,
            )
        except ValueError:
            estimated_duration_s = 0
            duration_std_s = 0
    #
    status_duration = (
        time.time() - g_state.time_new_status_time
        if g_state.time_new_status_time is not None
        else 0
    )
    detailed_performance_path = os.path.join(
        g_state.annflux_folder, "detailed_performance.csv"
    )
    num_unlabeled_certain = 0
    perc_likely_certain = 0
    average_precision = 0
    average_recall = 0
    if os.path.exists(detailed_performance_path):
        try:
            detailed_performance_ = pandas.read_csv(detailed_performance_path)
        except EmptyDataError:
            detailed_performance_ = []
        if len(detailed_performance_) > 0:
            num_unlabeled_certain = int(
                detailed_performance_.num_predicted_certain.sum()
            )
            perc_likely_certain = detailed_performance_.num_predicted_certain.sum() / (
                detailed_performance_.num_predicted_uncertain.sum()
                + num_unlabeled_certain
            )
            average_recall = detailed_performance_.recall.mean()
            average_precision = detailed_performance_.precision.mean()

    time_remaining_s = estimated_duration_s - status_duration
    return {
        "status": g_state.g_quick_status,
        "linear_status_epoch": g_state.linear_status_epoch,
        "trained_for_version": g_state.trained_for_version,
        "num_unlabeled_certain": num_unlabeled_certain,
        "num_total": len(g_state.features) if g_state.features is not None else "?",
        "time_remaining_s": f"{time_remaining_s:.2f}",
        "duration_std_s": f"{duration_std_s:.2f}",
        "duration_s": f"{estimated_duration_s:.2f}",
        "duration_perc": f"{time_remaining_s / estimated_duration_s if estimated_duration_s > 0 else 0:.2f}",
        "package_version": g_version,
        "num_labeled": len(g_state.labeled_indices) if g_state.labeled_indices else 0,
        "perc_likely_certain": perc_likely_certain,
        "average_precision": average_precision,
        "average_recall": average_recall,
        # "performance": json.load(open(performance_path))
    }


def standard_json_response(error_code, error_message, http_status_code):
    """

    :param error_code: str
    :param error_message: str
    :param http_status_code: str
    :return: json response with compiled message
    """
    if error_message is None:
        error_message = " ".join(error_code.split("_")).capitalize()
    logger.info("status: {}, error message: {}".format(http_status_code, error_message))
    response = make_response(
        flask.jsonify({"error": {"code": error_code, "message": error_message}}),
        http_status_code,
    )
    return response


@app.errorhandler(500)
def general_server_error(e):
    """
    General server error response
    :param e: error
    :return: json response
    """
    logger.debug(e)
    return standard_json_response("general_server_error", None, 500)


def ui_script_entry():
    os.environ["PROJECT_ROOT"] = (
        os.path.expanduser(sys.argv[1])
        if os.getenv("PROJECT_ROOT") is None
        else os.getenv("PROJECT_ROOT")
    )
    _init()
    app.run(
        debug=str2bool(os.getenv("APP_DEBUG", False)),
        host="0.0.0.0",
        threaded=True,
        port=int(os.getenv("PORT", "8006")),
    )


if __name__ == "__main__":
    ui_script_entry()
