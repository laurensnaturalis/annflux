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
import fcntl
import math
import os
import shutil
import tempfile

from PIL import Image
from keras.src.callbacks import Callback
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
    numpy_load,
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
from flask import make_response, render_template, request, send_file, Response, abort
from flask_httpauth import HTTPBasicAuth
# from tensorflow.python.keras.callbacks import Callback
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
#from annflux.training.tensorflow.tf_backend import linear_retraining
# from annflux.training.tensorflow.torch_backend import linear_retraining
from annflux.training.tensorflow.torch_backend import supcon_retraining as linear_retraining

project_root: Optional[str]
images_path: str
thumb_path_: str
failed_images_path: str
group_images_path: str
working_folder: str
exclusivity_path: str
label_definitions_path: str
label_provider_path: str
g_state: AnnFluxState
g_layout: str = "label"
logger: logging.Logger | None = None
training_logger: logging.Logger | None = None


def write_json_atomic(path: str, data: dict, indent: int = 2) -> None:
    """Write JSON file atomically using temp file + rename.
    
    This ensures the file is never in a partially written state,
    preventing corruption if the process crashes during write.
    """
    # Write to temp file in same directory for atomic rename
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(fd)
        # Atomic rename (POSIX guarantees this is atomic)
        os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


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

    global logger, training_logger
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
    g_state.certainty_path = os.path.join(g_state.project_folder, "annflux", "certainty.json")
    label_definitions_path = os.path.join(
        g_state.project_folder, "annflux", "label_defs.json"
    )
    g_state.performance_path = os.path.join(
        g_state.project_folder, "annflux", "performance.json"
    )

    log_level: int = logging.getLevelName(os.getenv("LOGGING_LEVEL", "INFO"))
    logs_dir = os.path.join(g_state.annflux_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    _existing_logs = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
    if _existing_logs:
        import shutil as _shutil
        from datetime import datetime as _dt
        _backup_dir = os.path.join(logs_dir, _dt.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(_backup_dir, exist_ok=True)
        for _f in _existing_logs:
            _shutil.move(os.path.join(logs_dir, _f), os.path.join(_backup_dir, _f))
    log_path = os.path.join(logs_dir, "annflux.log")
    logger = get_logger(log_path, level=log_level, name="annflux_server")
    training_log_path = os.path.join(logs_dir, "training.log")
    training_logger = get_logger(training_log_path, level=log_level, name="annflux_training")
    logging.getLogger("werkzeug").addFilter(NoStatus())
    logger.warning(
        f"Logging to {log_path} with level {logging.getLevelName(log_level)}"
    )
    training_logger.warning(
        f"Training logging to {training_log_path} with level {logging.getLevelName(log_level)}"
    )

    g_layout = "label"
    t = pandas.read_csv(g_state.annflux_path)
    columns = t.columns
    has_records = "record_id" in t and len(t.record_id.unique()) < len(t)
    if "patch_x" in columns: # or has_records:
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

_reclassification_lock = threading.Lock()
_annflux_csv_version = 0


class _AnnfluxCsvFileLock:
    """Exclusive advisory lock on annflux.csv using a .lock sidecar file."""

    def __init__(self, csv_path: str, bump_version: bool = False):
        self._lock_path = csv_path + ".lock"
        self._bump = bump_version
        self._fh = None

    def __enter__(self):
        self._fh = open(self._lock_path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        self._fh = None

    def _bump_version(self):
        if self._bump:
            global _annflux_csv_version
            _annflux_csv_version += 1

auth = HTTPBasicAuth()


# space separated list of entries generated by make_password_entry.py
users = dict([x_.split("|") for x_ in os.getenv("USERS", "").split()])


@auth.verify_password
def verify_password(username, password):
    print("USERS", users)
    if len(users) == 0 or (
        username in users and check_password_hash(users.get(username, ""), password)
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


@app.route("/simple")
@auth.login_required
def simple_annotator_endpoint():
    return render_template("simple_annotator.html")


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

    # Clean up old parquet files before creating new one
    annflux_dir = os.path.join(g_state.project_folder, "annflux")
    for fname in os.listdir(annflux_dir):
        if fname.endswith(".parquet") and fname.startswith("annflux_"):
            old_path = os.path.join(annflux_dir, fname)
            try:
                os.remove(old_path)
            except OSError:
                pass  # Ignore if file is in use or permission issues

    if not os.path.exists(annflux_pq_cache_path):
        if filter_query is not None:
            try:
                t = sql_to_pandas_query(
                    filter_query,
                    pandas.read_csv(
                        annflux_data_path, dtype={"label_predicted": str, "label_true": str}
                    ),
                )
            except ValueError as e:
                raise e
                abort(400, str(e))
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
        os.makedirs(failed_images_path, exist_ok=True)
        failed_thumb_path = os.path.join(failed_images_path, f"{uid}.jpg")
        if not os.path.exists(failed_thumb_path):
            generate_missing_thumbnail(uid).save(failed_thumb_path)
        thumb_path = failed_thumb_path
    else:
        with Image.open(image_path) as img:
            img.thumbnail((256, 256))  # TODO: configurable
            img.convert("RGB").save(thumb_path)

    return send_file(thumb_path, mimetype="image/jpg", as_attachment=False)

@app.route("/data/neighbors/<uid>")
def nearest_neighbors(uid):
    """ """
    state: AnnFluxState = g_state
    import numpy as np
    data = pandas.read_csv(state.annflux_path, dtype={"label_predicted": str, "label_true": str, "image_id": str}) # TODO: slow
    index = np.where(data.image_id == uid)[0][0] # TODO: slow
    print("NEIGHBORS", index, state.all_indices[index])
    data_neighbors = data.iloc[state.all_indices[index]]
    data_neighbors["nn_sort"] = range(len(data_neighbors))
    # data_neighbors = data_neighbors[data_neighbors["labeled"] == 1]

    return Response(data_neighbors.to_csv(index=True), mimetype="text/csv")


_clip_model_cache = None
_clip_processor_cache = None
_clip_device_cache = None
_uid_cache = None
_original_clip_features = None


def _get_clip_text_encoder():
    global _clip_model_cache, _clip_processor_cache, _clip_device_cache
    if _clip_model_cache is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from annflux.repository.model import ClipModel

        repo = Repository(os.path.join(g_state.annflux_folder, "datarepo"))
        clip_model_entry = repo.get(label=ClipModel).last()
        if clip_model_entry is None:
            raise RuntimeError("No CLIP model found in repository")
        config = json.load(open(os.path.join(clip_model_entry.path, "model.json")))
        clip_variant = config["model_variant"]
        _clip_device_cache = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model_cache = CLIPModel.from_pretrained(
            clip_variant, device_map=_clip_device_cache, torch_dtype=torch.float16
        )
        _clip_processor_cache = CLIPProcessor.from_pretrained(clip_variant)
        adapter_folder = os.path.join(clip_model_entry.path, "adapter")
        if os.path.exists(adapter_folder):
            _clip_model_cache.load_adapter(adapter_folder)
        logger.info(f"Loaded CLIP text encoder: {clip_variant}")
    return _clip_model_cache, _clip_processor_cache, _clip_device_cache


def _get_uid_list():
    global _uid_cache
    if _uid_cache is None:
        data = pandas.read_csv(g_state.annflux_path, usecols=["uid"], dtype={"uid": str})
        _uid_cache = data.uid.values
    return _uid_cache


def _get_original_clip_features():
    global _original_clip_features
    if _original_clip_features is None:
        from annflux.repository.resultset import Resultset

        repo = Repository(os.path.join(g_state.annflux_folder, "datarepo"))
        result_set = repo.get(label=Resultset, tag="unseen").first()
        if result_set is None:
            raise RuntimeError("No resultset found in repository")
        folder = result_set.path
        _original_clip_features = numpy_load(f"{folder}/last_full.npz", "lastFull")
        logger.info(f"Loaded original CLIP features from {folder}, shape={_original_clip_features.shape}")
    return _original_clip_features


@app.route("/search/natural_language")
def search_natural_language():
    import torch
    import numpy as np

    query = flask.request.args.get("query", "")
    if not query:
        return flask.jsonify({"uids": [], "scores": [], "error": "Empty query"})

    n = int(flask.request.args.get("n", 50))

    try:
        model, processor, device = _get_clip_text_encoder()
        features = _get_original_clip_features()
    except RuntimeError as e:
        logger.error(f"Failed to initialize natural language search: {e}")
        return flask.jsonify({"uids": [], "scores": [], "error": str(e)}), 500

    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        text_features = model.get_text_features(**inputs)
        text_embedding = text_features.cpu().numpy().flatten().astype(np.float32)

    text_embedding = text_embedding / np.linalg.norm(text_embedding)

    features = features.astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features_normalized = features / norms

    similarities = features_normalized @ text_embedding
    top_indices = np.argsort(similarities)[::-1][:n]

    uids = _get_uid_list()
    result_uids = uids[top_indices].tolist()
    result_scores = similarities[top_indices].tolist()

    logger.info(f"Natural language search: query='{query}', top score={result_scores[0]:.3f}, |uids|={len(result_uids)}")
    return flask.jsonify({"uids": result_uids, "scores": result_scores})


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
        abort(Response("Image not found", status=404))


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
            write_json_atomic(g_state.labels_path, annotations)
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
    write_json_atomic(g_state.labels_path, j_labels)

    async_mode = str2bool(request.args.get("async", "0"))
    if async_mode:
        # update the labeled column in annflux.csv immediately so the data
        # endpoint reflects the new labeled state right away
        _update_labeled_column(set(j_labels.keys()))
        # run reclassification in background; skip if already running
        if _reclassification_lock.locked():
            logger.info("Reclassification already running, skipping")
        else:
            t = threading.Thread(
                target=_background_reclassification,
                args=(is_group,),
                daemon=True,
            )
            t.start()
        return {"success": True, "async": True}

    do_quick_reclassification(is_group)

    return {
        "success": True,
    }


def do_quick_reclassification(is_group: bool, csv_write_lock=None):
    quick_reclassification(g_state, training_logger, "quick", is_group, csv_write_lock=csv_write_lock)
    # if g_state.train_thread is None or not g_state.train_thread.is_alive():
    #     g_state.train_thread = threading.Thread(
    #         target=quick_reclassification, args=(g_state, logger, "quick", is_group)
    #     )
    #     g_state.train_thread.start()
    #     g_state.train_thread.join()


def _background_reclassification(is_group: bool):
    with _reclassification_lock:
        try:
            do_quick_reclassification(is_group, csv_write_lock=_AnnfluxCsvFileLock(g_state.annflux_path, bump_version=True))
        except Exception as e:
            logger.error(f"Background reclassification failed: {e}", exc_info=True)


def _update_labeled_column(labeled_uids: set):
    """Synchronously flip the labeled column in annflux.csv for the given uids."""
    with _AnnfluxCsvFileLock(g_state.annflux_path):
        try:
            data = pandas.read_csv(
                g_state.annflux_path,
                dtype={"label_predicted": str, "label_true": str, "uid": str},
            )
            data["labeled"] = data["uid"].apply(lambda u: int(u in labeled_uids))
            data.to_csv(g_state.annflux_path, index=False)
        except Exception as e:
            logger.error(f"_update_labeled_column failed: {e}", exc_info=True)


@app.route("/annflux_csv_version", methods=["GET"])
def annflux_csv_version():
    return {"version": _annflux_csv_version}


CERTAINTY_VALUES = {"certain", "i_dont_know_the_species", "the_species_cannot_be_known"}


@app.route("/certainty", methods=["POST"])
def certainty():
    updates = request.get_json(force=True)  # {uid: certainty_value, ...}
    invalid = [v for v in updates.values() if v not in CERTAINTY_VALUES]
    if invalid:
        return {"error": f"Invalid certainty values: {invalid}"}, 400
    try:
        if os.path.exists(g_state.certainty_path):
            with open(g_state.certainty_path) as f:
                certainty_map = json.load(f)
        else:
            certainty_map = {}
        certainty_map.update(updates)
        with open(g_state.certainty_path, "w") as f:
            json.dump(certainty_map, f, indent=2)
    except Exception as exc:
        logger.error(f"certainty update failed: {exc}", exc_info=True)
        return {"error": str(exc)}, 500
    return {"success": True}


@app.route("/multilabel_examples", methods=["GET"])
def multilabel_examples():
    labels_param = request.args.get("labels", "")
    n = int(request.args.get("n", 10))
    target_set = frozenset(l.strip() for l in labels_param.split(",") if l.strip())
    data = pandas.read_csv(
        g_state.annflux_path,
        usecols=["uid", "label_true", "labeled"],
        dtype={"uid": str, "label_true": str},
    )
    labeled = data[data["labeled"] == 1].dropna(subset=["label_true"])
    def matches(lt):
        return frozenset(l.strip() for l in lt.split(",") if l.strip()) == target_set
    hits = labeled[labeled["label_true"].apply(matches)]
    return {"uids": hits["uid"].tolist()[:n]}


@app.route("/class_examples/<label>", methods=["GET"])
def class_examples(label: str):
    n = int(request.args.get("n", 10))
    data = pandas.read_csv(
        g_state.annflux_path,
        usecols=["uid", "label_true", "labeled"],
        dtype={"uid": str, "label_true": str},
    )
    labeled = data[data["labeled"] == 1].dropna(subset=["label_true"])
    hits = labeled[labeled["label_true"].str.contains(
        r"(?:^|,)\s*" + label.replace("(", r"\(").replace(")", r"\)") + r"\s*(?:,|$)",
        regex=True,
    )]
    uids = hits["uid"].tolist()[:n]
    return {"uids": uids}


def _visible_metadata_path() -> str:
    return os.path.join(g_state.annflux_folder, "visible_metadata.json")


@app.route("/metadata/visible", methods=["GET"])
def metadata_visible_get():
    columns = list(pandas.read_csv(g_state.annflux_path, nrows=0).columns)
    path = _visible_metadata_path()
    selected = json.load(open(path)) if os.path.exists(path) else []
    return {"columns": columns, "selected": selected}


@app.route("/metadata/visible", methods=["PUT"])
def metadata_visible_put():
    selected = request.get_json(force=True)
    with open(_visible_metadata_path(), "w") as f:
        json.dump(selected, f)
    return {"ok": True}


@app.route("/metadata/visible/page")
@auth.login_required
def metadata_visible_page():
    return render_template("metadata_visible.html")


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
        pandas.DataFrame(
            data={}, columns=["left", "right"] # ty: ignore[invalid-argument-type]
        ).to_csv(
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
    pandas.DataFrame(
        data=request.get_json(),
        columns=("left", "right"),  # ty: ignore[invalid-argument-type]
    ).to_csv(exclusivity_path, index=False)
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
    load_data(state, training_logger, no_linear_features=True)
    weights_path = linear_retraining(state, StatusUpdate(state))
    # shutil.copy(
    #     weights_path,
    #     os.path.join(state.annflux_folder, state.version_for_recompute + ".weights.h5"),
    # )
    #
    repo: Repository = AnnfluxSource(state.project_folder).repository
    # TODO: store linear model
    make_resultset(
        repo.get(label=Dataset).last(),
        state.features,
        repo,
        message=f"linear features from label state={len(state.labeled_indices)}",
    )
    training_logger.info(
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
    state.trained_for_version_previous = len(state.labeled_indices)
    #
    state.g_quick_status = "computing density peak"
    fast_density_peak_clustering(state.project_folder)
    peak_merge(state.project_folder)
    state.g_quick_status = "quicker classification"
    #
    quick_reclassification(state, training_logger)

    training_logger.info(f"retrain_job: done - {state.trained_for_version}")
    state.trained_for_version = len(state.labeled_indices)  # TODO: replace by hash?


def group_train_job(state: AnnFluxState):
    state.g_quick_status = "group training"
    source = AnnfluxSource(state.project_folder)

    if state.features is None or state.labeled_indices is None:
        load_data(state, training_logger)

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
    training_logger.info(f"group_train_job: done - {state.trained_for_version}")
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
                    training_logger.info(
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
    if g_state.is_initialized() and g_state.labeled_indices is not None:
        try:
            estimated_duration_s, duration_std_s = estimate_duration(
                g_state,
                (
                    g_state.g_quick_status,
                    len(g_state.features),
                    len(g_state.labeled_indices),
                ),
                training_logger,
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
    perc_likely_certain_unlabeled = 0
    average_precision = 0
    average_recall = 0
    if os.path.exists(detailed_performance_path):
        try:
            detailed_performance_ = pandas.read_csv(detailed_performance_path)
            num_unlabeled_certain = int(
                detailed_performance_["num_predicted_certain"].sum()
            )
            divisor = (
                detailed_performance_.num_predicted_uncertain.sum()
                + num_unlabeled_certain
            )
            perc_likely_certain = (
                detailed_performance_["num_predicted_certain"].sum() / divisor
                if divisor > 0
                else 0
            )
            # unlabeled-only certain percentage (columns may not exist in old CSV)
            num_certain_unlabeled = detailed_performance_.get("num_predicted_certain_unlabeled", pandas.Series([0])).sum()
            divisor_unlabeled = (
                detailed_performance_.get("num_predicted_uncertain_unlabeled", pandas.Series([0])).sum()
                + num_certain_unlabeled
            )
            perc_likely_certain_unlabeled = (
                num_certain_unlabeled / divisor_unlabeled
                if divisor_unlabeled > 0
                else 0
            )
            average_recall = detailed_performance_.recall.mean()
            average_precision = detailed_performance_.precision.mean()
        except EmptyDataError:
            pass

    # Load label_accuracy from performance.json
    label_accuracy = 0
    if os.path.exists(g_state.performance_path):
        try:
            perf_data = json.load(open(g_state.performance_path))
            label_accuracy = perf_data.get("label_accuracy", 0)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    time_remaining_s = estimated_duration_s - status_duration
    return {
        "status": g_state.g_quick_status,
        "linear_status_epoch": g_state.linear_status_epoch,
        "trained_for_version": g_state.trained_for_version,
        "num_unlabeled_certain": num_unlabeled_certain,
        "num_total": len(g_state.features) if g_state.is_initialized() else "?",
        "time_remaining_s": f"{time_remaining_s:.2f}",
        "duration_std_s": f"{duration_std_s:.2f}",
        "duration_s": f"{estimated_duration_s:.2f}",
        "duration_perc": f"{time_remaining_s / estimated_duration_s if estimated_duration_s > 0 else 0:.2f}",
        "package_version": g_version,
        "num_labeled": len(g_state.labeled_indices),
        "perc_likely_certain": perc_likely_certain,
        "perc_likely_certain_unlabeled": perc_likely_certain_unlabeled,
        "average_precision": average_precision,
        "average_recall": average_recall,
        "label_accuracy": label_accuracy,
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
    if os.getenv("PROJECT_ROOT") is None:
        os.environ["PROJECT_ROOT"] = os.path.expanduser(sys.argv[1])

    _init()
    app.run(
        debug=str2bool(os.getenv("APP_DEBUG", False)),
        host="0.0.0.0",
        threaded=True,
        use_reloader=False,
        port=int(os.getenv("PORT", "8006")),
    )


if __name__ == "__main__":
    ui_script_entry()
