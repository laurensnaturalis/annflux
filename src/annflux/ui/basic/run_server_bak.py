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
import time
import json
import logging
import sys
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image
from pandas.errors import EmptyDataError

import flask
import pandas
from flask import make_response, render_template, request, send_file, Response

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
from annflux.tools.core import AnnFluxState
from annflux.tools.data import (
    add_group_to_exclusivity,
    remove_uids_from_double_check,
    make_backup,
)
from annflux.tools.mixed import get_logger, str2bool, get_version
from annflux.tools.io import file_hash
from annflux.training.annflux.feature_extractor import make_resultset

# Import refactored modules
from annflux.ui.basic.server_auth import auth
from annflux.ui.basic.server_decorators import nocache
from annflux.ui.basic.server_utils import normalize_filter_query, standard_json_response, get_group_uids
from annflux.ui.basic.server_jobs import retrain_job, group_train_job, do_quick_reclassification
from annflux.ui.basic.server_init import _init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Global variables (will be initialized by _init)
project_root: Optional[str] = None
images_path: str = ""
thumb_path_: str = ""
failed_images_path: str = ""
group_images_path: str = ""
working_folder: str = ""
exclusivity_path: str = ""
label_definitions_path: str = ""
label_provider_path: str = ""
g_state: AnnFluxState = None
g_layout: str = ""
logger: logging.Logger = None

app = flask.Flask(
    __name__,
    static_url_path=os.getenv("STATIC_URL", "/static"),
)


def get_app():
    """Get the Flask app instance."""
    global project_root, images_path, thumb_path_, failed_images_path, group_images_path
    global working_folder, exclusivity_path, label_provider_path, label_definitions_path
    global g_state, g_layout, logger
    
    # Initialize global variables if not already done
    if g_state is None:
        try:
            _init()
            # After _init(), update the global variables in this module
            from annflux.ui.basic.server_init import _init_values
            
            project_root = _init_values['project_root']
            images_path = _init_values['images_path']
            thumb_path_ = _init_values['thumb_path_']
            failed_images_path = _init_values['failed_images_path']
            group_images_path = _init_values['group_images_path']
            working_folder = _init_values['working_folder']
            exclusivity_path = _init_values['exclusivity_path']
            label_provider_path = _init_values['label_provider_path']
            label_definitions_path = _init_values['label_definitions_path']
            g_state = _init_values['g_state']
            g_layout = _init_values['g_layout']
            logger = _init_values['logger']
        except RuntimeError as e:
            # If PROJECT_ROOT is not set, it's likely a test environment
            # In this case, we'll set up minimal initialization
            if "PROJECT_ROOT" not in os.environ:
                # For testing, we'll skip full initialization
                # The test fixtures will handle proper setup
                pass
            else:
                raise e
    return app


knn_type = "quick"
dump_linear_features = False
optimize_weight_exponent = False

num_unlabeled_certain = None

g_version = get_version()


@app.route("/annflux")
@app.route("/")
@auth.login_required
def annflux_endpoint():
    """Main UI endpoint."""
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
    Data endpoint with filtering support.
    
    Supports both simple format: label_predicted = 'label1'
    and pseudo-SQL format: row.label_predicted = 'label1'
    """
    annflux_data_path = os.path.join(g_state.project_folder, "annflux", "annflux.csv")
    time_start = time.time()
    filter_query = flask.request.args.get("filter_query")
    print(f"data_get: {filter_query=}")
    
    # Normalize the filter query
    if filter_query is not None:
        filter_query = normalize_filter_query(filter_query)
        print(f"data_get: normalized {filter_query=}")
    
    hash_ = file_hash(annflux_data_path)
    if filter_query is not None:
        hash_ += compute_hash(filter_query)
    print(f"{annflux_data_path} took {(time.time() - time_start) * 1000} ms")
    annflux_pq_cache_path = os.path.join(
        g_state.project_folder, "annflux", f"annflux_{hash_}.parquet"
    )

    if not os.path.exists(annflux_pq_cache_path):
        if filter_query is not None:
            try:
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
            except ValueError as e:
                logger.error(f"Failed to parse filter query: {filter_query}, error: {e}")
                return standard_json_response("invalid_filter_query", str(e), 400)
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
    """Group data endpoint."""
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


@app.route("/images/thumbnail/<uid>")
def thumbnail(uid):
    """Thumbnail endpoint for images."""
    if uid in get_group_uids(g_state, group_images_path):
        image_path = os.path.join(group_images_path, f"{uid}.jpg")
    else:
        image_path = os.path.join(images_path, f"{uid}.jpg")
    
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
    """Original thumbnail endpoint."""
    thumb_path = os.path.join(images_path.replace("images", "original"), f"{uid}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="image/jpg", as_attachment=False)
    return None


@app.route("/images/mask/thumbnail/<uid>")
def thumbnail_mask(uid):
    """Mask thumbnail endpoint."""
    thumb_path = os.path.join(images_path.replace("images", "mask"), f"{uid}.png")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="image/png", as_attachment=False)
    return None


@app.route("/images/full/<uid>")
def images_full(uid):
    """Full image endpoint."""
    file_path = os.path.join(images_path, f"{uid}.jpg")
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/jpg", as_attachment=False)
    else:
        print(f"{file_path} not found")


@app.route("/sounds/<uid>")
def sound(uid):
    """Sound endpoint for audio files."""
    thumb_path = os.path.join(images_path.replace("images", "wav"), f"{uid}.wav")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype="audio/wav", as_attachment=False)


@app.route("/v1/label_definitions/sort", methods=["GET"])
def label_defs_sort():
    """Sort label definitions by count."""
    if not os.path.exists(label_definitions_path):
        label_definitions = {"labels": []}
    else:
        label_definitions = json.load(open(label_definitions_path))
    
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
    Add new label definition.
    
    Expects input: [new_label, parent_label] or [new_label, parent_label, exclusive_under_parent]
    """
    label_def = request.get_json(force=True)
    logger.info(f"label_def={label_def}")
    label_definitions: Dict[str, List[Tuple[str, str]]]
    if not os.path.exists(label_definitions_path):
        label_definitions = {"labels": []}
    else:
        label_definitions = json.load(open(label_definitions_path))
    
    if label_def not in label_definitions["labels"]:
        label_definitions["labels"].append(label_def)
        # Handle undetermined labels
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
            
            # Assign undetermined state to appropriate images
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
            remove_uids_from_double_check(modified_uids, g_state.doublecheck_path)
            
            # Handle exclusivity
            if exclusive_under_parent and has_parent:
                group_children = [
                    t_[0] for t_ in label_definitions["labels"] if t_[1] == parent_label
                ]
                add_group_to_exclusivity(group_children, exclusivity_path)

        # Save label definitions
        with open(label_definitions_path, "w") as f:
            json.dump(label_definitions, f)
    return {"result": "ok"}


@app.route("/version")
def version_endpoint():
    """Version endpoint."""
    return g_version


@app.route("/label", methods=["POST"])
def label():
    """Label submission endpoint."""
    g_state.new_labeled_uids = set()
    if os.path.exists(g_state.labels_path):
        with open(g_state.labels_path, "r") as f:
            j_labels = json.load(f)
    else:
        j_labels = {}
    
    label_update = request.get_json(force=True)
    logger.info(f"label_update={label_update}")
    
    # Handle double check
    if os.path.exists(g_state.doublecheck_path):
        with open(g_state.doublecheck_path, "r") as f:
            j_doublecheck = json.load(f)
    else:
        j_doublecheck = {"checked": []}
    
    is_group = False
    for uid in label_update:
        if uid in j_labels:
            print(f"Adding {uid} to double check")
            j_doublecheck["checked"].append(uid)
        g_state.new_labeled_uids.add(uid)
    
    with open(g_state.doublecheck_path, "w") as f:
        json.dump(j_doublecheck, f, indent=2)
    
    # Update labels
    j_labels.update({k: v for k, v in label_update.items() if v != "n/a" and v != ""})
    with open(g_state.labels_path, "w") as f:
        json.dump(j_labels, f, indent=2)

    do_quick_reclassification(g_state, is_group)

    return {
        "success": True,
    }


@app.route("/performance")
@app.route("/v1/performance")
def performance():
    """Performance metrics endpoint."""
    return (
        json.load(open(g_state.performance_path))
        if os.path.exists(g_state.performance_path)
        else {}
    )


@app.route("/detailed_performance/data")
def detailed_performance_data():
    """Detailed performance data endpoint."""
    return send_file(
        os.path.join(g_state.annflux_folder, "detailed_performance.csv"),
        mimetype="text_csv",
        as_attachment=False,
    )


@app.route("/exclusivity/data")
def exclusivity_data():
    """Exclusivity data endpoint."""
    if not os.path.exists(exclusivity_path):
        pandas.DataFrame(
            data={}, columns=["left", "right"] # ty: ignore[invalid-argument-type]
        ).to_csv(
            exclusivity_path, index=False
        )

    return send_file(exclusivity_path, mimetype="text_csv", as_attachment=False)


@app.route("/label_provider/data")
def label_provider_data():
    """Label provider data endpoint."""
    return send_file(label_provider_path, mimetype="text_csv", as_attachment=False)


@app.route("/labels/css")
def labels_css():
    """CSS for labels endpoint."""
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
    """Post exclusivity data endpoint."""
    pandas.DataFrame(
        data=request.get_json(), columns=("left", "right") # ty: ignore[invalid-argument-type]
    ).to_csv(
        exclusivity_path, index=False
    )
    return {"success": True}


@app.route("/exclusivity")
def exclusivity_ui():
    """Exclusivity UI endpoint."""
    return render_template("exclusivity.html")


@app.route("/class_examples")
def class_examples_ui():
    """Class examples UI endpoint."""
    return render_template("class_examples.html")


@app.route("/label_provider")
def label_provider_ui():
    """Label provider UI endpoint."""
    return render_template("label_provider.html")


@app.route("/label_defs")
def label_defs_list():
    """Label definitions list endpoint."""
    return json.load(open(label_definitions_path))


@app.route("/detailed_performance")
@app.route("/ui/detailed_performance")
def detailed_performance():
    """Detailed performance UI endpoint."""
    return render_template("detailed_performance.html")


@app.route("/status", methods=["POST"])
def status():
    """Status endpoint."""
    label_update = request.get_json(force=True)
    auto_linear_train_idle_time = int(os.getenv("AUTO_LINEAR_TRAIN_IDLE_TIME", 1800))
    group_train = label_update["groupTrain"] if "groupTrain" in label_update else False
    
    # Auto linear training
    if label_update["idleTime"] > auto_linear_train_idle_time:
        if g_state.train_thread is None or not g_state.train_thread.is_alive():
            if g_state.labeled_indices is not None:
                if g_state.trained_for_version != len(g_state.labeled_indices):
                    logger.info(
                        f"Training from status: {g_state.trained_for_version=}"
                        f", {len(g_state.labeled_indices)=}, {label_update['idleTime']}"
                    )
                    g_state.train_thread = threading.Thread(
                        target=retrain_job, args=(g_state, logger)
                    )
                    g_state.train_thread.start()
    
    # Group training
    if group_train:
        if g_state.train_thread is None or not g_state.train_thread.is_alive():
            g_state.train_thread = threading.Thread(
                target=group_train_job, args=(g_state,)
            )
            g_state.train_thread.start()
    
    # Calculate estimated duration
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
                logger,
            )
        except ValueError:
            estimated_duration_s = 0
            duration_std_s = 0
    
    status_duration = (
        time.time() - g_state.time_new_status_time
        if g_state.time_new_status_time is not None
        else 0
    )
    
    # Get detailed performance metrics
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
            average_recall = detailed_performance_.recall.mean()
            average_precision = detailed_performance_.precision.mean()
        except EmptyDataError:
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
        "average_precision": average_precision,
        "average_recall": average_recall,
    }


@app.errorhandler(500)
def general_server_error(e):
    """General server error response."""
    logger.debug(e)
    return standard_json_response("general_server_error", None, 500)


def ui_script_entry():
    """UI script entry point."""
    if os.getenv("PROJECT_ROOT") is None:
        os.environ["PROJECT_ROOT"] = os.path.expanduser(sys.argv[1])

    _init()
    # After _init(), update the global variables in this module
    from annflux.ui.basic.server_init import _init_values

    global project_root, images_path, thumb_path_, failed_images_path, group_images_path
    global working_folder, exclusivity_path, label_provider_path, label_definitions_path
    global g_state, g_layout, logger
    project_root = _init_values["project_root"]
    images_path = _init_values["images_path"]
    thumb_path_ = _init_values["thumb_path_"]
    failed_images_path = _init_values["failed_images_path"]
    group_images_path = _init_values["group_images_path"]
    working_folder = _init_values["working_folder"]
    exclusivity_path = _init_values["exclusivity_path"]
    label_provider_path = _init_values["label_provider_path"]
    label_definitions_path = _init_values["label_definitions_path"]
    g_state = _init_values["g_state"]
    g_layout = _init_values["g_layout"]
    logger = _init_values["logger"]
    app.run(
        debug=str2bool(os.getenv("APP_DEBUG", False)),
        host="0.0.0.0",
        threaded=True,
        port=int(os.getenv("PORT", "8006")),
    )


if __name__ == "__main__":
    ui_script_entry()
