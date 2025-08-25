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
import json
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import flask
import pandas
import pytest

from annflux.data.bombus_plant_test.data import BombusPlantTest, DataSource
from annflux.scripts.annflux_cli import go_command
from annflux.shared import AnnfluxSource
from annflux.ui.basic.run_server import _init, get_app

annflux_data_path: str | None = None
data_source: DataSource | None = None

def create_app():
    #
    global annflux_data_path, data_source
    data_source = BombusPlantTest()
    data_source.download()
    data_folder = Path(os.path.expanduser("~/annflux/data/bombus-plant-test"))
    shutil.rmtree(data_folder)
    data_source.copy_to(data_folder)
    print("here")
    go_command(
        AnnfluxSource(data_folder), start_labels=["No plant", "Flowering", "Vegetative"]
    )
    annflux_folder = data_folder / "annflux"
    annflux_data_path = annflux_folder / "annflux.csv"
    os.environ["PROJECT_ROOT"] = str(data_folder)
    #
    app = get_app()
    app.config.from_mapping(TESTING=True)
    _init()
    return app


@pytest.fixture
def app():
    app: flask.Flask = create_app()

    yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


def test_index(client):
    response = client.get("/")
    print(response.data)
    assert b"/static/annflux_layout.js" in response.data


def test_refresh(client):
    response = client.post("/label", json={})
    print(response.data)
    assert json.loads(response.data) == {"success": True}

def test_label(client):
    true_labels = json.load(open(data_source.true_labels_path))
    response = client.post("/label", json={})  # TODO(issue): most_needed column not available before refresh
    t = get_annflux_data(client)
    t.sort_values("most_needed", inplace=True)
    assert t["labeled"].max() == 0
    active_uids = (t[t["labeled"]==0].uid.values[:50])
    labeling = {}
    for active_uid in active_uids:
        labeling[active_uid] = true_labels[active_uid]
    response = client.post("/label", json=labeling)
    t = get_annflux_data(client)
    assert t["labeled"].sum() == 50
    # assert json.loads(response.data) == {"success": True}
    print(get_json(client, "/performance"))

def test_label_loop(client):
    true_labels = json.load(open(data_source.true_labels_path))
    response = client.post("/label", json={})  # TODO(issue): most_needed column not available before refresh
    t = get_annflux_data(client)
    t.sort_values("most_needed", inplace=True)
    assert t["labeled"].max() == 0
    active_set_size = 50
    active_round = 0
    percentage_near_labeled = 0
    while t["labeled"].sum() < len(t):
        t.sort_values("most_needed", inplace=True)
        active_uids = (t[t["labeled"]==0].uid.values[:active_set_size])
        labeling = {active_uid: true_labels[active_uid] for active_uid in active_uids}
        client.post("/label", json=labeling)
        t = get_annflux_data(client)
        assert t["labeled"].sum() == (active_round + 1) * active_set_size
        active_round += 1
        j_performance = get_json(client, "/performance")
        print(j_performance)
        percentage_near_labeled = j_performance["percentage_near_labeled"]
        print([t_[2] for t_ in j_performance["test_performance"]])


def test_al_strategies(client):
    true_labels = json.load(open(data_source.true_labels_path))
    client.post("/label", json={})  # TODO(issue): most_needed column not available before refresh
    t = get_annflux_data(client)
    num_test_images = 50
    assert t["in_test"].sum() == num_test_images
    test_labeling = {uid_: true_labels[uid_] for uid_ in t[t["in_test"]==1]["uid"]}
    client.post("/label", json=test_labeling)
    print(get_json(client, "/performance"))
    assert len(get_json(client, "/performance")["test_performance"]) == 0
    t = get_annflux_data(client)
    active_set_size = 10
    active_round = 0
    percentage_near_labeled = 0
    active_strategy = "most_needed"
    avg_accuracies = []
    avg_recalls = []
    avg_precisions = []
    while t["labeled"].sum() < len(t):
        t.sort_values(active_strategy, inplace=True)
        active_uids = (t[t["labeled"]==0].uid.values[:active_set_size])
        labeling = {active_uid: true_labels[active_uid] for active_uid in active_uids}
        client.post("/label", json=labeling)
        t = get_annflux_data(client)
        assert t["labeled"].sum() == (active_round + 1) * active_set_size + num_test_images
        active_round += 1
        j_performance = get_json(client, "/performance")
        print(j_performance)
        percentage_near_labeled = j_performance["percentage_near_labeled"]
        if percentage_near_labeled > 0.99:
            active_strategy = "fre"
        print(active_strategy)
        detailed_performance = get_data_csv(client, "/detailed_performance/data")
        avg_recalls.append(detailed_performance["recall"].mean())
        avg_precisions.append(detailed_performance["precision"].mean())
        avg_accuracies = [t_[2] for t_ in j_performance["test_performance"]]
    pandas.DataFrame(data={"accuracy": avg_accuracies, "recall": avg_recalls, "precision": avg_precisions}).to_csv("test_al_strategies.csv")
    print(avg_recalls)
    print(avg_precisions)


def get_annflux_data(client) -> pandas.DataFrame:
    response = client.get("/data")
    t = pandas.read_parquet(BytesIO(response.data))
    return t

def get_data_csv(client, url) -> pandas.DataFrame:
    response = client.get(url)
    return pandas.read_csv(BytesIO(response.data))

def get_json(client, url):
    response = client.get(url)
    return json.loads(response.data)