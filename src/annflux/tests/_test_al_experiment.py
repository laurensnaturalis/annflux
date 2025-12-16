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
from io import BytesIO
from pathlib import Path
from time import sleep

import flask
import pandas
import pytest

from annflux.data.bombus_plant_test.data import (
    DataSource,
    StreetSurfaceVis,
)
from annflux.scripts.annflux_cli import go_command
from annflux.shared import AnnfluxSource
from annflux.ui.basic.run_server import _init, get_app

annflux_data_path: str | None | Path = None
data_source: DataSource | None = None


def create_app():
    #
    global annflux_data_path, data_source
    data_source = StreetSurfaceVis()
    data_source.download()
    data_folder = Path(os.path.expanduser(f"~/annflux/data/{data_source.name}"))
    if os.path.isdir(data_folder):
        shutil.rmtree(data_folder)
    data_source.copy_to(data_folder)
    go_command(
        AnnfluxSource(data_folder),
        start_labels=set(json.load(open(data_source.true_labels_path)).values()),
        random_seed=42,
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


def test_al_strategies(client):
    true_labels = json.load(open(data_source.true_labels_path))
    client.post(
        "/label", json={}
    )  # TODO(issue): most_needed column not available before refresh
    t = get_annflux_data(client)
    num_test_images = 912
    assert t["in_test"].sum() == num_test_images
    test_labeling = {uid_: true_labels[uid_] for uid_ in t[t["in_test"] == 1]["uid"]}
    client.post("/label", json=test_labeling)
    print(get_json(client, "/performance"))
    assert len(get_json(client, "/performance")["test_performance"]) == 0
    t = get_annflux_data(client)
    active_set_size = 1000
    active_round = 0
    percentage_near_labeled = 0
    active_strategy = "most_needed"
    avg_accuracies = []
    avg_recalls = []
    avg_precisions = []
    strategies = []
    while t["labeled"].sum() < len(t):
        t.sort_values(active_strategy, inplace=True)
        active_uids = t[t["labeled"] == 0].uid.values[:active_set_size]
        labeling = {active_uid: true_labels[active_uid] for active_uid in active_uids}
        client.post("/label", json=labeling)
        t = get_annflux_data(client)
        assert t["labeled"].sum() == (
            active_round + 1
        ) * active_set_size + num_test_images or t["labeled"].sum() == len(t)
        j_performance = get_json(client, "/performance")
        print(j_performance)
        percentage_near_labeled = j_performance["percentage_near_labeled"]
        if percentage_near_labeled > 0.99:
            active_strategy = "fre"
        print(active_strategy)
        detailed_performance = get_data_csv(client, "/detailed_performance/data")
        avg_recalls.append(detailed_performance["recall"].mean())
        avg_precisions.append(detailed_performance["precision"].mean())
        avg_accuracies.append(j_performance["test_performance"][-1][2])
        strategies.append(active_strategy)
        # trigger linear training
        if active_round % 10 == 1:
            client.post(
                "/status", json={"idleTime": 1800 + 1}
            )  # TODO: replace constant
            sleep(5)
            while client.post("/status", json={"idleTime": 0}).json["status"] == "training":
                print("=========== waiting for training")
                sleep(5)
            while client.post("/status", json={"idleTime": 0}).json["status"] != "idle":
                print("=========== waiting to become idle")
                sleep(1)
        active_round += 1
    print(avg_accuracies)
    print(avg_recalls)
    print(avg_precisions)

    print(len(avg_accuracies), len(avg_recalls), len(avg_precisions))
    pandas.DataFrame(
        data={
            "accuracy": avg_accuracies,
            "recall": avg_recalls,
            "precision": avg_precisions,
            "strategies": strategies,
        }
    ).to_csv(
        "/home/lhogeweg/Documents/annflux_ln/src/annflux/projects/al_evaluation/test_al_strategies.csv"
    )


def get_annflux_data(client) -> pandas.DataFrame:
    response = client.get("/data")
    t = pandas.read_parquet(BytesIO(response.data))
    t["uid"] = t["uid"].astype(str)
    return t


def get_data_csv(client, url) -> pandas.DataFrame:
    response = client.get(url)
    return pandas.read_csv(BytesIO(response.data))


def get_json(client, url):
    response = client.get(url)
    return json.loads(response.data)
