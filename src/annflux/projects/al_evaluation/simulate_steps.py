import json
import os
import shutil
from pathlib import Path

import requests
import pandas

from annflux.data.bombus_plant_test.data import PapBig, StreetSurfaceVis
from annflux.scripts.annflux_cli import go_command
from annflux.shared import AnnfluxSource


class Client(object):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url

    def post(self, endpoint, json):
        requests.post(f"{self.base_url}{endpoint}", json=json)


url = "http://127.0.0.1:8006"
client = Client(url)
data_source = PapBig()


def init():
    data_source.download()
    data_folder = Path(os.path.expanduser(f"~/annflux/data/{data_source.name}"))
    if os.path.isdir(data_folder):
        shutil.rmtree(data_folder)
    data_source.copy_to(data_folder)
    go_command(
        AnnfluxSource(data_folder),
        start_labels=set(json.load(open(data_source.true_labels_path)).values()),
        random_seed=42,
        # existing_feature_cache_path="/home/lhogeweg/annflux/data/papbig_features/model_0e627184de1ec11ed8dccf0ba5dde8624a45a4bbe8cce74d8fcbb444_dataset_ed541cdf7fd64bb876b2d0a2ebabeaf0489a9dd1938194c27d7ff8a7.zarr",
    )
    annflux_folder = data_folder / "annflux"
    annflux_data_path = annflux_folder / "annflux.csv"
    print(f"Project in {data_folder}")

def set_test_labeling():
    true_labels = json.load(open(data_source.true_labels_path))
    t = pandas.read_parquet(url + "/data")
    t["uid"] = t["uid"].astype(str)
    test_labeling = {uid_: true_labels[uid_] for uid_ in t[t["in_test"] == 1]["uid"]}
    client.post("/label", json=test_labeling)
    t = pandas.read_parquet(url + "/data")
    print(len(t[(t.in_test==1) & (t.labeled==1)]))


def first_round():
    # 500 annotations + linear training
    client.post("/label", json={})
    true_labels = json.load(open(data_source.true_labels_path))
    active_set_size = 500
    active_strategy = "dp_most_needed"
    t = pandas.read_parquet(url + "/data")
    t["uid"] = t["uid"].astype(str)
    t.sort_values(active_strategy, inplace=True)
    active_uids = t[t["labeled"] == 0].uid.values[:active_set_size]
    labeling = {active_uid: true_labels[active_uid] for active_uid in active_uids}
    # client.post("/label", json=labeling)
    client.post("/label", json=labeling)

def second_plus_round():
    # 500 annotations
    client.post("/label", json={})
    true_labels = json.load(open(data_source.true_labels_path))
    active_set_size = 2000
    active_strategy = "fre_strat"
    t = pandas.read_parquet(url + "/data")
    t["uid"] = t["uid"].astype(str)
    print(len(t[(t.labeled==1) & (t.in_test==0)]))
    t.sort_values(active_strategy, inplace=True)
    active_uids = t[t["labeled"] == 0].uid.values[:active_set_size]
    labeling = {active_uid: true_labels[active_uid] for active_uid in active_uids}
    # client.post("/label", json=labeling)
    client.post("/label", json=labeling)
    t = pandas.read_parquet(url + "/data")
    print(len(t[(t.labeled == 1) & (t.in_test == 0)]))


if __name__ == "__main__":
    second_plus_round()
