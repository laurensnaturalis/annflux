import argparse
import itertools
import json
from typing import List


def split_label_string(label_string: str, remove_unknown=True) -> List[str]:
    return [(x_.replace("=?", "") if remove_unknown else x_) for x_ in label_string.split(",") if x_.strip() != ""]


def m(path, out_path):
    labels = json.load(open(path))

    unique_labels = set(itertools.chain.from_iterable([split_label_string(x_) for x_ in labels.values()]))

    with open(out_path, "w") as f:
        json.dump({"labels": list((label_, "null") for label_ in unique_labels)}, f, indent=2)

"""
{"labels": [["Beaver-bite", "null"], ["Molehill", "null"], ["Scat", "null"], ["Snow", "null"], ["Soil", "null"], ["Size-reference", "null"]]}"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Process labels")
    parser.add_argument("input_folder", help="Folder containing label files")

    args = parser.parse_args()

    label_file = f"{args.input_folder}/labels.json"
    out_path = f"{args.input_folder}/label_defs.json"

    m(label_file, out_path)
