import argparse
from collections import Counter
import json

import numpy as np
import pandas


def m(csv_path, out_label_path):
    t = pandas.read_csv(csv_path)

    label_true = t.label_true
    # label_true = [" ".join(x_.split()[:2]) for x_ in label_true]

    map_ = {}

    counts = Counter(label_true)

    print(counts)

    label_true = [x_ if counts[x_] > 1 and not pandas.isna(x_) else None for x_ in label_true]

    print(label_true)
    print(type(label_true[0]))

    out = [
        (t.image_id.values[i_], map_.get(str(x_), str(x_)))
        for i_, x_ in enumerate(label_true)
        if x_ is not None
    ]

    with open(out_label_path, "w") as f:
        json.dump(dict(out), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract labels from images")
    parser.add_argument("--csv-path", help="path to csv file with image metadata")
    parser.add_argument("--out-label-path", help="output path for labels file")

    args = parser.parse_args()
    m(args.csv_path, args.out_label_path)
