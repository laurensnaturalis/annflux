import shutil
import os
from collections import Counter

import pandas


def m():
    t = pandas.read_csv("/mnt/big/indeed/lepisea/annflux/annflux.csv")
    images_folder = "/mnt/big/indeed/lepisea/images"

    print(len(t))

    t["label_true_split"] = t["label_true"].apply(lambda x_: x_.split(",") if not pandas.isnull(x_) else [])
    t["num_labels"] = t["label_true_split"].apply(lambda x_: len(x_))
    t["label_true"] = t["label_true_split"].apply(lambda x_: ','.join([y_ for y_ in x_ if len(y_.split()) <= 2]))
    t = t[t["num_labels"] > 2]
    included_labels = [x_ for x_, count_ in Counter(t["label_true"]).most_common() if count_ >= 10]
    t = t[t["label_true"].isin(included_labels)]
    print(len(t["label_true"].unique()))

    for _, row in t.iterrows():
        path = os.path.join(images_folder, row["uid"] + ".jpg")
        shutil.copy(path, "/home/lhogeweg/annflux/datasources/papbig/images")
        print(os.path.exists(path))
    t.to_csv("/home/lhogeweg/annflux/datasources/papbig/meta.csv", index=False)
    print(len(t))


if __name__ == '__main__':
    m()