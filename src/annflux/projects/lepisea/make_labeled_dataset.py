import shutil
import os
import pandas


def m():
    t = pandas.read_csv("/mnt/big/indeed/lepisea/annflux/annflux.csv")
    images_folder = "/mnt/big/indeed/lepisea/images"

    print(len(t))

    t["label_true"] = t["label_true"].apply(lambda x_: x_.split(",") if not pandas.isnull(x_) else [])
    t["num_labels"] = t["label_true"].apply(lambda x_: len(x_))

    t = t[t["num_labels"] > 1]

    for _, row in t.iterrows():
        path = os.path.join(images_folder, row["uid"] + ".jpg")
        shutil.copy(path, "/home/lhogeweg/annflux/datasources/papbig/images")
        print(os.path.exists(path))
    t.to_csv("/home/lhogeweg/annflux/datasources/papbig/meta.csv", index=False)
    print(len(t))


if __name__ == '__main__':
    m()