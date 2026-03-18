from collections import Counter

import pandas


def count_info(x_):
    if pandas.isna(x_):
        return "unlabeled"
    tokens = x_.split(",")

    if len(tokens) == 1:
        if tokens[0].endswith("ae"):
            return "family"
        else:
            return "unknown"
    elif len(tokens) == 2:
        return "genus"
    elif len(tokens) == 3:
        return "species"
    else:
        return "infraspecies"






def m():
    t = pandas.read_csv("/mnt/big/indeed/lepisea/annflux/annflux.csv")

    t["count_info"] = t["label_true"].apply(lambda x_: count_info(x_))
    print(t["label_true"])

    print(Counter(t["count_info"]))

    print(t[t["count_info"] == "unknown"]["label_true"])


if __name__ == '__main__':
    m()