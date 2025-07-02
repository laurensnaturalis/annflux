from collections import Counter

import pandas


def m():
    t = pandas.read_csv("/mnt/big/indeed/mollusca/annflux/annflux.csv")
    # t = pandas.read_csv("/mnt/big/indeed/diopsis-hazehorst-apr5-6/annflux/annflux.csv")

    labeled = t[t.labeled == 1]

    print(len(labeled))

    for cluster in sorted(labeled["dp_cluster"].unique()):
        print(cluster, len(t[t["dp_cluster"] == cluster]), Counter(labeled[labeled["dp_cluster"] == cluster].label_true).most_common())


if __name__ == "__main__":
    m()
