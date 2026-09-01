import json

import pandas


def m():
    # t = pandas.read_csv("/home/lhogeweg/annflux/datasources/diopsis-coco/labels.csv")

    # print(t["label_true"].unique())

    j_data = json.load(open("/home/lhogeweg/annflux/datasources/diopsis-coco/instances_train.json"))

    names = [j_cat["name"] for j_cat in j_data["categories"]]

    n_species = 0
    n_families = 0
    for name in names:
        if len(name.split()) > 1:
            n_species += 1
        elif name.endswith("ae"):
            n_families += 1
        else:
            print(name)
    print(len(names))

    print(n_species)
    print(n_families)






if __name__ == '__main__':
    m()