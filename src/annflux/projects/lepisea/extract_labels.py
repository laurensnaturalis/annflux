import json
from collections import Counter
from pydoc import parentname

import pandas
import glob


def make_species(t_):
    if pandas.isna(t_[0]) and not pandas.isna(t_[1]):
        return "ERROR"
    t_ = map(str, t_)
    t_ = [x_ if "?" not in x_ else "" for x_ in t_]
    t_ = [x_ if x_ != "nan" else "" for x_ in t_]
    return " ".join(t_).strip()


def make_subspecies(t_):
    has_none = False
    for x_ in t_:
        has_none |= pandas.isna(x_) is None
        if not pandas.isna(x_) and has_none:
            return "ERROR"
    t_ = map(str, t_)
    t_ = [x_ if "?" not in x_ else "" for x_ in t_]
    t_ = [x_ if x_ != "nan" else "" for x_ in t_]
    return " ".join(t_).strip()


def make_sex(t_):
    sex = str(t_[2])
    if sex == "nan":
        return ""
    if len(t_[0].strip()) == 0:
        sex = "ERROR"
    elif len(t_[1].strip()) == 0:
        sex = make_subspecies([t_[0], t_[2]])
    else:
        sex = make_subspecies([t_[1], t_[2]])

    return sex


def m():
    tables = [
        pandas.read_csv(fn)
        for fn in glob.glob("/mnt/big/indeed/lepisea/source_data/*.csv")
    ]

    table = pandas.concat(tables)

    print(table.count())

    exit(0)

    columns = ["Family", "Genus", "Species", "Subspecies", "Infrasubspecies", "Sex"]
    taxonomy = table[columns]
    taxonomy.drop_duplicates(inplace=True)
    # print(taxonomy)
    taxonomy["Name"] = taxonomy[columns].apply(
        lambda t_: " ".join(map(str, t_)), axis=1
    )
    taxonomy["Species"] = taxonomy[["Genus", "Species"]].apply(make_species, axis=1)
    taxonomy["Subspecies"] = taxonomy[["Species", "Subspecies"]].apply(
        make_subspecies, axis=1
    )
    taxonomy["Sex"] = taxonomy[["Species", "Subspecies", "Sex"]].apply(make_sex, axis=1)
    taxonomy.dropna(subset=columns, inplace=True, how="all")
    print(len(taxonomy))
    name_to_taxonomy = dict(zip(taxonomy["Name"].values, taxonomy[columns].values))
    taxonomy.to_csv("taxonomy.csv")

    annotations = {}
    child_to_parent = {}
    table["Name"] = table[columns].apply(lambda t_: " ".join(map(str, t_)), axis=1)
    print(name_to_taxonomy)
    for _, row in table.iterrows():
        multilabel_ = name_to_taxonomy[row["Name"]]
        multilabel = []
        for val in multilabel_:
            if not pandas.isna(val) and len(val.strip()) > 0 and val not in multilabel:
                multilabel.append(val)
        if "ERROR" in multilabel:
            continue
        for i_ in reversed(range(len(multilabel))):
            if multilabel[i_] not in child_to_parent:
                if i_ > 0:
                    child_to_parent[multilabel[i_]] = multilabel[i_ - 1]
                else:
                    child_to_parent[multilabel[i_]] = "null"
            else:
                parent = multilabel[i_ - 1] if i_ > 0 else "null"
                if parent != child_to_parent[multilabel[i_]]:
                    if child_to_parent[multilabel[i_]] == "null":
                        child_to_parent[multilabel[i_]] = parent
                    elif parent == "null":
                        pass
                    else:
                        print(f"Found conflicting parent for {multilabel[i_]}, {parent} vs {child_to_parent[multilabel[i_]]}")



        if len(multilabel) > 0:
            multilabel = ",".join(multilabel)
            if multilabel != "ERROR":
                annotations[row["Registration nr"].replace(".", "_")] = multilabel

    with open("/mnt/big/indeed/lepisea/annflux/labels.json", "w") as f:
        json.dump(annotations, f, indent=2)
    with open("/mnt/big/indeed/lepisea/annflux/label_defs.json", "w") as f:
        json.dump({"labels": list(child_to_parent.items())}, f, indent=2)

    print(table.columns)


if __name__ == "__main__":
    m()
