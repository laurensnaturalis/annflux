import json

import pandas
import glob
import os

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

    table = pandas.read_csv("/code/src/annflux/projects/lepisea_june/Augmented_images_other_sources_fixed.csv", encoding="latin-1")
    table["Order"] = "Lepidoptera"
    table["Family"] = table["Family"].replace("Riodinidae", "Lycaenidae")
    table.loc[table["Genus"] == "Papilio", "Family"] = "Papilionidae"
    table.loc[table["Genus"] == "Autoba", "Family"] = "Erebidae"
    table.loc[table["Family"] == "lycaenidae", "Family"] = "Lycaenidae"
    table.loc[table["Family"] == "Lycaenidae", "Butterfly_Moth"] = "Butterfly"
    print(table.count())

    
    columns = ["Order", "Butterfly_Moth", "Family", "Genus", "Species", "Subspecies"]
    taxonomy = table[columns]
    taxonomy.drop_duplicates(inplace=True)
    # print(taxonomy)
    taxonomy["Name"] = taxonomy[columns].apply(
        lambda t_: " ".join(map(str, t_)), axis=1
    )
    taxonomy["Species"] = taxonomy[["Genus", "Species"]].apply(make_species, axis=1)
    # taxonomy["subspecies"] = taxonomy[["Species", "Subspecies"]].apply(
    #     make_subspecies, axis=1
    # )
    # taxonomy["sex"] = taxonomy[["Species", "Subspecies", "Sex"]].apply(make_sex, axis=1)
    taxonomy.dropna(subset=columns, inplace=True, how="all")
    print(len(taxonomy))
    name_to_taxonomy = dict(zip(taxonomy["Name"].values, taxonomy[columns].values))
    taxonomy.to_csv("taxonomy_other.csv")

    annotations = {}
    child_to_parent = {}
    seen_conflicts = set()
    table["Name"] = table[columns].apply(lambda t_: " ".join(map(str, t_)), axis=1)
    print(name_to_taxonomy)
    for i, row in table.iterrows():
        # if i  == 0:
        #     print(row["image_filename"])
        #     print(row["Name"])
        #     print(row[columns])
        #     print(f"Processing row {i}")
        #     break

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
                        conflict_key = (multilabel[i_], parent, child_to_parent[multilabel[i_]])
                        if conflict_key not in seen_conflicts:
                            seen_conflicts.add(conflict_key)
                            print(f"Found conflicting parent for {multilabel[i_]}: {parent} and {child_to_parent[multilabel[i_]]}")



        if len(multilabel) > 0:
            multilabel = ",".join(multilabel)
            if multilabel != "ERROR":
                annotations[os.path.splitext(row["image_filename"])[0].replace(".", "_")] = multilabel

    with open("labels_other.json", "w") as f:
        json.dump(annotations, f, indent=2)
    with open("label_defs_other.json", "w") as f:
        json.dump({"labels": list(child_to_parent.items())}, f, indent=2)

    print(table.columns)


if __name__ == "__main__":
    m()
