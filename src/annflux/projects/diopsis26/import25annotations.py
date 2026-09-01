import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas


def get_ancestors(label, relations):
    """
    Compute all ancestors for a given label in a hierarchical structure.

    Args:
        label (str): The label for which to find ancestors.
        relations (list of lists): List of child-parent relations.

    Returns:
        list: List of ancestors, ordered from immediate parent to root.
    """
    ancestors = []
    current = label

    while current is not None:
        # Find the parent of the current label
        parent = None
        for child, par in relations:
            if child == current:
                parent = par
                break

        if parent is None or parent == "null":
            break

        ancestors.append(parent)
        current = parent

    return ancestors


label_to_num_ancestors = {}


def m():
    # make hierarchy
    label_defs_25 = json.load(
        open("/mnt/big/Projects/diopsis_annotation_2/annflux/label_defs.json")
    )
    label_defs_25 = [t_[:2] for t_ in label_defs_25["labels"]]
    label_defs_coco = json.load(
        open("/home/lhogeweg/annflux/datasources/diopsis-coco/label_defs.json")
    )
    labels = label_defs_coco["labels"]
    for label in label_defs_25:
        if label not in labels:
            labels.append(label)
    #
    labels = [",".join(label) for label in labels]
    label_remap = {
        "Arachnida,Animalia": "Arachnida,Arthropoda",
        "Insecta,Animalia": "Insecta,Arthropoda",
        "Gastropoda,Animalia": "Gastropoda,Mollusca",
    }
    labels = [label_remap.get(label, label).split(",") for label in labels]
    label_defs = labels
    del labels

    #
    for label, _ in label_defs:
        label_to_num_ancestors[label] = len(get_ancestors(label, label_defs))
    print(label_defs)
    print(label_to_num_ancestors)

    # 25 annotated data
    data = json.load(open("/mnt/big/Projects/diopsis_annotation_2/annflux/labels.json"))

    quality_keys = {
        "Multiple organisms",
        "Low quality",
        "Inaccurate detection",
        "sys:review",
        "dead",
    }
    uid_to_label = {}
    filtered_labels = []
    for image_id, label_agg in data.items():
        labels = set(label_agg.split(",")) - quality_keys
        labels = [label for label in labels if "=?" not in label]
        if "Animalia" in labels and "No animalia" in labels:
            labels = {"No animalia"}
        if len(labels) > 0:
            label_ = ",".join(sorted(labels, key=label_to_num_ancestors.get))
            filtered_labels.append(label_)
            uid_to_label[image_id] = label_

    # diopsis public
    # /home/lhogeweg/annflux/datasources/diopsis-coco/labels.json
    data = json.load(
        open("/home/lhogeweg/annflux/datasources/diopsis-coco/labels.json")
    )

    for image_id, label_agg in data.items():
        labels = set(label_agg.split(",")) - {"Object"}
        if len(labels) > 0:
            if "Insecta" in labels and "Arthropoda" not in labels:
                labels.add("Arthropoda")
            label_ = ",".join(sorted(labels, key=label_to_num_ancestors.get))
            filtered_labels.append(label_)
            uid_to_label[image_id] = label_
    # print(filtered_labels)
    # exit(0)

    # clean
    # make remaps to fix issues
    remap = {}
    counts = Counter(filtered_labels)
    for label, count in counts.most_common():
        labels = label.split(",")
        if label == "Animalia,Arthropoda,Insecta,Lepidoptera,Trichoptera":
            remap[label] = "sys:review"
        elif "Leptoceridae" in labels:
            if "Trichoptera" in labels:
                remap[label] = "Animalia,Insecta,Arthropoda,Trichoptera,Leptoceridae"
            elif label == "Animalia,Arthropoda,Insecta,Leptoceridae":
                remap[label] = "Animalia,Insecta,Arthropoda,Trichoptera,Leptoceridae"
            elif "Diptera" in labels:
                remap[label] = ",".join(
                    [x_ for x_ in label.split(",") if x_ != "Leptoceridae"]
                )
            else:
                remap[label] = "sys:review"
        elif "Limoniidae" in labels:
            if "Araneae" not in labels:
                remap[label] = "Animalia,Arthropoda,Insecta,Diptera,Limoniidae"
        elif labels[-1] == "Diptera":
            if "No animalia" not in labels:
                remap[label] = "Animalia,Arthropoda,Insecta,Diptera"
            else:
                remap[label] = "sys:review"
        elif labels[-1] == "Lepidoptera":
            if label == "Animalia,Arthropoda,Lepidoptera":
                remap[label] = "Animalia,Arthropoda,Insecta,Lepidoptera"
            else:
                remap[label] = "sys:review"
        elif labels[-1] == "Chironomidae":
            if len(labels) > 5:
                remap[label] = "sys:review"
        elif labels[-1] == "Araneae":
            remap[label] = "Animalia,Arthropoda,Arachnida,Araneae"
        elif labels[-1] == "Caenidae":
            remap[label] = "Animalia,Arthropoda,Insecta,Diptera,Caenidae"
        elif labels[-1] == "Geometridae":
            remap[label] = "Animalia,Arthropoda,Insecta,Lepidoptera,Geometridae"
        elif labels[-1] == "Chiasmia clathrata":
            remap[label] = (
                "Animalia,Arthropoda,Insecta,Lepidoptera,Geometridae,Chiasmia,Chiasmia clathrata"
            )
        elif labels[-1] == "Tetragnathidae":
            if len(labels) > 5:
                remap[label] = "sys:review"
        elif labels[-1] == "Chrysopidae":
            remap[label] = "Animalia,Insecta,Arthropoda,Neuroptera,Chrysopidae"
        elif labels[-1] == "Spilosoma":
            remap[label] = "Animalia,Arthropoda,Insecta,Lepidoptera,Erebidae,Spilosoma"
        elif label in [
            "Animalia,Mollusca,Arthropoda,Gastropoda,Insecta",
            "Animalia,Gastropoda,Arthropoda,Mollusca,Insecta",
            "Animalia,Arthropoda,Mollusca,Gastropoda,Insecta",
            "Animalia,Arthropoda,Insecta,Hymenoptera,Diptera,Ptychopteridae",
        ]:
            remap[label] = "sys:review"
        elif labels[-1] == "Gastropoda":
            remap[label] = "Animalia,Mollusca,Gastropoda"
        elif label == "Animalia,Arachnida":
            remap[label] = "Animalia,Arthropoda,Arachnida"
        elif label == "Animalia,Arthropoda,Insecta,Chironomidae":
            remap[label] = "Animalia,Arthropoda,Insecta,Diptera,Chironomidae"
        elif label == "Animalia,Arthropoda,Arachnida,Araneae,Tetragnathidae,Tipulidae":
            remap[label] = "Animalia,Arthropoda,Arachnida,Araneae,Tetragnathidae"
        elif label == "Animalia,Arachnida,Opiliones":
            remap[label] = "Animalia,Arthropoda,Arachnida,Opiliones"
        elif label == "Animalia,Arthropoda,Insecta,Lepidoptera,Geometridae,Crambidae":
            remap[label] = "sys:review"

    filtered_labels = [remap.get(label, label) for label in filtered_labels]
    counts = Counter(filtered_labels)
    # print([(x_, c_) for (x_, c_) in counts.most_common() if c_ >= 3])
    clean_df = pandas.DataFrame(
        data=[
            (x_, c_, x_.split(",")[-1]) for (x_, c_) in counts.most_common() if c_ >= 3
        ],
        columns=("full_labels", "count", "end_label"),
    )
    dup_counts = (
        clean_df.groupby("end_label")
        .count()
        .sort_values("count", ascending=False)
        .reset_index()
    )
    dup_counts.to_csv("dup.csv")
    max_dup = dup_counts.max().iloc[1]
    print(f"{max_dup=}")
    counts = Counter(filtered_labels)

    #



    #
    new_uid_to_label = {}
    for uid, label in uid_to_label.items():
        new_uid_to_label[uid] = remap.get(label, label)
    print(len(new_uid_to_label))

    counts = Counter(uid_to_label.values())
    # print(counts)
    pandas.DataFrame(data=counts.items(), columns=["label", "count"]).sort_values(
        "count"
    ).to_csv("counts.csv")
    suff_labels = [x_ for (x_, c_) in counts.most_common() if c_ >= 3]
    insuff_labels = [x_ for (x_, c_) in counts.most_common() if c_ < 3]
    print(len(insuff_labels))
    print(len(suff_labels))
    new_uid_to_label = {}
    removed = []
    for uid, label in uid_to_label.items():
        if label in suff_labels:
            new_uid_to_label[uid] = remap.get(label, label)
        else:
            removed.append(label)
    print(Counter(removed))
    print(len(new_uid_to_label))
    # print(new_uid_to_label)
    #
    json.dump(new_uid_to_label, open("/mnt/big/indeed/diopsis_26/labels.json", "w"), indent=2)
    json.dump(
        {"labels": label_defs},
        open("/mnt/big/indeed/diopsis_26/label_defs.json", "w"),
        indent=2,
    )
    exit(0)
    # output
    out_folder = Path("/mnt/big/indeed/diopsis_26")
    images_folder = out_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for uid in new_uid_to_label:
        n += 1
        if uid.startswith("img_"):
            shutil.copy(
                os.path.join(
                    "/home/lhogeweg/annflux/datasources/diopsis-coco/images",
                    uid + ".jpg",
                ),
                images_folder,
            )
        else:
            shutil.copy(
                os.path.join(
                    "/mnt/big/Projects/diopsis_annotation_2/images",
                    uid + ".jpg",
                ),
                images_folder,
            )
        # print(n)


if __name__ == "__main__":
    m()
