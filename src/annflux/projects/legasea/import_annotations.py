import json
import re

import pandas

from annflux.tools.io import basename_no_extension


def use_in_parenthesis(x_):
    match = False
    if not pandas.isna(x_):
        match = re.search(r'\((.*)\)', x_)
    if match:
        return match.group(1)
    else:
        return x_


def m():
    t = pandas.read_csv("/mnt/big/indeed/legasea_big/source_data/all_taxa_A.csv")
    ranks = ["superkingdom", "kingdom", "division", "class", "order", "family", "genus", "species"]
    t["superkingdom"] = None
    for rank in ranks[:-1]:
        t[rank] = t[rank].apply(use_in_parenthesis)
    t["species"] = t[["genus", "specific_epithet"]].apply(
        lambda x_: " ".join([y_ for y_ in x_ if not pandas.isna(y_)]).strip(), axis=1
    )

    label_defs = json.load(open("/mnt/big/indeed/legasea_big/annflux/label_defs.json"))
    print(label_defs["labels"])
    out_pairs = set()
    for pair in [ranks[i : i + 2][::-1] for i in range(len(ranks) - 1)]:
        for pair_ in t[[*pair]].values:
            if (
                not pandas.isna(pair_[0])
                # and not pandas.isna(pair_[1])
                and len(pair_[0]) > 0
                # and len(pair_[1]) > 0
                and not pair_[0] == pair_[1]
            ):
                out_pairs.add(tuple(pair_))
    print(out_pairs)
    print(t.columns)
    label_defs["labels"].extend(out_pairs)

    with open("/mnt/big/indeed/legasea_big/annflux/label_defs.json", "w") as f:
        json.dump(label_defs, f, indent=2       )


def m2():
    t = pandas.read_csv("/mnt/big/indeed/legasea_big/source_data/all_images_A_T_NBC_only.csv")
    t["uid"] = t["image_url"].apply(lambda x_: basename_no_extension(x_))
    print(t["image_id"].str.replace(":", "_NS_"))

    label_defs = json.load(open("/mnt/big/indeed/legasea_big/annflux/label_defs.json"))

    children, parents = zip(*label_defs["labels"])

    for label_def in t.taxon_full_name_A.unique():
        if use_in_parenthesis(label_def) not in set(children):
            print(label_def)

    labels_path = "/mnt/big/indeed/legasea_big/annflux/labels.json"
    labels = json.load(open(labels_path))
    for _, row in t.iterrows():
        labels[row["uid"]] = use_in_parenthesis(row["taxon_full_name_A"])

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    del t
    group_data_path = "/mnt/big/indeed/legasea_big/annflux/group0_annflux.csv"
    t = pandas.read_csv(group_data_path)

    labels_for_t = [labels.get(uid) for uid in t["uid"]]
    t["label_true"] = labels_for_t

    t.to_csv(group_data_path, index=False)



if __name__ == "__main__":
    m2()
