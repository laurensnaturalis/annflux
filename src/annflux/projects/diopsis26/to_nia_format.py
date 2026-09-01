import json

import pandas

from annflux.projects.diopsis26.import25annotations import get_ancestors


def m():
    t = pandas.read_csv("/mnt/big/indeed/diopsis_26/annflux/annflux.csv")

    # ,image_url,observation_id,image_id,taxon_id_at_source,taxon_full_name
    # TODO: check taxon_full_name unique
    t2 = pandas.DataFrame(
        data=zip(t["filename"], t["uid"], t["label_true"]),
        columns=["filename", "image_id", "label_true"],
    )
    t2 = t2[t2["label_true"] != "sys:review"]

    t2.rename(columns={"filename": "image_url"}, inplace=True)

    label_defs = json.load(open("/mnt/big/indeed/diopsis_26/annflux/label_defs.json"))[
        "labels"
    ]
    label_to_num_ancestors = {}
    for label, _ in label_defs:
        label_to_num_ancestors[label] = len(get_ancestors(label, label_defs))

    t2["taxon_full_name"] = t2["label_true"].apply(
        lambda x_: sorted(x_.split(","), key=label_to_num_ancestors.get)[-1]
    )
    t2["taxon_id_at_source"] = t2["taxon_full_name"].apply(
        lambda x_: f"DIOPSIS:{x_.upper().replace(' ', '_').replace('.','')}"
    )
    t2["observation_id"] = t2["image_id"].apply(lambda x_: "DIOPSIS:R" + x_)
    t2["image_id"] = t2["image_id"].apply(lambda x_: "DIOPSIS:" + x_)

    # label_to_taxon_name = dict(t2.drop_duplicates(("taxon_full_name", "label_true"))[["taxon_full_name", "label_true"]].values)
    t2.to_csv("/mnt/big/storage/diopsis26/source_data/all_images.csv")
    # TODO: remove sys:review
    # taxon_full_name,taxon_id_at_source,status_at_source,accepted_taxon_id_at_source,kingdom,division,class,order,family,genus,specific_epithet
    taxon_rows = []
    for taxon_full_name in t2["taxon_full_name"].unique():
        taxonomy = list(reversed(get_ancestors(taxon_full_name, label_defs))) + [taxon_full_name]
        species_tokens = taxonomy[-1].split()
        if len(species_tokens) == 2:
            taxonomy[-1] = species_tokens[1]
        for _ in range(7 - len(taxonomy)):
            taxonomy.append(None)
        row = [taxon_full_name, f"DIOPSIS:{taxon_full_name.upper().replace(' ', '_').replace('.','')}", "accepted", ""]
        for taxon in taxonomy:
            row.append(taxon)
        taxon_rows.append(row)
        print(taxon_full_name, row)

    taxa = pandas.DataFrame(
        taxon_rows,
        columns="taxon_full_name,taxon_id_at_source,status_at_source,accepted_taxon_id_at_source,kingdom,division,class,order,family,genus,specific_epithet".split(
            ","
        ),
    ).to_csv("/mnt/big/storage/diopsis26/source_data/all_taxa.csv", index=False)


if __name__ == "__main__":
    m()
