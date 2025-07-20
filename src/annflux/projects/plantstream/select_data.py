import pandas
import numpy as np

def m():
    t = pandas.read_parquet(
        "/media/lhogeweg/My Book/storage/msm24/source_data/all_images.parquet"
    )
    del t["location.longitude"]
    taxa = pandas.read_csv(
        "/media/lhogeweg/My Book/storage/msm24/source_data/all_taxa.csv"
    )
    taxa = taxa[taxa.kingdom == "Plantae"]

    print(len(taxa))

    plants = t[t.taxon_id_at_source.isin(set(taxa.taxon_id_at_source))]
    plants.reset_index(drop=True, inplace=True)

    print(plants.morph.unique())

    for morph in plants.morph.unique():
        if morph is None:
            continue
        morph_morph = np.where(plants.morph == morph)[0]
        print(morph, morph_morph)
        if morph.startswith("source_") or morph == "unknown":
            plants.loc[morph_morph, "morph"] = ""
        else:
            if "." in morph or " " in morph:
                morph_ = morph.replace(".", "").replace(" ", "-")
                print(morph, morph_)
                plants.loc[morph_morph, "morph"] = morph_
    print(plants.morph.unique())
    print(plants.dtypes)
    # plants = plants.sample(n=1000)
    plants["image_id"] = plants["image_id"].str.replace(":", "_NS_")
    plants.to_parquet("/mnt/big/indeed/plantstream/source_data/stream_source.pq")


if __name__ == "__main__":
    m()
