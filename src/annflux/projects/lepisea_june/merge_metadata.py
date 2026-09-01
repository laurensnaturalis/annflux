"""Merge metadata columns from source CSV into annflux.csv."""
import os
import pandas


def main():
    source_csv = "/home/lhogeweg/Documents/annflux_ln/src/annflux/projects/lepisea_june/Papillot_And_Malaysian_Undersides_croppaths.csv"
    annflux_csv = "/mnt/big/indeed/lepisea2/annflux/annflux.csv"

    merge_columns = ["Country", "Island", "StateProvince", "Locality", "Collectors", "image_url"]

    # Read source CSV and create join key (normalized image_filename without extension)
    src = pandas.read_csv(source_csv, encoding="latin-1", low_memory=False)
    src = src.dropna(subset=["image_filename"])
    src["join_key"] = src["image_filename"].apply(
        lambda x: os.path.splitext(str(x))[0].replace(".", "_")
    )
    # Keep only needed columns and deduplicate
    src = src[["join_key"] + merge_columns].drop_duplicates(subset="join_key")

    # Read annflux CSV
    annflux = pandas.read_csv(annflux_csv, low_memory=False)

    print(f"annflux rows: {len(annflux)}")
    print(f"source rows (unique keys): {len(src)}")

    # Drop columns that already exist in annflux to avoid duplicates
    existing_merge_cols = [c for c in merge_columns if c in annflux.columns]
    if existing_merge_cols:
        print(f"Dropping existing columns: {existing_merge_cols}")
        annflux = annflux.drop(columns=existing_merge_cols)

    # Merge on uid == join_key
    merged = annflux.merge(src, left_on="uid", right_on="join_key", how="left")
    merged = merged.drop(columns=["join_key"])

    matched = merged[merge_columns[-1]].notna().sum()
    print(f"Matched: {matched} / {len(annflux)}")

    # Save
    merged.to_csv(annflux_csv, index=False)
    print(f"Saved to {annflux_csv}")


if __name__ == "__main__":
    main()
