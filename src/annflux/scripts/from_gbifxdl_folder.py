import glob
import os
import re

import pandas
from taxonlib.tools.taxa import remove_author

from annflux.tools.io import write_label_defs


def split_on_capitals(s):
    return re.findall('[A-Z][a-z]*', s)

def m(class_folder: str, metadata_path: str):
    class_folders = [x_ for x_ in os.listdir(class_folder) if os.path.isdir(os.path.join(class_folder, x_))]
    metadata = pandas.read_parquet(metadata_path)
    taxonkey_to_name = {str(row.taxonKey):remove_author(row.scientificName) for _, row in metadata.iterrows()}
    print(taxonkey_to_name)
    print(metadata.scientificName.values[0], remove_author(metadata.scientificName.values[0]))
    # exit(0)

    for _, row in metadata.iterrows():
        fn = os.path.join(class_folder, row.taxonKey, row.url_hash + ".jpeg")
        print(fn, os.path.exists(fn))
    # exit(1)

    images_folder = os.path.join(class_folder, "..", "images")
    os.makedirs(images_folder, exist_ok=True)

    image_paths = glob.glob(os.path.join(class_folder, "**", "*.jpg")) + glob.glob(os.path.join(class_folder, "**", "*.JPG")) + glob.glob(os.path.join(class_folder, "**", "*.jpeg"))

    bla_ = set([os.path.basename(x_) for x_ in image_paths])
    print(bla_)
    print(len(image_paths), len(bla_))
    assert len(image_paths) == len(bla_)

    for image_path in image_paths:
        tmp_ = os.path.join(images_folder, os.path.splitext(os.path.basename(image_path))[0] + ".jpg")
        if not os.path.exists(tmp_):
            os.symlink(image_path, tmp_)

    label_defs = []
    org_to_label_def = {}

    for taxonKey in class_folders:
        label = [taxonkey_to_name[taxonKey]]
        if len(label) == 1:
            label_defs.append((label[0], "null"))
        elif len(label) == 2:
            label_ = label[::-1]
            if label_[0] in ["Other"]:
                label_ = (f"{label_[1]}_{label_[0]}", label_[1])
            label_defs.append(label_)
        org_to_label_def[taxonKey] = label_defs[-1]
    assert len(org_to_label_def) == len(class_folders)

    print(org_to_label_def)

    print(label_defs)

    write_label_defs(class_folder, label_defs)

if __name__ == '__main__':
    m("/home/lhogeweg/Documents/dugnatforhavet/dataset/dataset_dir", "/home/lhogeweg/Documents/dugnatforhavet/dataset/dataset_dir/0068260-250525065834625.parquet")