import os
import re

from annflux.tools.io import write_label_defs


def split_on_capitals(s):
    return re.findall('[A-Z][a-z]*', s)

def m(class_folder: str):
    class_folders = os.listdir(class_folder)

    # images_folder = os.path.join(class_folder, "..", "images")
    # # os.makedirs(images_folder, exist_ok=False)
    #
    # image_paths = glob.glob(os.path.join(class_folder, "**", "*.jpg")) + glob.glob(os.path.join(class_folder, "**", "*.JPG"))
    #
    # bla_ = set([os.path.basename(x_) for x_ in image_paths])
    # print(bla_)
    # print(len(image_paths), len(bla_))
    # assert len(image_paths) == len(bla_)
    #
    # for image_path in image_paths:
    #     tmp_ = os.path.join(images_folder, os.path.splitext(os.path.basename(image_path))[0] + ".jpg")
    #     if not os.path.exists(tmp_):
    #         os.symlink(image_path, tmp_)

    label_defs = []
    org_to_label_def = {}

    for org_name in class_folders:
        label = split_on_capitals(org_name)
        if len(label) == 1:
            label_defs.append((label[0], "null"))
        elif len(label) == 2:
            label_ = label[::-1]
            if label_[0] in ["Other"]:
                label_ = (f"{label_[1]}_{label_[0]}", label_[1])
            label_defs.append(label_)
        org_to_label_def[org_name] = label_defs[-1]
    assert len(org_to_label_def) == len(class_folders)

    print(org_to_label_def)

    print(label_defs)

    write_label_defs(class_folder, label_defs)

if __name__ == '__main__':
    m("/mnt/big/indeed/broadTaxonClassifier/raw")