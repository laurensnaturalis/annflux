import json

from annflux.tools.data import canon_


def m():
    j_ann = json.load(open("/mnt/big/Projects/diopsis_annotation_2/annflux/labels.json"))

    print(j_ann)
    j_ann_new = {}
    for key in j_ann:
        labels = j_ann[key].split(",")

        new_labels = []
        for label_ in labels:
            if "=?" not in label_:
                new_labels.append(label_)
        label_str = ",".join(new_labels)

        if label_str != "Animalia":
            j_ann_new[key] = canon_(label_str)

    with open("/mnt/big/Projects/diopsis_annotation_2/annflux/labels_new.json", "w") as f:
        json.dump(j_ann_new, f, indent=2)





if __name__ == '__main__':
    m()