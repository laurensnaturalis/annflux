import ast
import json
import os

import pandas


def m():
    t_labels = pandas.read_csv(
        "/home/lhogeweg/annflux/datasources/diopsis-coco/labels.csv"
    )

    t_ancestors = pandas.read_csv(
        "/home/lhogeweg/annflux/datasources/diopsis-coco/name_to_ancestors.csv"
    )
    ancetors = dict(
        zip(t_ancestors["name"], map(ast.literal_eval, t_ancestors["ancestors"]))
    )
    ancetors["Object"] = ["Object"]

    with open("/home/lhogeweg/annflux/datasources/diopsis-coco/labels.json", "w") as f:
        json.dump(
            dict(
                zip(
                    [os.path.splitext(x_)[0] for x_ in t_labels["basename of crop"]],
                    [",".join(ancetors[x_]) for x_ in t_labels["label_true"]],
                )
            ),
            f,
            indent=2,
        )

    label_defs = []
    for key, val in ancetors.items():
        label_defs.append((key, val[1] if len(val) > 1 else "null"))
    with open(
        "/home/lhogeweg/annflux/datasources/diopsis-coco/label_defs.json", "w"
    ) as f:
        json.dump(
            {
                "labels": label_defs
            },
            f,
            indent=2,
        )
    print(ancetors)
    print(t_labels)


if __name__ == "__main__":
    m()
