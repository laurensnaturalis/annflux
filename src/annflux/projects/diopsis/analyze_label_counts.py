import json
from collections import Counter


def m():
    j_ann = json.load(open("/mnt/big/Projects/diopsis_annotation_2/annflux/labels.json"))

    for key, count in Counter(j_ann.values()).most_common():
        print(key, count)


if __name__ == '__main__':
    m()