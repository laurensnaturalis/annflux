import json


def m():
    defs = json.load(open("/mnt/big/indeed/legasea_big/annflux/label_defs.json"))

    seen = set()
    defs_new = []
    for label in defs['labels']:
        if label[0] not in seen:
            seen.add(label[0])
            defs_new.append(label)

    with open("/mnt/big/indeed/legasea_big/annflux/label_defs.json", 'w') as f:
        json.dump({"labels": defs_new}, f)


if __name__ == '__main__':
    m()