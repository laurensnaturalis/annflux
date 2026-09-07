import json

def update_label_defs(main_path, other_path):
    with open(main_path, "r") as f:
        main = json.load(f)
        print(f"Main labels: {len(main['labels'])}")
    with open(other_path, "r") as f:
        other = json.load(f)
        print(f"Other labels: {len(other['labels'])}")
    main["labels"].extend(other["labels"])
    seen = set()
    unique = []
    for item in main["labels"]:
        key = tuple(item)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    main["labels"] = [list(item) for item in unique]
    print(f"Combined labels: {len(main['labels'])}")
    with open(main_path, "w") as f:
        json.dump(main, f, indent=2)


if __name__ == "__main__":
    update_label_defs("/storage/lepisea-annotation-2/annflux/label_defs.json", 
    "/code/src/annflux/projects/lepisea_june/label_defs_other.json")
    pass