import json

def update_labels(main_path, other_path):
    with open(main_path, "r") as f:
        main = json.load(f)
        print(f"Main labels: {len(main)}")
    with open(other_path, "r") as f:
        other = json.load(f)
        print(f"Other labels: {len(other)}")
    overlap = set(main) & set(other)
    print(f"Overlapping keys: {len(overlap)}")
    main.update(other)
    print(f"Combined labels: {len(main)}")
    with open(main_path, "w") as f:
        json.dump(main, f, indent=2)


if __name__ == "__main__":
    update_labels("/storage/lepisea-annotation-2/annflux/labels.json",
    "/code/src/annflux/projects/lepisea_june/labels_other.json")
    pass
