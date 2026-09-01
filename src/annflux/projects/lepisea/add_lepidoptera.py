"""
Adds 'Lepidoptera' as a root label to the lepisea project:
1. Inserts ["Lepidoptera", "null"] into label_defs.json (if not already present).
2. Sets the parent of every current root label to "Lepidoptera" in label_defs.json.
3. Adds "Lepidoptera" to every row in labels.json that does not already have it.
"""
import json
import os

PROJECT = "/mnt/big/indeed/lepisea/annflux"
LABEL_DEFS_PATH = os.path.join(PROJECT, "label_defs.json")
LABELS_PATH = os.path.join(PROJECT, "labels.json")
PARENT = "Lepidoptera"


def main():
    # ── label_defs.json ──────────────────────────────────────────────────────
    with open(LABEL_DEFS_PATH) as f:
        defs = json.load(f)

    entries = defs["labels"]

    # Collect current root labels (parent == "null"), excluding Lepidoptera itself
    roots = [e[0] for e in entries if e[1] == "null" and e[0] != PARENT]
    print(f"Current root labels ({len(roots)}): {roots}")

    # Ensure Lepidoptera itself is present as a root
    if not any(e[0] == PARENT for e in entries):
        entries.insert(0, [PARENT, "null"])
        print(f"Inserted [{PARENT}, null] into label_defs")
    else:
        # Make sure it's a root
        for e in entries:
            if e[0] == PARENT:
                e[1] = "null"
        print(f"{PARENT} already present")

    # Re-parent all current roots to Lepidoptera
    for e in entries:
        if e[0] in roots:
            e[1] = PARENT

    with open(LABEL_DEFS_PATH, "w") as f:
        json.dump(defs, f, indent=2, ensure_ascii=False)
    print(f"Updated {LABEL_DEFS_PATH}: {len(roots)} families now point to {PARENT}")

    # ── labels.json ──────────────────────────────────────────────────────────
    if not os.path.exists(LABELS_PATH):
        print(f"{LABELS_PATH} not found, skipping label update")
        return

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    updated = 0
    for uid, lbl_str in labels.items():
        lbls = [lbl.strip() for lbl in lbl_str.split(",") if lbl.strip()] if lbl_str else []
        if PARENT not in lbls:
            lbls.append(PARENT)
            labels[uid] = ",".join(lbls)
            updated += 1

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"Added {PARENT} to {updated} / {len(labels)} labeled images in {LABELS_PATH}")


if __name__ == "__main__":
    main()
