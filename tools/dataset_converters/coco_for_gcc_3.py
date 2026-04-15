import json
from copy import deepcopy
from pathlib import Path
from collections import Counter

SRC_JSON = Path("/Utilisateurs/edreau01/datasets/datasets_coco/dataset_coco/annotations/instances_val2017.json")
DST_JSON = Path("/Utilisateurs/edreau01/datasets/datasets_coco/dataset_coco_gccnet/annotations/instances_val2017.json")

CLASSES = [
    "Bait_1_Squid",
    "Bait_2_Sardine",
    "Ray",
    "Sunfish",
    "Pilotfish",
]

# ✅ À ADAPTER : old category_id (dans ton COCO source) -> nom final GCC
OLD_ID_TO_NEW_NAME = {
    1: "Bait_1_Squid",
    2: "Bait_2_Sardine",   # <-- ajoute ça si c'est bien ça
    3: "Ray",
    4: "Sunfish",
    5: "Pilotfish",
}

def main():
    coco = json.loads(SRC_JSON.read_text(encoding="utf-8"))

    # ---- Debug source
    old_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    src_counts = Counter(a["category_id"] for a in coco["annotations"])
    print("SOURCE: counts by old category_id:")
    for cid in sorted(old_id_to_name):
        print(f"  {cid} ({old_id_to_name[cid]}): {src_counts.get(cid, 0)}")

    # ---- New categories
    new_categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    name_to_new_id = {c["name"]: c["id"] for c in new_categories}

    # old id -> new id
    old_id_to_new_id = {}
    for old_id, new_name in OLD_ID_TO_NEW_NAME.items():
        if new_name not in name_to_new_id:
            raise ValueError(f"{new_name} pas dans CLASSES")
        old_id_to_new_id[old_id] = name_to_new_id[new_name]

    # ---- Convert annotations
    new_annotations = []
    for ann in coco["annotations"]:
        old_cat = ann["category_id"]
        if old_cat in old_id_to_new_id:
            ann2 = deepcopy(ann)
            ann2["category_id"] = old_id_to_new_id[old_cat]
            ann2["id"] = len(new_annotations) + 1
            new_annotations.append(ann2)

    # ---- Keep all images (including background)
    new_images = coco["images"][:]

    coco_out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }

    DST_JSON.parent.mkdir(parents=True, exist_ok=True)
    DST_JSON.write_text(json.dumps(coco_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Sanity checks
    new_counts = Counter(a["category_id"] for a in new_annotations)
    print("\nOUTPUT: counts by new category_id (1..5):", dict(sorted(new_counts.items())))

    # classes sans instances
    print("\nOUTPUT: counts by class name:")
    for c in new_categories:
        cid = c["id"]
        print(f"  {cid} {c['name']}: {new_counts.get(cid, 0)}")

    # ids source ignorés
    ignored = sorted(set(src_counts.keys()) - set(old_id_to_new_id.keys()))
    if ignored:
        print("\n⚠️ WARNING: old category_id présents dans SOURCE mais ignorés par le mapping:", ignored)

if __name__ == "__main__":
    main()
