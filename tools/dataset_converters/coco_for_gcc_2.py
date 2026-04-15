import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

"""
Convert COCO dataset annotations to GCC-Net format by filtering classes
and remapping category IDs.

- Keeps images with 0 annotations (background images).
- Remaps category IDs and reindexes annotation IDs.
- Creates a lightweight copy of images into a target folder:
    prefer hardlink -> fallback symlink -> fallback copy
"""

# ========= PATHS =========
SRC_JSON = Path(
    '/Utilisateurs/edreau01/datasets/datasets_coco/'
    'dataset_coco/annotations/instances_train2017.json'
)

DST_JSON = Path(
    '/Utilisateurs/edreau01/datasets/datasets_coco/'
    'dataset_coco_gccnet/annotations/instances_train2017.json'
)

# Folder that contains the source images referenced by the COCO JSON.
# IMPORTANT: In COCO, file_name is usually relative to this folder.
SRC_IMG_DIR = Path(
    '/Utilisateurs/edreau01/datasets/datasets_coco/'
    'dataset_coco/train2017'
)

# Folder where you want the "lightweight copied" images to be placed.
DST_IMG_DIR = Path(
    '/Utilisateurs/edreau01/datasets/datasets_coco/'
    'dataset_coco_gccnet/train2017'
)

# Strategy: "hardlink" (recommended), "symlink", or "copy"
COPY_MODE = "hardlink"

# If True, fail fast when an image file is missing. If False, skip missing ones.
STRICT_MISSING_IMAGES = True

# ========= CLASSES GCC-Net =========
CLASSES = [
    'Bait_1_Squid',
    'Bait_2_Sardine',
    'Ray',
    'Sunfish',
    'Pilotfish'
]

# mapping ancien nom → nouveau nom
OLD_TO_NEW = {
    'class_0': 'Bait_1_Squid',
    'class_1': 'Bait_2_Sardine',
    'class_2': 'Ray',
    'class_3': 'Sunfish',
    'class_4': 'Pilotfish'
}


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """
    Create a lightweight copy of src -> dst.
    Returns the method used: "hardlink" / "symlink" / "copy" / "skipped".
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If already exists, do nothing
    if dst.exists():
        return "skipped"

    try:
        if mode == "hardlink":
            os.link(src, dst)  # hardlink
            return "hardlink"
        elif mode == "symlink":
            os.symlink(src, dst)  # symlink
            return "symlink"
        elif mode == "copy":
            shutil.copy2(src, dst)
            return "copy"
        else:
            raise ValueError(f"Unknown COPY_MODE={mode}")
    except OSError:
        # Fallback order: hardlink -> symlink -> copy
        if mode == "hardlink":
            try:
                os.symlink(src, dst)
                return "symlink"
            except OSError:
                shutil.copy2(src, dst)
                return "copy"
        if mode == "symlink":
            shutil.copy2(src, dst)
            return "copy"
        raise


def main():
    with open(SRC_JSON, 'r') as f:
        coco = json.load(f)

    # ========= NOUVELLES CATEGORIES =========
    new_categories = [{'id': i + 1, 'name': name} for i, name in enumerate(CLASSES)]
    name_to_new_id = {c['name']: c['id'] for c in new_categories}

    # ancien id → ancien nom
    old_id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # ancien id → nouveau id
    old_id_to_new_id = {}
    for old_id, old_name in old_id_to_name.items():
        if old_name in OLD_TO_NEW:
            new_name = OLD_TO_NEW[old_name]
            old_id_to_new_id[old_id] = name_to_new_id[new_name]

    # ========= FILTRER & REMAPPER ANNOTATIONS =========
    new_annotations = []
    for ann in coco['annotations']:
        if ann['category_id'] in old_id_to_new_id:
            ann2 = deepcopy(ann)
            ann2['category_id'] = old_id_to_new_id[ann2['category_id']]
            ann2['id'] = len(new_annotations) + 1  # réindex propre
            new_annotations.append(ann2)

    # ========= GARDER TOUTES LES IMAGES (y compris background) =========
    # (Optionnel) on peut aussi réindexer les images si tu veux "clean" les ids,
    # mais MMDet s'en fiche tant qu'elles sont uniques.
    new_images = deepcopy(coco['images'])

    # ========= COPIE LÉGÈRE DES IMAGES + option: réécrire file_name =========
    # On va :
    # - copier/linker chaque image SRC_IMG_DIR/file_name -> DST_IMG_DIR/file_name
    # - garder file_name identique (simple et COCO-compatible)
    copied = {"hardlink": 0, "symlink": 0, "copy": 0, "skipped": 0}
    missing = []

    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    for img in new_images:
        fn = img.get("file_name")
        if not fn:
            continue

        src_path = SRC_IMG_DIR / fn
        dst_path = DST_IMG_DIR / fn

        if not src_path.exists():
            missing.append(str(src_path))
            if STRICT_MISSING_IMAGES:
                raise FileNotFoundError(f"Missing image file: {src_path}")
            else:
                # Option: keep image entry but it will fail at training time.
                # Better: drop the image entry if missing.
                continue

        method = link_or_copy(src_path, dst_path, COPY_MODE)
        copied[method] = copied.get(method, 0) + 1

        # If you want the JSON to point to a different relative path, you can
        # modify img["file_name"] here. Usually not needed if you set img_prefix
        # to DST_IMG_DIR in your config.

    # ========= ÉCRITURE DU NOUVEAU JSON =========
    coco_out = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories
    }

    DST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DST_JSON, 'w') as f:
        json.dump(coco_out, f, indent=2)

    print('✔ Nouveau JSON GCC-Net créé')
    print(f'✔ Images conservées      : {len(new_images)}')
    print(f'✔ Annotations conservées : {len(new_annotations)}')
    print(f'✔ Classes finales        : {len(new_categories)}')
    print('✔ Images link/copy summary:', copied)
    if not STRICT_MISSING_IMAGES and missing:
        print(f'⚠ Missing images skipped: {len(missing)} (showing up to 5)')
        for p in missing[:5]:
            print('   -', p)


if __name__ == '__main__':
    main()
