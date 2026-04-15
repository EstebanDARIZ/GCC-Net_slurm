import json
from copy import deepcopy
from pathlib import Path

# ========= PATHS =========
SRC_JSON = Path(
    '/home/esteban-dreau-darizcuren/doctorat/dataset/'
    'dataset_coco_format/dataset_coco/annotations/instances_val2017.json'
)

DST_JSON = Path(
    '/home/esteban-dreau-darizcuren/doctorat/dataset/'
    'dataset_coco_format/dataset_coco_gccnet/annotations/instances_val2017.json'
)

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


def main():
    with open(SRC_JSON, 'r') as f:
        coco = json.load(f)

    # ========= NOUVELLES CATEGORIES =========
    new_categories = [
        {'id': i + 1, 'name': name}
        for i, name in enumerate(CLASSES)
    ]
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
    used_image_ids = set()

    for ann in coco['annotations']:
        if ann['category_id'] in old_id_to_new_id:
            ann = deepcopy(ann)
            ann['category_id'] = old_id_to_new_id[ann['category_id']]
            ann['id'] = len(new_annotations) + 1  # réindex propre
            new_annotations.append(ann)
            used_image_ids.add(ann['image_id'])

    # ========= FILTRER LES IMAGES SANS ANNOTATION =========
    new_images = [
        img for img in coco['images']
        if img['id'] in used_image_ids
    ]

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


if __name__ == '__main__':
    main()
