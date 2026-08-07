"""
Crée une copie filtrée d'un dataset COCO en retirant une classe et toutes les
images qui la contiennent.

Usage:
    python tools/filter_dataset_class.py \
        --input-root  /Utilisateurs/edreau01/datasets/dataset_sam_2.0/coco_format \
        --output-root /Utilisateurs/edreau01/datasets/dataset_sam_2.2/coco_format \
        --exclude-class Tuna
"""

import argparse
import json
import shutil
from pathlib import Path

SPLITS = ['train2017', 'val2017']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter a COCO dataset by removing one class and every '
                     'image that contains it')
    parser.add_argument('--input-root', required=True,
                         help='Root of the source coco_format dataset '
                              '(contains annotations/, train2017/, val2017/)')
    parser.add_argument('--output-root', required=True,
                         help='Root of the new coco_format dataset to create')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exclude-class', help='Category name to remove')
    group.add_argument('--exclude-class-id', type=int,
                        help='Category id (as in the COCO json) to remove')
    parser.add_argument('--symlink', action='store_true',
                         help='Symlink images instead of copying them '
                              '(faster, saves disk space)')
    return parser.parse_args()


def filter_split(input_root, output_root, split, exclude_class, exclude_class_id, symlink):
    ann_path = input_root / 'annotations' / f'instances_{split}.json'
    if not ann_path.exists():
        print(f'[skip] {ann_path} introuvable')
        return

    with open(ann_path) as f:
        coco = json.load(f)

    categories = coco['categories']
    if exclude_class_id is not None:
        excluded = next((c for c in categories if c['id'] == exclude_class_id), None)
    else:
        excluded = next((c for c in categories if c['name'] == exclude_class), None)

    if excluded is None:
        raise ValueError(f'Classe à exclure introuvable dans {ann_path} '
                          f'(exclude_class={exclude_class!r}, '
                          f'exclude_class_id={exclude_class_id!r})')

    excluded_id = excluded['id']
    print(f'[{split}] Exclusion de la catégorie {excluded["id"]} ({excluded["name"]})')

    # Images contenant la classe exclue -> on ne les transfère pas du tout
    excluded_image_ids = {a['image_id'] for a in coco['annotations']
                           if a['category_id'] == excluded_id}

    kept_images = [im for im in coco['images'] if im['id'] not in excluded_image_ids]
    kept_image_ids = {im['id'] for im in kept_images}

    kept_annotations = [a for a in coco['annotations']
                         if a['image_id'] in kept_image_ids
                         and a['category_id'] != excluded_id]

    # Renumérote les catégories restantes pour rester contiguës (1..N)
    remaining_categories = [c for c in categories if c['id'] != excluded_id]
    old_to_new_id = {c['id']: new_id for new_id, c in enumerate(remaining_categories, start=1)}
    new_categories = [{**c, 'id': old_to_new_id[c['id']]} for c in remaining_categories]
    for a in kept_annotations:
        a['category_id'] = old_to_new_id[a['category_id']]

    new_coco = {
        'info': coco.get('info', {}),
        'categories': new_categories,
        'images': kept_images,
        'annotations': kept_annotations,
    }

    out_img_dir = output_root / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    in_img_dir = input_root / split

    n_copied = 0
    for im in kept_images:
        src = in_img_dir / im['file_name']
        dst = out_img_dir / im['file_name']
        if dst.exists():
            continue
        if symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        n_copied += 1

    out_ann_path = output_root / 'annotations' / f'instances_{split}.json'
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann_path, 'w') as f:
        json.dump(new_coco, f)

    print(f'[{split}] images: {len(coco["images"])} -> {len(kept_images)} '
          f'({len(excluded_image_ids)} retirées car contenant {excluded["name"]}), '
          f'fichiers transférés: {n_copied}')
    print(f'[{split}] annotations: {len(coco["annotations"])} -> {len(kept_annotations)}')


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for split in SPLITS:
        filter_split(input_root, output_root, split,
                     args.exclude_class, args.exclude_class_id, args.symlink)

    print(f'Done. Nouveau dataset écrit dans {output_root}')


if __name__ == '__main__':
    main()
