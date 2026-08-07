"""
Convertit un dataset COCO (train2017/ + val2017/ + annotations/instances_*.json)
en dataset YOLO (images/{train,val} + labels/{train,val}), à côté du format COCO.

Usage:
    python tools/convert_coco_to_yolo.py \
        --coco-root   /Utilisateurs/edreau01/datasets/dataset_sam_2.0/coco_format \
        --output-root /Utilisateurs/edreau01/datasets/dataset_sam_2.0/yolo_format
"""

import argparse
import json
import shutil
from pathlib import Path

# nom du split COCO -> nom du split YOLO (attendu par les data_sam_*.yaml d'ultralytics)
SPLITS = {'train2017': 'train', 'val2017': 'val'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a COCO dataset to YOLO format')
    parser.add_argument('--coco-root', required=True,
                         help='Root of the coco_format dataset '
                              '(contains annotations/, train2017/, val2017/)')
    parser.add_argument('--output-root', required=True,
                         help='Root of the yolo_format dataset to create')
    parser.add_argument('--symlink', action='store_true',
                         help='Symlink images instead of copying them '
                              '(faster, saves disk space)')
    return parser.parse_args()


def convert_split(coco_root, output_root, coco_split, yolo_split, symlink):
    ann_path = coco_root / 'annotations' / f'instances_{coco_split}.json'
    if not ann_path.exists():
        print(f'[skip] {ann_path} introuvable')
        return

    with open(ann_path) as f:
        coco = json.load(f)

    # category_id (1-based, contigu) -> class_id YOLO (0-based)
    cat_to_yolo = {c['id']: c['id'] - 1 for c in coco['categories']}

    anns_by_image = {}
    for a in coco['annotations']:
        anns_by_image.setdefault(a['image_id'], []).append(a)

    out_img_dir = output_root / 'images' / yolo_split
    out_lbl_dir = output_root / 'labels' / yolo_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    in_img_dir = coco_root / coco_split

    for im in coco['images']:
        width, height = im['width'], im['height']
        src = in_img_dir / im['file_name']
        dst = out_img_dir / im['file_name']
        if not dst.exists():
            if symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        lines = []
        for a in anns_by_image.get(im['id'], []):
            x, y, w, h = a['bbox']
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            cls = cat_to_yolo[a['category_id']]
            lines.append(f'{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        lbl_path = out_lbl_dir / (Path(im['file_name']).stem + '.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')

    print(f'[{yolo_split}] images: {len(coco["images"])}, '
          f'annotations: {len(coco["annotations"])}')


def main():
    args = parse_args()
    coco_root = Path(args.coco_root)
    output_root = Path(args.output_root)

    for coco_split, yolo_split in SPLITS.items():
        convert_split(coco_root, output_root, coco_split, yolo_split, args.symlink)

    print(f'Done. Nouveau dataset YOLO écrit dans {output_root}')


if __name__ == '__main__':
    main()
