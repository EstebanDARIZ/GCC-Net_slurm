"""
Convertit un dataset YOLO (images/ + labels/*.txt) en fichier COCO JSON.

Usage:
    python tools/convert_yolo_to_coco.py \
        --img-dir  /Utilisateurs/edreau01/datasets/dataset_test_3.0/images \
        --lbl-dir  /Utilisateurs/edreau01/datasets/dataset_test_3.0/labels \
        --classes  Bait_1_Squid Bait_2_Sardine Ray Sunfish Pilotfish Shark Jellyfish Fish \
        --output   /Utilisateurs/edreau01/datasets/dataset_test_3.0/labels_test.json
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert YOLO labels to COCO JSON')
    parser.add_argument('--img-dir', required=True,
                        help='Directory containing images')
    parser.add_argument('--lbl-dir', required=True,
                        help='Directory containing YOLO .txt label files')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Ordered list of class names (index 0, 1, ...)')
    parser.add_argument('--output', required=True,
                        help='Output JSON file path')
    return parser.parse_args()


def yolo_to_coco(img_dir, lbl_dir, classes, output_path):
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

    categories = [{'id': i + 1, 'name': name, 'supercategory': 'object'}
                  for i, name in enumerate(classes)]

    images = []
    annotations = []
    ann_id = 1

    img_files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    for img_id, img_path in enumerate(img_files, start=1):
        # Lire les dimensions de l'image
        with Image.open(img_path) as img:
            width, height = img.size

        images.append({
            'id': img_id,
            'file_name': img_path.name,
            'width': width,
            'height': height,
        })

        # Chercher le fichier label correspondant
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                print(f'  [WARN] ligne ignorée dans {lbl_path.name}: {line!r}')
                continue

            cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])

            # YOLO → COCO (coordonnées absolues, coin haut-gauche)
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            abs_w = bw * width
            abs_h = bh * height

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cls_id + 1,  # COCO ids commencent à 1
                'bbox': [round(x1, 2), round(y1, 2),
                         round(abs_w, 2), round(abs_h, 2)],
                'area': round(abs_w * abs_h, 2),
                'iscrowd': 0,
            })
            ann_id += 1

    coco = {
        'info': {'description': 'Converted from YOLO format'},
        'categories': categories,
        'images': images,
        'annotations': annotations,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f'Done: {len(images)} images, {len(annotations)} annotations')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    args = parse_args()
    yolo_to_coco(args.img_dir, args.lbl_dir, args.classes, args.output)
