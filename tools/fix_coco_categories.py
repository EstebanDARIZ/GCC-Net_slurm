"""
Corrige un fichier COCO JSON exporté depuis YOLO :
  - Remplace les noms de catégories numériques ('0', '1', ...) par les vrais noms de classes
  - Décale les IDs de catégories de 0-based à 1-based (standard COCO)
  - Supprime les catégories sans aucune annotation

Usage:
    python tools/fix_coco_categories.py \
        --input  /Utilisateurs/edreau01/datasets/dataset_test_2.1/dataset_test_2.1/labels_test.json \
        --output /Utilisateurs/edreau01/datasets/dataset_test_2.1/dataset_test_2.1/labels_test.json \
        --classes Bait_1_Squid Bait_2_Sardine Ray Sunfish Pilotfish
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fix COCO JSON category names and IDs')
    parser.add_argument('--input', required=True,
                        help='Input COCO JSON file path')
    parser.add_argument('--output', required=True,
                        help='Output COCO JSON file path (peut être identique à --input)')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Vrais noms de classes dans l\'ordre (index 0, 1, ...)')
    return parser.parse_args()


def fix_categories(input_path, output_path, classes):
    with open(input_path) as f:
        coco = json.load(f)

    annotations = coco.get('annotations', [])
    categories = coco.get('categories', [])

    # --- 1. Identifier les category_ids réellement utilisés dans les annotations ---
    used_ids = set(a['category_id'] for a in annotations)
    print(f'IDs utilisés dans les annotations : {sorted(used_ids)}')

    # --- 2. Construire le mapping : ancien id → (nouvel id 1-based, vrai nom) ---
    # Les catégories sont triées par id pour garantir un ordre stable
    sorted_cats = sorted(categories, key=lambda c: c['id'])

    id_remap = {}  # ancien_id → nouvel_id
    new_categories = []
    new_id = 1

    for cat in sorted_cats:
        old_id = cat['id']
        if old_id not in used_ids:
            print(f'  Catégorie {old_id} ({cat["name"]!r}) ignorée — aucune annotation')
            continue

        # Mapper le nom : si le nom est un entier, remplacer par le vrai nom de classe
        if cat['name'].isdigit():
            idx = int(cat['name'])
            if idx < len(classes):
                new_name = classes[idx]
            else:
                new_name = f'class_{idx}'
                print(f'  [WARN] index {idx} hors de la liste --classes, nom générique utilisé')
        else:
            new_name = cat['name']  # nom déjà correct, on le garde

        id_remap[old_id] = new_id
        new_categories.append({
            'id': new_id,
            'name': new_name,
            'supercategory': cat.get('supercategory', 'object'),
        })
        print(f'  Catégorie {old_id} ({cat["name"]!r}) → id={new_id}, name={new_name!r}')
        new_id += 1

    # --- 3. Mettre à jour les category_id dans les annotations ---
    for ann in annotations:
        old_id = ann['category_id']
        if old_id in id_remap:
            ann['category_id'] = id_remap[old_id]
        else:
            print(f'  [WARN] annotation id={ann["id"]} a un category_id={old_id} inconnu')

    coco['categories'] = new_categories

    # --- 4. Sauvegarder ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f'\nRésultat : {len(new_categories)} catégories, {len(annotations)} annotations')
    print(f'Fichier sauvegardé : {output_path}')


if __name__ == '__main__':
    args = parse_args()
    fix_categories(args.input, args.output, args.classes)
