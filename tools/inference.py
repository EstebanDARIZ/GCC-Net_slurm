# Inférence GCC-Net sur un dossier d'images
# Sorties : images annotées (bounding boxes) + CSV des détections + labels YOLO

import argparse
import os
import os.path as osp
import csv
from pathlib import Path

import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from PIL import Image

# Extensions d'images supportées
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inférence GCC-Net sur un dossier d\'images')
    parser.add_argument('config',      help='Chemin vers le fichier de config (.py)')
    parser.add_argument('checkpoint',  help='Chemin vers le checkpoint (.pth)')
    parser.add_argument('input_dir',   help='Dossier contenant les images à traiter')
    parser.add_argument('output_dir',  help='Dossier de sortie pour les résultats')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Seuil de confiance pour filtrer les détections (défaut: 0.3)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='ID du GPU à utiliser (défaut: 0)')
    parser.add_argument(
        '--no-annotated-imgs',
        action='store_true',
        help='Désactiver la sauvegarde des images annotées')
    return parser.parse_args()


def load_model(config_path, checkpoint_path, gpu_id=0):
    """Charge le modèle GCC-Net depuis la config et le checkpoint."""
    device = f'cuda:{gpu_id}'
    print(f"[INFO] Chargement du modèle sur {device}...")

    cfg = Config.fromfile(config_path)

    # Désactiver les poids pré-entraînés (on charge le checkpoint)
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.get('backbone', {}):
        cfg.model.backbone.init_cfg = None

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Récupérer les noms de classes depuis le checkpoint si disponible
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # Fallback : noms génériques
        print("[WARN] Pas de CLASSES dans le checkpoint, utilisation de noms génériques.")
        model.CLASSES = [f'class_{i}' for i in range(100)]

    model.cfg = cfg
    model.to(device)
    model.eval()

    print(f"[INFO] Modèle chargé. Classes : {model.CLASSES}")
    return model, device


def get_image_paths(input_dir):
    """Retourne la liste de toutes les images dans le dossier (récursif)."""
    img_paths = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            if fname.lower().endswith(IMG_EXTENSIONS):
                img_paths.append(osp.join(root, fname))
    return img_paths


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convertit des coordonnées absolues [x1, y1, x2, y2] en format YOLO normalisé
    [cx, cy, w, h] avec des valeurs dans [0, 1].
    """
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def run_inference(model, img_paths, output_dir, score_thr, device, save_imgs=True):
    """
    Lance l'inférence sur toutes les images et sauvegarde :
      - Les images annotées avec les bounding boxes
      - Un fichier CSV récapitulatif des détections
      - Un fichier .txt par image au format YOLO (class_id cx cy w h score)
    """
    annotated_dir = osp.join(output_dir, 'annotated_images')
    yolo_dir      = osp.join(output_dir, 'yolo_labels')

    if save_imgs:
        os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)

    csv_path = osp.join(output_dir, 'detections.csv')

    total = len(img_paths)
    print(f"[INFO] {total} image(s) trouvée(s) dans le dossier d'entrée.")
    print(f"[INFO] Seuil de confiance : {score_thr}")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'image_name', 'class_id', 'class_name',
            'score', 'x1', 'y1', 'x2', 'y2'
        ])

        for idx, img_path in enumerate(img_paths):
            img_name  = osp.basename(img_path)
            img_stem  = osp.splitext(img_name)[0]   # nom sans extension
            print(f"[{idx+1}/{total}] Traitement : {img_name}", end=' ... ')

            # --- Dimensions de l'image (nécessaires pour la normalisation YOLO) ---
            pil_img = Image.open(img_path)
            img_w, img_h = pil_img.size

            # --- Inférence ---
            with torch.no_grad():
                result = inference_detector(model, img_path)

            # --- Écriture CSV + labels YOLO ---
            # result est une liste de tableaux numpy, un par classe
            # shape de chaque tableau : (N, 5) avec [x1, y1, x2, y2, score]
            n_detections = 0
            yolo_lines   = []

            for class_id, bboxes in enumerate(result):
                class_name = model.CLASSES[class_id] if class_id < len(model.CLASSES) else f'class_{class_id}'
                for bbox in bboxes:
                    score = float(bbox[4])
                    if score >= score_thr:
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

                        # CSV
                        writer.writerow([
                            img_name, class_id, class_name,
                            f'{score:.4f}',
                            f'{x1:.1f}', f'{y1:.1f}', f'{x2:.1f}', f'{y2:.1f}'
                        ])

                        # Format YOLO : class_id cx cy w h score
                        cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                        yolo_lines.append(
                            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.4f}"
                        )

                        n_detections += 1

            # Sauvegarde du fichier .txt YOLO (vide si aucune détection)
            yolo_txt_path = osp.join(yolo_dir, f"{img_stem}.txt")
            with open(yolo_txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                if yolo_lines:
                    f.write('\n')

            # --- Image annotée ---
            if save_imgs:
                out_img_path = osp.join(annotated_dir, img_name)
                model.show_result(
                    img_path,
                    result,
                    score_thr=score_thr,
                    show=False,
                    out_file=out_img_path
                )

            print(f"{n_detections} détection(s)")

    print(f"\n[✓] CSV sauvegardé      : {csv_path}")
    print(f"[✓] Labels YOLO         : {yolo_dir}")
    if save_imgs:
        print(f"[✓] Images annotées     : {annotated_dir}")
    print("[✓] Inférence terminée !")


def main():
    args = parse_args()

    # Vérifications
    assert osp.isfile(args.config),     f"Config introuvable : {args.config}"
    assert osp.isfile(args.checkpoint), f"Checkpoint introuvable : {args.checkpoint}"
    assert osp.isdir(args.input_dir),   f"Dossier d'entrée introuvable : {args.input_dir}"
    assert torch.cuda.is_available(),   "CUDA non disponible ! Vérifiez votre installation PyTorch/CUDA."

    os.makedirs(args.output_dir, exist_ok=True)

    # Chargement du modèle
    model, device = load_model(args.config, args.checkpoint, args.gpu_id)

    # Récupération des images
    img_paths = get_image_paths(args.input_dir)
    if not img_paths:
        print(f"[ERREUR] Aucune image trouvée dans : {args.input_dir}")
        return

    # Inférence
    save_imgs = not args.no_annotated_imgs
    run_inference(model, img_paths, args.output_dir, args.score_thr, device, save_imgs)


if __name__ == '__main__':
    main()