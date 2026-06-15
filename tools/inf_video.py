# Inférence GCC-Net sur toutes les vidéos d'un dossier (récursif)
# Sortie : un CSV par vidéo, écrit à côté de la vidéo

"""
Voici ce qui a changé :

Arguments : video + output_dir → input_dir (dossier racine, parcouru récursivement)

Nouveau comportement :

Trouve toutes les vidéos (.mp4, .MP4, .avi, .mov, .mkv) dans le dossier et ses sous-dossiers
Écrit le CSV à côté de chaque vidéo, même nom, extension .csv
Plus aucun code d'images annotées
Nouveau flag : --skip-existing pour reprendre une inférence interrompue sans retraiter les vidéos déjà faites

Exemple d'utilisation :


python tools/inf_video.py \
  configs/autoassign/autoassign_gcc_sam_2.0.py \
  work_dirs/autoassign_gcc_sam_2.0/epoch_30.pth \
  /Utilisateurs/edreau01/datasets/BORIS \
  --score-thr 0.3 --frame-step 5 --skip-existing
"""


import argparse
import os
import os.path as osp
import csv
import time

import cv2
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector

VIDEO_EXTENSIONS = {'.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inférence GCC-Net sur toutes les vidéos d\'un dossier')
    parser.add_argument('config',     help='Chemin vers le fichier de config (.py)')
    parser.add_argument('checkpoint', help='Chemin vers le checkpoint (.pth)')
    parser.add_argument('input_dir',  help='Dossier racine contenant les vidéos (parcouru récursivement)')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Seuil de confiance (défaut: 0.3)')
    parser.add_argument(
        '--frame-step',
        type=int,
        default=1,
        help='Analyser 1 frame toutes les N frames (défaut: 1 = toutes)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='ID du GPU (défaut: 0)')
    parser.add_argument(
        '--device',
        default=None,
        help='Device : "cpu", "cuda:0", etc. Écrase --gpu-id si spécifié.')
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Désactiver la confirmation temporelle.')
    parser.add_argument(
        '--confirm-frames',
        type=int,
        default=5,
        metavar='X',
        help='Frames avant/après à analyser pour confirmer une détection (défaut: 5).')
    parser.add_argument(
        '--confirm-thr',
        type=int,
        default=3,
        metavar='T',
        help='Confirmations minimum requises parmi les 2X frames voisines (défaut: 3).')
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        default=None,
        metavar='CLASS_ID',
        help='Ne garder que ces class_id (ex: --classes 1 5). Sans ce paramètre, toutes les classes.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Ignorer les vidéos dont le CSV existe déjà.')
    return parser.parse_args()


def find_videos(root_dir):
    videos = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if osp.splitext(fname)[1] in VIDEO_EXTENSIONS:
                videos.append(osp.join(dirpath, fname))
    return sorted(videos)


def load_model(config_path, checkpoint_path, device):
    print(f"[INFO] Chargement du modèle sur {device}...")
    cfg = Config.fromfile(config_path)

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.get('backbone', {}):
        cfg.model.backbone.init_cfg = None

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print("[WARN] Pas de CLASSES dans le checkpoint, noms génériques utilisés.")
        model.CLASSES = [f'class_{i}' for i in range(100)]

    model.cfg = cfg
    model.to(device)
    model.eval()

    print(f"[INFO] Modèle chargé. Classes : {model.CLASSES}")
    return model


def frame_to_timecode(frame_idx, fps):
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def confirm_detection(model, cap, frame_idx, class_id, score_thr, window, min_confirmations):
    if window == 0:
        return True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    confirmations = 0

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        check_idx = frame_idx + offset
        if check_idx < 0 or check_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
        ret, check_frame = cap.read()
        if not ret:
            continue

        with torch.no_grad(), torch.cuda.amp.autocast():
            check_result = inference_detector(model, check_frame)

        if class_id < len(check_result):
            for bbox in check_result[class_id]:
                if float(bbox[4]) >= score_thr:
                    confirmations += 1
                    break

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
    return confirmations >= min_confirmations


def run_inference(model, video_path, score_thr, frame_step, device,
                  allowed_classes=None, confirm_frames=5, confirm_thr=3):

    csv_path = osp.splitext(video_path)[0] + '.csv'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {video_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[INFO] Vidéo       : {video_path}")
    print(f"[INFO] FPS         : {fps:.2f}  |  Frames : {total_frames}  |  Step : {frame_step}  |  Seuil : {score_thr}")
    if allowed_classes is not None:
        names = [model.CLASSES[i] for i in allowed_classes if i < len(model.CLASSES)]
        print(f"[INFO] Classes     : {allowed_classes} ({', '.join(names)})")
    if confirm_frames > 0:
        print(f"[INFO] Confirmation: ±{confirm_frames} frames, min {confirm_thr} requises")

    with open(csv_path, 'w', newline='') as csvfile:
        csvfile.write(f'# video_path: {osp.abspath(video_path)}\n')
        writer = csv.writer(csvfile)
        writer.writerow([
            'frame_idx', 'timecode', 'class_id', 'class_name',
            'score', 'x1', 'y1', 'x2', 'y2', 'inference_time_ms'
        ])

        frame_idx        = 0
        analyzed         = 0
        total_detections = 0
        inference_times  = []
        frames_to_process = (total_frames + frame_step - 1) // frame_step
        t_start          = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                analyzed += 1
                timecode = frame_to_timecode(frame_idx, fps)

                t0 = time.perf_counter()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    result = inference_detector(model, frame)
                inf_ms = (time.perf_counter() - t0) * 1000
                inference_times.append(inf_ms)

                candidates = []
                for class_id, bboxes in enumerate(result):
                    if allowed_classes is not None and class_id not in allowed_classes:
                        continue
                    class_name = model.CLASSES[class_id] if class_id < len(model.CLASSES) else f'class_{class_id}'
                    for bbox in bboxes:
                        score = float(bbox[4])
                        if score >= score_thr:
                            candidates.append((class_id, class_name, score,
                                               float(bbox[0]), float(bbox[1]),
                                               float(bbox[2]), float(bbox[3])))

                confirmed_classes = set()
                for class_id, *_ in candidates:
                    if class_id not in confirmed_classes:
                        if confirm_detection(model, cap, frame_idx, class_id,
                                             score_thr, confirm_frames, confirm_thr):
                            confirmed_classes.add(class_id)

                n_det = 0
                for class_id, class_name, score, x1, y1, x2, y2 in candidates:
                    if class_id in confirmed_classes:
                        writer.writerow([
                            frame_idx, timecode, class_id, class_name,
                            f'{score:.4f}',
                            f'{x1:.1f}', f'{y1:.1f}', f'{x2:.1f}', f'{y2:.1f}',
                            f'{inf_ms:.1f}'
                        ])
                        n_det += 1

                if n_det > 0:
                    total_detections += n_det
                    print(f"  [frame {frame_idx:06d} | {timecode}] {n_det} détection(s)  ({inf_ms:.0f} ms)")

                if analyzed % 100 == 0:
                    print(f"[INFO] Progression : {analyzed}/{frames_to_process} frames analysées...")

            frame_idx += 1

        total_s = time.perf_counter() - t_start
        avg_ms  = sum(inference_times) / len(inference_times) if inference_times else 0

        csvfile.write(f'# total_time_s: {total_s:.2f}\n')
        csvfile.write(f'# avg_inference_ms: {avg_ms:.1f}\n')
        csvfile.write(f'# min_inference_ms: {min(inference_times) if inference_times else 0:.1f}\n')
        csvfile.write(f'# max_inference_ms: {max(inference_times) if inference_times else 0:.1f}\n')

    cap.release()

    h, m = divmod(int(total_s), 3600)
    m, s = divmod(m, 60)
    print(f"[✓] {analyzed} frames, {total_detections} détection(s) — {h:02d}h{m:02d}m{s:02d}s — CSV : {csv_path}")


def main():
    args = parse_args()

    assert osp.isfile(args.config),     f"Config introuvable : {args.config}"
    assert osp.isfile(args.checkpoint), f"Checkpoint introuvable : {args.checkpoint}"
    assert osp.isdir(args.input_dir),   f"Dossier introuvable : {args.input_dir}"

    device = args.device if args.device else f'cuda:{args.gpu_id}'
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA non disponible, basculement sur CPU.")
        device = 'cpu'

    videos = find_videos(args.input_dir)
    if not videos:
        print(f"[WARN] Aucune vidéo trouvée dans : {args.input_dir}")
        return

    print(f"[INFO] {len(videos)} vidéo(s) trouvée(s) dans {args.input_dir}")

    if args.skip_existing:
        videos = [v for v in videos if not osp.exists(osp.splitext(v)[0] + '.csv')]
        print(f"[INFO] {len(videos)} vidéo(s) à traiter après filtrage des CSV existants")

    if not videos:
        print("[INFO] Toutes les vidéos ont déjà un CSV. Rien à faire.")
        return

    model = load_model(args.config, args.checkpoint, device)

    confirm_frames = 0 if args.no_confirm else args.confirm_frames

    for i, video_path in enumerate(videos, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}] {osp.basename(video_path)}")
        print('='*60)
        run_inference(
            model, video_path, args.score_thr, args.frame_step, device,
            allowed_classes=args.classes,
            confirm_frames=confirm_frames,
            confirm_thr=args.confirm_thr
        )

    print(f"\n[✓] Inférence terminée sur {len(videos)} vidéo(s).")


if __name__ == '__main__':
    main()
