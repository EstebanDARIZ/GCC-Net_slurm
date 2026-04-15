#!/usr/bin/env python3
import argparse
import torch
from effdet import create_model
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import time

"""
Inference script for EfficientDet object detection model using PyTorch (Slurm-friendly).
Fixes:
- No local import (no `from config import CLASS_NAMES`)
- Classes passed via --classes or --classes-file (fallback: class_0..class_{n-1})
- Saves visualizations + a text summary
- Prints per-image inference time + global FPS (average over all images)
"""

def load_class_names(classes_arg=None, classes_file=None, num_classes=None):
    if classes_file:
        with open(classes_file, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        return names

    if classes_arg:
        names = [c.strip() for c in classes_arg.split(",") if c.strip()]
        return names

    if num_classes is None:
        raise ValueError("Provide --classes / --classes-file or --num-classes")
    return [f"class_{i}" for i in range(num_classes)]


def load_model(model_name, checkpoint, num_classes, image_size, device):
    print(f"[INFO] Loading model {model_name} …")

    bench = create_model(
        model_name,
        bench_task='predict',
        num_classes=num_classes,
        checkpoint_path=checkpoint,
        image_size=(image_size, image_size),
    )

    bench.eval()
    bench.to(device)

    print("[INFO] Model loaded.")
    return bench


def preprocess_image(image_path, image_size, device):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((image_size, image_size))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img, img_tensor


def run_inference(bench, img_tensor):
    with torch.no_grad():
        detections = bench(img_tensor)
    return detections[0]  # (N, 6): x1,y1,x2,y2,score,label


def draw_detections(img, detections, image_size, class_names, score_thresh=0.85):
    det = detections.detach().cpu().numpy()
    detected_names = []

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if det.size == 0:
        return img_cv, detected_names

    boxes_n = det[:, 0:4]
    boxes = det[:, 0:4]
    scores = det[:, 4]
    labels = det[:, 5].astype(int)

    w, h = img.size
    scale_x = w / image_size
    scale_y = h / image_size

    # scale coords back to original image size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    for box, score, cls in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        if 0 <= cls < len(class_names):
            cls_name = class_names[cls-1]
        else:
            cls_name = f"cls_{cls}"

        detected_names.append(cls_name)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_cv,
            f"{cls_name}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return img_cv, detected_names


def xyxy2xywhn(box, img_w, img_h):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return [xc, yc, w, h]


def save_boxe(img_path, out_dir_txt, detections, score_thresh, image_size):
    det = detections.detach().cpu().numpy()

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]   # ← (H, W) dans le bon ordre

    scores = det[:, 4]
    labels = det[:, 5].astype(int)
    boxes  = det[:, 0:4].copy()    # coords dans l'espace image_size

    # Remettre à l'échelle de l'image originale
    scale_x = img_w / image_size
    scale_y = img_h / image_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    path_txt = os.path.join(out_dir_txt, img_name + ".txt")

    lines = []
    for box, score, cls in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        xywhn = xyxy2xywhn(box, img_w, img_h)   # normalisation YOLO
        lines.append(f"{cls -1} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f} {score:.6f}\n")

    with open(path_txt, 'w') as f:   # une seule ouverture
        f.writelines(lines)
        


def main():
    parser = argparse.ArgumentParser(description="EfficientDet inference script (Slurm-friendly)")

    parser.add_argument("--model", required=True, help="Model name (ex: tf_efficientdet_d0)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth or .pth.tar)")
    parser.add_argument("--image-dir", required=True, help="Path to input images directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--image-size", type=int, default=512, help="Image size used during training")
    parser.add_argument("--output-dir", required=True, help="Directory to save results (images + txt)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--score-thresh", type=float, default=0.10, help="Score threshold for displaying detections")

    # ✅ Solution 1: pass classes without local import
    parser.add_argument("--classes", default=None, help="Comma-separated class names (e.g. A,B,C)")
    parser.add_argument("--classes-file", default=None, help="Text file: one class name per line")

    # FPS settings
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (images) before timing")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Device: {device}")

    CLASS_NAMES = load_class_names(args.classes, args.classes_file, args.num_classes)
    print(f"[INFO] Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_txt = os.path.join(out_dir, "labels")
    os.makedirs(out_dir_txt, exist_ok=True)

    # Collect images
    exts = (".jpg", ".jpeg", ".png")
    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort()

    if args.max_images and args.max_images > 0:
        images = images[:args.max_images]

    if not images:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    # Load model
    start_load = time.time()
    bench = load_model(args.model, args.checkpoint, args.num_classes, args.image_size, device)
    load_time = time.time() - start_load
    print(f"[INFO] Loading model time: {load_time:.3f}s")

    total_params = sum(p.numel() for p in bench.parameters())
    print(f"[INFO] Number of parameters: {total_params:,}")

    # Warmup (important for FPS)
    warmup_n = min(args.warmup, len(images))
    if warmup_n > 0:
        print(f"[INFO] Warmup on {warmup_n} images…")
        for i in range(warmup_n):
            img, img_tensor = preprocess_image(str(images[i]), args.image_size, device)
            _ = run_inference(bench, img_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    class_count = {name: 0 for name in CLASS_NAMES}
    per_image = []

    # Timed loop
    print(f"[INFO] Running inference on {len(images)} images…")
    t0 = time.time()

    for idx, img_path in enumerate(images, start=1):
        img, img_tensor = preprocess_image(str(img_path), args.image_size, device)

        t_inf0 = time.time()
        det = run_inference(bench, img_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf = time.time() - t_inf0

        print(img_path)
        save_boxe(img_path, out_dir_txt, det, score_thresh=args.score_thresh, image_size=args.image_size)
        result_img, cls_names = draw_detections(img, det, args.image_size, CLASS_NAMES, score_thresh=args.score_thresh)


        # Count
        for cname in cls_names:
            if cname in class_count:
                class_count[cname] += 1

        # Save
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), result_img)

        per_image.append((img_path.name, cls_names, t_inf))
        print(f"[{idx:04d}/{len(images):04d}] {img_path.name} | inf={t_inf*1000:.2f} ms | det={len(cls_names)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.time() - t0

    fps = (len(images) / t_total) if t_total > 0 else 0.0
    avg_ms = (sum(t for _, _, t in per_image) / len(per_image)) * 1000.0

    print(f"\n[RESULT] Total time: {t_total:.3f}s for {len(images)} images")
    print(f"[RESULT] Avg inference time (per image): {avg_ms:.2f} ms")
    print(f"[RESULT] FPS (end-to-end loop): {fps:.2f}")

    # Write summary txt
    txt_path = out_dir / "classes_detected.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for img_name, cls_list, t_inf in per_image:
            if cls_list:
                classes_str = ", ".join(cls_list)
            else:
                classes_str = ""
            f.write(f"{img_name} : {classes_str} | inf_ms={t_inf*1000:.2f}\n")

        f.write("\n--- TOTAL DETECTIONS ---\n")
        for cname, count in class_count.items():
            f.write(f"{cname} : {count}\n")
        f.write("------------------------\n")

        f.write(f"\n--- SPEED ---\n")
        f.write(f"total_time_s: {t_total:.6f}\n")
        f.write(f"avg_inference_ms: {avg_ms:.6f}\n")
        f.write(f"fps: {fps:.6f}\n")

    print(f"[INFO] Wrote summary: {txt_path}")


if __name__ == "__main__":
    main()
