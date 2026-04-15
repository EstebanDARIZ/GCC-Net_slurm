#!/usr/bin/env python3
import argparse
import torch
from effdet import create_model
from PIL import Image
import numpy as np
import cv2
import os
import time

'''
Inference script for EfficientDet object detection model using PyTorch.
Exemple usage:
python inference.py \
    --model efficientdet_d0 \
    --checkpoint /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/output/train/20251202-113817-efficientdet_d0/model_best.pth.tar \
    --image 5.png \
    --num-classes 5 \
    --output out.jpg
'''
def load_model(model_name, checkpoint_path, num_classes, device="cuda"):
    # IMPORTANT: ne pas passer checkpoint_path à create_model
    model = create_model(model_name, num_classes=num_classes, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # selon comment le checkpoint est stocké
    state = ckpt.get("state_dict", ckpt)
    # enlever le prefix "module." si entrainé en DDP
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded with strict=False | missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device).eval()
    return model

def load_model_old(model_name, checkpoint, num_classes, image_size, device):
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
    return detections[0]


def draw_detections(img, detections, image_size, score_thresh=0.1):
    det = detections.cpu().numpy()   # detections is (N, 6)

    if det.size == 0:
        print("[INFO] No detections")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    boxes  = det[:, 0:4]
    scores = det[:, 4]
    labels = det[:, 5].astype(int)

    w, h = img.size
    scale_x = w / image_size
    scale_y = h / image_size

    # scale coords back to original image size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for box, score, cls in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{cls}:{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    return img_cv



def main():
    parser = argparse.ArgumentParser(description="EfficientDet inference script")

    parser.add_argument("--model", required=True, help="Model name (ex: tf_efficientdet_lite0)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth.tar")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--image-size", type=int, default=512, help="Image size used during training")
    parser.add_argument("--output", default="result.jpg", help="Output image file")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    bench = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,
        device
    )

    # Load & preprocess image
    img, img_tensor = preprocess_image(args.image, args.image_size, device)

    # Inference
    time_0 = time.time()
    detections = run_inference(bench, img_tensor)
    print("Time : ", time.time() - time_0)

    # Draw results
    result = draw_detections(img, detections, args.image_size)

    # Save image
    cv2.imwrite(args.output, result)
    print(f"[INFO] Saved detection result to: {args.output}")


if __name__ == "__main__":
    main()
