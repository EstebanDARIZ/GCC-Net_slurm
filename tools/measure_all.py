import argparse
import time
import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.cnn import get_model_complexity_info

from mmdet.models import build_detector


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class TwoInputBackboneWrapper(nn.Module):
    """
    Wrap a backbone that expects forward(x, y) into a forward(x) for FLOPs tools.
    By default, we feed y = x (same tensor) just to satisfy the signature.
    """
    def __init__(self, backbone, mode="same"):
        super().__init__()
        self.backbone = backbone
        self.mode = mode

    def forward(self, x):
        if self.mode == "same":
            y = x
        else:
            y = torch.zeros_like(x)
        return self.backbone(x, y)


def measure_inference_time_backbone(detector, device, input_shape, warmup=20, runs=100):
    detector.eval()
    backbone = detector.backbone

    dummy = torch.randn(1, *input_shape, device=device)

    # detect if backbone needs 2 inputs (on correct device)
    is_two_input = False
    try:
        _ = backbone(dummy, dummy)
        is_two_input = True
    except TypeError:
        is_two_input = False

    with torch.no_grad():
        for _ in range(warmup):
            if is_two_input:
                _ = backbone(dummy, dummy)
            else:
                _ = backbone(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            if is_two_input:
                _ = backbone(dummy, dummy)
            else:
                _ = backbone(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    avg_time = (end - start) / runs
    fps = 1.0 / avg_time
    mode_used = "backbone(x,x)" if is_two_input else "backbone(x)"
    return avg_time, fps, mode_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--flops-scope", default="backbone", choices=["backbone", "full"])
    parser.add_argument("--img-size", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--two-input-mode", default="same", choices=["same", "zeros"])
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    input_shape = (3, args.img_size, args.img_size)

    # -----------------------------
    # Build detector + load weights
    # -----------------------------
    print("\nLoading config...")
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    print("Building detector...")
    detector = build_detector(cfg.model)
    load_checkpoint(detector, args.checkpoint, map_location="cpu")
    detector.to(device)
    detector.eval()

    # -----------------------------
    # Params (full)
    # -----------------------------
    total_params, trainable_params = count_params(detector)
    print("\n==============================")
    print("PARAMETERS (FULL MODEL)")
    print("==============================")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    # -----------------------------
    # FLOPs
    # -----------------------------
    if args.flops_scope == "backbone":
        backbone = detector.backbone
        bb_total, bb_trainable = count_params(backbone)

        # detect whether backbone needs (x,y) using dummy tensors on correct device
        dummy = torch.randn(1, *input_shape, device=device)
        needs_wrap = False
        try:
            _ = backbone(dummy, dummy)
            needs_wrap = True
        except TypeError:
            needs_wrap = False

        backbone_for_flops = backbone
        if needs_wrap:
            backbone_for_flops = TwoInputBackboneWrapper(backbone, mode=args.two_input_mode).to(device)

        flops, params = get_model_complexity_info(
            backbone_for_flops,
            input_shape,
            as_strings=True,
            print_per_layer_stat=False,
        )

        print("\n==============================")
        print("FLOPs (BACKBONE)")
        print("==============================")
        if needs_wrap:
            print(f"Note: backbone expects (x,y) -> measured with y={args.two_input_mode} (wrapper).")
        print(f"FLOPs      : {flops}")
        print(f"Params     : {params}")
        print(f"Backbone total params     : {bb_total:,}")
        print(f"Backbone trainable params : {bb_trainable:,}")

    else:
        print("\n[WARNING] FLOPs for FULL detector often fails in MMDet (forward expects dict inputs).")
        print("          Use --flops-scope backbone (recommended).")
        flops, params = get_model_complexity_info(
            detector,
            input_shape,
            as_strings=True,
            print_per_layer_stat=False,
        )
        print("\n==============================")
        print("FLOPs (FULL MODEL)")
        print("==============================")
        print(f"FLOPs  : {flops}")
        print(f"Params : {params}")

    # -----------------------------
    # Inference time / FPS (backbone)
    # -----------------------------
    avg_time, fps, mode_used = measure_inference_time_backbone(
        detector,
        device,
        input_shape,
        warmup=args.warmup,
        runs=args.runs
    )

    print("\n==============================")
    print("INFERENCE SPEED")
    print("==============================")
    print(f"Mode used        : {mode_used}")
    print(f"Avg time / image : {avg_time * 1000:.2f} ms")
    print(f"FPS              : {fps:.2f}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
