"""
train.py — Sensory Spotlight
=============================
Train the visual and audio predictive models on the Aria dataset,
then run a quick offline availability index check.

Usage:
    python train.py --aria_root /path/to/aria_data --epochs 15
"""

import argparse
import torch
import sys
from pathlib import Path

# Make sure local imports work when run as a script
sys.path.insert(0, str(Path(__file__).parent))

from models.visual_predictor   import VisualPredictor, train_visual_predictor
from models.audio_predictor    import AudioPredictor,  train_audio_predictor
from models.availability_index import compute_availability_timeline
from aria_loader               import build_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train Sensory Spotlight predictors")
    p.add_argument("--aria_root",      required=True, help="Root of Aria dataset")
    p.add_argument("--epochs",         type=int, default=50,
                   help="Max epochs — early stopping cuts this short if converged")
    p.add_argument("--batch_size",     type=int, default=1024)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--patience",       type=int, default=5,
                   help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--workers",        type=int, default=24,
                   help="DataLoader workers (4-8 for desktop, 0 on Pi)")
    p.add_argument("--skip_visual",    action="store_true")
    p.add_argument("--skip_audio",     action="store_true")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Sensory Spotlight — Training")
    print("=" * 60)
    print(f"  Aria root   : {args.aria_root}")
    print(f"  Max epochs  : {args.epochs}  (early stop patience={args.patience})")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Device      : {device.upper()}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Weights saved as CPU tensors (Pi-compatible)")
    print("=" * 60)

    # ── load data ──────────────────────────────────────────────────────────────
    vis_trn, vis_val, aud_trn, aud_val = build_dataloaders(
        aria_root   = args.aria_root,
        batch_size  = args.batch_size,
        num_workers = args.workers,
    )

    # ── visual model ──────────────────────────────────────────────────────────
    if not args.skip_visual:
        print("\n[1/2] Training visual predictor …")
        vis_model = VisualPredictor()
        train_visual_predictor(
            model        = vis_model,
            train_loader = vis_trn,
            val_loader   = vis_val,
            epochs       = args.epochs,
            lr           = args.lr,
            save_path    = str(ckpt / "visual_predictor.pt"),
            device       = device,
            patience     = args.patience,
        )
    else:
        print("\n[1/2] Skipping visual training.")
        vis_model = VisualPredictor()
        vis_ckpt  = ckpt / "visual_predictor.pt"
        if vis_ckpt.exists():
            vis_model.predictor.load_state_dict(torch.load(vis_ckpt, map_location="cpu"))
            print(f"      Loaded checkpoint: {vis_ckpt}")

    # ── audio model ───────────────────────────────────────────────────────────
    if not args.skip_audio:
        print("\n[2/2] Training audio predictor …")
        aud_model = AudioPredictor()
        train_audio_predictor(
            model        = aud_model,
            train_loader = aud_trn,
            val_loader   = aud_val,
            epochs       = args.epochs,
            lr           = args.lr,
            save_path    = str(ckpt / "audio_predictor.pt"),
            device       = device,
            patience     = args.patience,
        )
    else:
        print("\n[2/2] Skipping audio training.")
        aud_model = AudioPredictor()
        aud_ckpt  = ckpt / "audio_predictor.pt"
        if aud_ckpt.exists():
            aud_model.load_state_dict(torch.load(aud_ckpt, map_location="cpu"))
            print(f"      Loaded checkpoint: {aud_ckpt}")

    # ── quick offline availability check on val set ───────────────────────────
    print("\n── Offline availability index (validation sample) ──")

    vis_model.to(device)
    aud_model.to(device)
    vis_model.eval()
    aud_model.eval()

    vis_errors, aud_errors = [], []

    # Collect a small batch of errors from each val loader
    with torch.no_grad():
        for frame_t, frame_t1 in vis_val:
            frame_t = frame_t.to(device)
            frame_t1 = frame_t1.to(device)
            pred   = vis_model(frame_t)
            actual = vis_model.extract(frame_t1)
            errs   = torch.nn.functional.mse_loss(pred, actual, reduction="none")
            errs   = errs.mean(dim=1)   # per-sample MSE
            vis_errors.extend(errs.tolist())
            if len(vis_errors) >= 100:
                break

        for mfcc_t, mfcc_t1 in aud_val:
            mfcc_t = mfcc_t.to(device)
            mfcc_t1 = mfcc_t1.to(device)
            pred = aud_model(mfcc_t)
            errs = torch.nn.functional.mse_loss(pred, mfcc_t1, reduction="none")
            errs = errs.mean(dim=1)
            aud_errors.extend(errs.tolist())
            if len(aud_errors) >= 100:
                break

    # Align lengths
    n = min(len(vis_errors), len(aud_errors), 100)
    timeline = compute_availability_timeline(vis_errors[:n], aud_errors[:n])

    routed_visual = sum(1 for s in timeline if s.recommended_channel.value == "visual")
    routed_audio  = sum(1 for s in timeline if s.recommended_channel.value == "audio")
    routed_either = sum(1 for s in timeline if s.recommended_channel.value == "either")
    avg_conf      = sum(s.confidence for s in timeline) / n

    print(f"\n  Timesteps evaluated : {n}")
    print(f"  → Routed to visual  : {routed_visual} ({routed_visual/n*100:.1f}%)")
    print(f"  → Routed to audio   : {routed_audio}  ({routed_audio/n*100:.1f}%)")
    print(f"  → Either (tied)     : {routed_either} ({routed_either/n*100:.1f}%)")
    print(f"  Mean confidence     : {avg_conf:.4f}")
    print("\nDone. Checkpoints saved to:", ckpt.resolve())


if __name__ == "__main__":
    main()