"""
export.py — Sensory Spotlight
==============================
Packages trained MLP weights into a single Pi-ready bundle.
Run this on your GPU machine after training. Copy the output to the Pi.

Usage:
    python export.py --checkpoint_dir checkpoints/ --out pi_deploy/

What gets exported:
    pi_deploy/
        visual_predictor.pt   ← MLP head weights only (CPU tensors, ~440KB)
        audio_predictor.pt    ← Audio MLP weights (CPU tensors, <10KB)
        config.json           ← Architecture config so Pi knows what to rebuild
        README.txt            ← Copy-paste instructions for the Pi

What does NOT get exported:
    - MobileNetV3-Small backbone (Pi downloads from torchvision on first run,
      or you can copy it separately — it's ~10MB)
    - Training code, dataset code, GPU dependencies
"""

import argparse
import json
import shutil
import torch
from pathlib import Path


EXPORT_CONFIG = {
    "visual": {
        "backbone":    "mobilenet_v3_small",
        "embed_dim":   576,
        "hidden_dim":  128,
        "input_size":  [224, 224],
    },
    "audio": {
        "n_mfcc":      13,
        "hidden_dim":  64,
        "sample_rate": 16000,
        "window_secs": 0.5,
    },
    "index": {
        "ema_alpha":   0.01,
        "threshold":   0.3,
    }
}

PI_README = """
Sensory Spotlight — Pi Inference Package
=========================================

Files:
  visual_predictor.pt  — MLP head weights (MobileNetV3-Small backbone auto-downloads)
  audio_predictor.pt   — Audio MLP weights
  config.json          — Architecture config

Setup on Pi:
  pip install torch torchvision librosa numpy Pillow

Run inference:
  python infer.py --visual_ckpt visual_predictor.pt --audio_ckpt audio_predictor.pt

First run will download MobileNetV3-Small (~10MB) from torchvision if not cached.
Subsequent runs use the cached backbone (~/.cache/torch/hub/).

Expected latency on Pi 4 (4GB):
  Visual feature extraction : ~35ms
  Visual MLP forward        : <1ms
  Audio MFCC + MLP          : ~5ms
  Availability routing      : <1ms
  Total per timestep        : ~40ms → ~25 routing decisions/sec
"""


def export(checkpoint_dir: str, out_dir: str):
    src  = Path(checkpoint_dir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    vis_src = src / "visual_predictor.pt"
    aud_src = src / "audio_predictor.pt"

    # Verify checkpoints exist
    if not vis_src.exists():
        raise FileNotFoundError(f"Visual checkpoint not found: {vis_src}\nRun train.py first.")
    if not aud_src.exists():
        raise FileNotFoundError(f"Audio checkpoint not found: {aud_src}\nRun train.py first.")

    # Load and verify they are CPU tensors
    vis_state = torch.load(vis_src, map_location="cpu")
    aud_state = torch.load(aud_src, map_location="cpu")

    any_cuda = any(v.is_cuda for v in vis_state.values())
    if any_cuda:
        print("  Warning: visual weights are on CUDA — converting to CPU …")
        vis_state = {k: v.cpu() for k, v in vis_state.items()}

    any_cuda = any(v.is_cuda for v in aud_state.values())
    if any_cuda:
        print("  Warning: audio weights are on CUDA — converting to CPU …")
        aud_state = {k: v.cpu() for k, v in aud_state.items()}

    # Save to export dir
    torch.save(vis_state, dest / "visual_predictor.pt")
    torch.save(aud_state, dest / "audio_predictor.pt")

    # Write config
    with open(dest / "config.json", "w") as f:
        json.dump(EXPORT_CONFIG, f, indent=2)

    # Write README
    (dest / "README.txt").write_text(PI_README)

    # Size report
    vis_kb = (dest / "visual_predictor.pt").stat().st_size / 1024
    aud_kb = (dest / "audio_predictor.pt").stat().st_size / 1024

    print(f"\n  Export complete → {dest.resolve()}")
    print(f"  visual_predictor.pt : {vis_kb:.1f} KB")
    print(f"  audio_predictor.pt  : {aud_kb:.1f} KB")
    print(f"  config.json + README included")
    print(f"\n  Copy the '{dest}/' folder to your Pi and run infer.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--out",            default="pi_deploy")
    args = p.parse_args()
    export(args.checkpoint_dir, args.out)