"""
infer.py — Sensory Spotlight (Pi inference)
============================================
Loads exported CPU weights and runs the availability index in real time.

Usage:
    python infer.py --visual_ckpt visual_predictor.pt --audio_ckpt audio_predictor.pt
"""

import argparse
import csv
import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

try:
    import librosa
except ImportError:
    raise ImportError("pip install librosa")


EMBED_DIM     = 576
HIDDEN_DIM_V  = 128
SAMPLE_RATE   = 16_000
N_MFCC        = 13
HOP_LENGTH    = 512
N_FFT         = 1024
WINDOW_SECS   = 0.5
WINDOW_FRAMES = int(SAMPLE_RATE * WINDOW_SECS)
FEATURE_DIM   = N_MFCC * 2 + 5
HIDDEN_DIM_A  = 128

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VisualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM_V),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN_DIM_V, EMBED_DIM),
        )
    def forward(self, x): return self.net(x)


class AudioMLP(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM_A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, feature_dim),
        )
    def forward(self, x): return self.net(x)


def load_backbone():
    backbone = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    backbone.classifier = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def extract_audio_features(audio_window: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if len(audio_window) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    y = audio_window.astype(np.float32)
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std  = mfccs.std(axis=1)
    mfcc_mean = (mfcc_mean - mfcc_mean.mean()) / (mfcc_mean.std() + 1e-8)
    mfcc_std  = mfcc_std / (mfcc_std.max() + 1e-8)
    rms       = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms_mean  = float(rms.mean())
    rms_max   = float(rms.max())
    rms_var   = float(np.sqrt(rms.var() + 1e-8))
    spec      = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    flux      = float(np.log1p(np.mean(np.diff(spec, axis=1) ** 2))) if spec.shape[1] > 1 else 0.0
    zcr       = float(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0].mean()) * 2.0
    return np.concatenate([mfcc_mean, mfcc_std, [rms_mean, rms_max, rms_var], [flux], [zcr]]).astype(np.float32)


def extract_mfcc(audio_window: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    return extract_audio_features(audio_window, sr)


class RunningNorm:
    """
    Per-sequence calibrated normaliser with separate EMA alphas for
    mean and variance.

    Phase 1 (first 30 frames): collect raw errors, seed mean/variance
      from actual sequence statistics. Returns 0.5 during warmup.
    Phase 2: Z-score against running statistics.

    Separate alphas:
      alpha_mean=0.05 -> mean adapts in ~20 frames (tracks slow drift)
      alpha_var=0.01  -> variance adapts in ~100 frames (stays stable)

    Slow variance is critical: when a spike hits, we want it to remain
    a visible outlier for several frames rather than immediately inflating
    sigma and normalising itself away. This gives the state machine time
    to respond before the signal self-corrects.

    clip_sigmas:
      vis=2.0 -> moderate, reduces single-frame volatility
      aud=1.5 -> tighter clip, wider dynamic range in [0,1]
    """
    def __init__(self, alpha_mean=0.05, alpha_var=0.01, clip_sigmas=2.0):
        self.alpha_mean  = alpha_mean
        self.alpha_var   = alpha_var
        self.clip_sigmas = clip_sigmas
        self._mean       = None
        self._var        = None
        self._calibrated = False
        self._calib_buf  = []

    def seed(self, errors: list):
        if len(errors) < 2:
            return
        arr              = np.array(errors, dtype=np.float32)
        self._mean       = float(arr.mean())
        self._var        = float(arr.var()) + 1e-6
        self._calibrated = True
        self._calib_buf  = []

    def reset(self):
        self._mean       = None
        self._var        = None
        self._calibrated = False
        self._calib_buf  = []

    def normalise(self, v: float) -> float:
        if not self._calibrated:
            self._calib_buf.append(v)
            if len(self._calib_buf) >= 30:
                self.seed(self._calib_buf)
            return 0.5
        self._mean = self.alpha_mean * v + (1 - self.alpha_mean) * self._mean
        residual   = (v - self._mean) ** 2
        self._var  = self.alpha_var * residual + (1 - self.alpha_var) * self._var
        std        = max(self._var ** 0.5, 1e-6)
        z          = (v - self._mean) / std
        half       = self.clip_sigmas
        return float(np.clip((z + half) / (2 * half), 0.0, 1.0))


class SensorySpotlight:
    def __init__(self, visual_ckpt: str, audio_ckpt: str, threshold: float = 0.1, device=None):
        print("Loading backbone ...", end=" ", flush=True)
        self.backbone = load_backbone()
        print("done")
        self.vis_mlp = VisualMLP()
        self.vis_mlp.load_state_dict(torch.load(visual_ckpt, map_location=device))
        self.vis_mlp.eval()
        self.aud_mlp = AudioMLP()
        self.aud_mlp.load_state_dict(torch.load(audio_ckpt, map_location=device))
        self.aud_mlp.eval()
        # separate clip_sigmas per channel
        # vis: 2.0 — moderate, reduces single-frame volatility
        # aud: 1.8 — log compression already handles spike skew,
        #            so we can afford slightly wider clip for smoother signal
        self.vis_norm = RunningNorm(alpha_mean=0.05, alpha_var=0.01, clip_sigmas=2.0)
        self.aud_norm = RunningNorm(alpha_mean=0.05, alpha_var=0.01, clip_sigmas=1.8)
        self.threshold      = threshold
        self._last_vis_emb   = None
        self._last_audio_feat = None

    @torch.no_grad()
    def _embed_frame(self, frame: Image.Image) -> torch.Tensor:
        t = TRANSFORM(frame).unsqueeze(0)
        return self.backbone(t)

    def _audio_features(self, window: np.ndarray) -> torch.Tensor:
        return torch.tensor(extract_audio_features(window)).unsqueeze(0)

    def step(self, frame: Image.Image, audio_window: np.ndarray) -> dict:
        t0 = time.perf_counter()

        cur_vis_emb = self._embed_frame(frame)
        if self._last_vis_emb is not None:
            pred_vis = self.vis_mlp(self._last_vis_emb)
            vis_err  = nn.functional.mse_loss(pred_vis, cur_vis_emb).item()
        else:
            vis_err = 0.0
        self._last_vis_emb = cur_vis_emb

        cur_audio_feat = self._audio_features(audio_window)
        if self._last_audio_feat is not None:
            pred_aud = self.aud_mlp(self._last_audio_feat)
            aud_err  = nn.functional.mse_loss(pred_aud, cur_audio_feat).item()
        else:
            aud_err = 0.0
        self._last_audio_feat = cur_audio_feat

        # Log-compress audio error before normalising.
        # Raw aud_err is right-skewed — mostly near-zero with rare large spikes.
        # Without compression those spikes blow past clip_sigmas instantly and
        # floor aud_avail at 0.000. log1p pulls the tail in so spikes produce
        # a strong but finite signal. Scale x1000 first so the compressed range
        # aligns with vis_err order of magnitude.
        aud_err_norm = float(np.log1p(aud_err * 1000))

        # high error -> low availability
        # Floor at 0.05: hard zero adds no information over near-zero and
        # causes state machine instability — a channel stuck at 0.000 looks
        # identical whether mildly or severely occupied.
        vis_avail = max(0.05, 1.0 - self.vis_norm.normalise(vis_err))
        aud_avail = max(0.05, 1.0 - self.aud_norm.normalise(aud_err_norm))

        vis_ok = vis_avail >= self.threshold
        aud_ok = aud_avail >= self.threshold

        if not vis_ok and not aud_ok:
            channel = "visual" if vis_avail > aud_avail else "audio"
        elif not vis_ok:
            channel = "audio"
        elif not aud_ok:
            channel = "visual"
        elif abs(vis_avail - aud_avail) < 0.05:
            channel = "either"
        else:
            channel = "visual" if vis_avail > aud_avail else "audio"

        latency = (time.perf_counter() - t0) * 1000
        return {
            "channel":        channel,
            "vis_avail":      round(vis_avail, 4),
            "aud_avail":      round(aud_avail, 4),
            "confidence":     round(abs(vis_avail - aud_avail), 4),
            "vis_error_raw":  round(vis_err, 6),
            "aud_error_raw":  round(aud_err, 6),
            "aud_error_norm": round(aud_err_norm, 4),   # log-compressed, for diagnostics
            "latency_ms":     round(latency, 1),
        }


def run_on_files(engine, frame_dir: str, wav_path: str,
                 max_steps: int = 200, fps: int = 10,
                 out_csv: str = None):
    frames = sorted(
        glob.glob(os.path.join(frame_dir, "*.jpg")) +
        glob.glob(os.path.join(frame_dir, "*.png"))
    )
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    samples_per_frame = SAMPLE_RATE // fps

    print(f"  Audio : {len(audio)/SAMPLE_RATE:.1f}s  ({len(audio)} samples @ {SAMPLE_RATE}Hz)")
    print(f"  Frames: {len(frames)}  @  {fps}fps  ({samples_per_frame} samples/frame)")
    print(f"  Window: {WINDOW_FRAMES} samples ({WINDOW_SECS}s) centred on each frame")
    print(f"  Note  : first 30 steps are normaliser calibration — scores fixed at 0.5\n")
    print(f"{'t':>4}  {'channel':<8}  {'vis_avail':>10}  {'aud_avail':>10}"
          f"  {'vis_err':>9}  {'aud_err':>9}  {'ms':>6}")
    print("-" * 75)

    rows = []

    for t, frame_path in enumerate(frames[:max_steps]):
        centre = t * samples_per_frame
        start  = max(0, centre - WINDOW_FRAMES // 2)
        end    = start + WINDOW_FRAMES
        if end > len(audio):
            start = max(0, len(audio) - WINDOW_FRAMES)
            end   = len(audio)
        window = audio[start:end]
        if len(window) < WINDOW_FRAMES:
            window = np.pad(window, (0, WINDOW_FRAMES - len(window)))

        frame  = Image.open(frame_path).convert("RGB")
        result = engine.step(frame, window)
        result['t'] = t
        result['frame'] = os.path.basename(frame_path)
        rows.append(result)
        print(
            f"{t:>4}  {result['channel']:<8}  "
            f"{result['vis_avail']:>10.4f}  {result['aud_avail']:>10.4f}  "
            f"{result['vis_error_raw']:>9.5f}  {result['aud_error_raw']:>9.5f}  "
            f"{result['latency_ms']:>6.1f}ms"
        )

    # ── save to CSV ──
    if out_csv is None:
        out_csv = os.path.join(frame_dir, '..', 'infer_results.csv')
    out_csv = os.path.abspath(out_csv)
    fieldnames = ['t', 'frame', 'channel', 'vis_avail', 'aud_avail',
                  'confidence', 'vis_error_raw', 'aud_error_raw',
                  'aud_error_norm', 'latency_ms']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visual_ckpt", required=True)
    p.add_argument("--audio_ckpt",  required=True)
    p.add_argument("--source",      default="files", choices=["files", "picamera"])
    p.add_argument("--frame_dir",   default="/home/dinithi/Documents/Github/Map_x/SS1/pi_deploy/frmaes_game/captured_frames")
    p.add_argument("--wav",         default="/home/dinithi/Documents/Github/Map_x/SS1/pi_deploy/video3_comb/audio_comb.wav")
    p.add_argument("--fps",         type=int,   default=10)
    p.add_argument("--max_steps",   type=int,   default=1999999)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--out_csv",     default="/home/dinithi/Documents/Github/Map_x/SS1/pi_deploy/frmaes_game/infer_results_latency.csv", help="Path to save results CSV (default: <frame_dir>/../infer_results.csv)")
    args = p.parse_args()

    engine = SensorySpotlight(args.visual_ckpt, args.audio_ckpt, args.threshold)

    if args.source == "files":
        if not args.frame_dir or not args.wav:
            raise ValueError("--frame_dir and --wav required for --source files")
        run_on_files(engine, args.frame_dir, args.wav, args.max_steps, args.fps, args.out_csv)
    elif args.source == "picamera":
        print("Pi camera source: wire engine.step(frame, audio_window) to your capture loop.")