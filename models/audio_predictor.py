"""
Audio Predictive Model — Sensory Spotlight
==========================================
Architecture:
  - Extract a 31-dim feature vector from a short audio window:
      13 MFCC means     — spectral shape (what it sounds like)
      13 MFCC stds      — spectral variability within the window
       1 RMS mean       — average loudness
       1 RMS max        — peak loudness (preserves sudden loud events)
       1 RMS variance   — energy dynamics (catches onsets)
       1 spectral flux  — frame-to-frame spectral change rate
       1 zero crossing  — noisiness / tonality proxy
  - Tiny 2-layer MLP predicts features of the *next* window (window_t+1)
  - Prediction error (MSE) is the availability signal:
      high error → auditory channel occupied / surprising
      low error  → auditory channel has headroom

Why 31 dims instead of 13:
  Mean MFCCs alone flush out sudden loud events — a 50ms crash in a
  0.5s window averages to near-silence. RMS max and spectral flux
  preserve these onset events, which are the primary driver of
  auditory load in naturalistic environments.

Pi 5 target: <5ms per forward pass (CPU, float32)
Dependencies: librosa, torch, numpy
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

try:
    import librosa
except ImportError:
    raise ImportError("pip install librosa")


# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # Hz — Aria records at 48k; downsample for speed
N_MFCC        = 13       # number of MFCC coefficients
HOP_LENGTH    = 512      # samples between frames
N_FFT         = 1024     # FFT window
WINDOW_SECS   = 0.5      # seconds per analysis window
WINDOW_FRAMES = int(SAMPLE_RATE * WINDOW_SECS)   # samples per window

# Feature vector: 13 mfcc_mean + 13 mfcc_std + rms_mean + rms_max + rms_var
#                 + spectral_flux + zcr = 31 dims
FEATURE_DIM   = N_MFCC * 2 + 5   # 31
HIDDEN_DIM    = 128               # slightly larger to handle richer input


# ── feature extraction ────────────────────────────────────────────────────────
def extract_audio_features(
    audio_window: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract a 31-dim feature vector from a raw audio window.

    Feature layout:
      [0:13]  mfcc_mean     — spectral shape (what it sounds like)
      [13:26] mfcc_std      — spectral variability within window
      [26]    rms_mean      — average loudness
      [27]    rms_max       — peak loudness  ← catches sudden loud events
      [28]    rms_var       — energy variance ← captures onset structure
      [29]    spectral_flux — frame-to-frame spectral change rate
      [30]    zcr           — zero crossing rate (noisiness proxy)

    The critical fix over plain MFCC mean:
      A 50ms crash in a 0.5s window averages to near-silence in mfcc_mean.
      rms_max still peaks. spectral_flux still spikes. The model sees the event.

    Returns: (31,) float32 numpy array
    """
    if len(audio_window) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    y = audio_window.astype(np.float32)

    # MFCCs — spectral shape and variability
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std  = mfccs.std(axis=1)
    # standardise mean per coefficient → roughly [-3, 3]
    mfcc_mean = (mfcc_mean - mfcc_mean.mean()) / (mfcc_mean.std() + 1e-8)
    # standardise std similarly → [0, ~2]
    mfcc_std  = mfcc_std / (mfcc_std.max() + 1e-8)

    # RMS energy — average, peak, variance
    # Audio is normalised -1..1 so RMS is already in [0, 1].
    # rms_var can be tiny — scale it up to be on par with mean/max.
    rms      = librosa.feature.rms(y=y, frame_length=N_FFT,
                                   hop_length=HOP_LENGTH)[0]
    rms_mean = float(rms.mean())                    # [0, 1]
    rms_max  = float(rms.max())                     # [0, 1]
    rms_var  = float(np.sqrt(rms.var() + 1e-8))     # sqrt → same scale as mean/max

    # Spectral flux — log-compressed to bring it to [0, ~5] range
    # Raw flux can be in [0, 1000+] which dominates MSE completely.
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    if spec.shape[1] > 1:
        raw_flux = float(np.mean(np.diff(spec, axis=1) ** 2))
        flux = float(np.log1p(raw_flux))            # log(1+x) → [0, ~7]
    else:
        flux = 0.0

    # Zero crossing rate — already in [0, 1], scale to [0, ~2] to match others
    zcr = float(librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0].mean()) * 2.0

    return np.concatenate([
        mfcc_mean,                     # 13  range ~[-3,  3]
        mfcc_std,                      # 13  range  [0,   1]
        [rms_mean, rms_max, rms_var],  #  3  range  [0,   1]
        [flux],                        #  1  range  [0,  ~7]
        [zcr],                         #  1  range  [0,  ~2]
    ]).astype(np.float32)              # 31 total — all features now comparable


def extract_mfcc(audio_window: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Backwards-compatible alias — existing code keeps working."""
    return extract_audio_features(audio_window, sr)


# ── MLP prediction head ───────────────────────────────────────────────────────
class AudioPredictor(nn.Module):
    """
    Predicts the 31-dim feature vector of window t+1 given window t.
    3-layer MLP: 31 → 128 → 64 → 31
    ~12K parameters — still trivially runs on Pi.

    Extra hidden layer vs original because the input is richer —
    the model needs capacity to learn interactions between energy
    features and spectral features.
    """
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 31) — feature vector of current window
        returns: (B, 31) — predicted feature vector of next window
        """
        return self.net(x)


# ── availability signal ────────────────────────────────────────────────────────
def audio_prediction_error(
    model: AudioPredictor,
    window_t: np.ndarray,
    window_t1: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> float:
    """
    Compute MSE between predicted and actual next-window feature vectors.
    Higher → auditory channel is more occupied / unpredictable.

    Uses the full 31-dim feature vector including RMS energy stats and
    spectral flux, so sudden loud events are not averaged away.

    Args:
        model:     trained AudioPredictor (eval mode)
        window_t:  current audio window  (numpy array)
        window_t1: next audio window     (numpy array)

    Returns:
        scalar MSE error (float)
    """
    feat_t  = extract_audio_features(window_t,  sr)
    feat_t1 = extract_audio_features(window_t1, sr)

    x = torch.tensor(feat_t,  dtype=torch.float32).unsqueeze(0)   # (1, 31)
    y = torch.tensor(feat_t1, dtype=torch.float32).unsqueeze(0)   # (1, 31)

    model.eval()
    with torch.no_grad():
        pred  = model(x)
        error = nn.functional.mse_loss(pred, y)
    return error.item()


# ── dataset helper ─────────────────────────────────────────────────────────────
class AriaAudioWindowDataset(torch.utils.data.Dataset):
    """
    Wraps a list of audio windows extracted from Aria/Ego4D recordings.
    Each item is a (feat_t, feat_t1) consecutive 31-dim feature pair.

    Usage:
        windows = [audio_array_0, audio_array_1, ...]  # pre-sliced windows
        dataset = AriaAudioWindowDataset(windows)
    """
    def __init__(self, windows: list, sr: int = SAMPLE_RATE):
        self.features = [extract_audio_features(w, sr) for w in windows]

    def __len__(self) -> int:
        return max(0, len(self.features) - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_t  = torch.tensor(self.features[idx],     dtype=torch.float32)
        feat_t1 = torch.tensor(self.features[idx + 1], dtype=torch.float32)
        return feat_t, feat_t1


# ── training loop ──────────────────────────────────────────────────────────────
def train_audio_predictor(
    model: AudioPredictor,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: str = "checkpoints/audio_predictor.pt",
    device: str = "cpu",
    patience: int = 5,
):
    """
    Train the MLP to predict next-window features.
    Loss: MSE between predicted and actual 31-dim feature vector.

    Includes:
      - Per-epoch train + val loss logging
      - Best checkpoint saved by val loss (not final epoch)
      - Early stopping after `patience` epochs without val improvement
      - Loss curve written to <save_path stem>_losses.csv for paper figures
    """
    import csv
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status':>14}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*14}")

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        total = 0.0
        for feat_t, feat_t1 in train_loader:
            feat_t, feat_t1 = feat_t.to(device), feat_t1.to(device)
            pred = model(feat_t)
            loss = criterion(pred, feat_t1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * feat_t.size(0)
        train_loss = total / len(train_loader.dataset)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        total = 0.0
        with torch.no_grad():
            for feat_t, feat_t1 in val_loader:
                feat_t, feat_t1 = feat_t.to(device), feat_t1.to(device)
                pred = model(feat_t)
                loss = criterion(pred, feat_t1)
                total += loss.item() * feat_t.size(0)
        val_loss = total / len(val_loader.dataset)

        scheduler.step(val_loss)

        # ── checkpoint + early stopping ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            status           = "✓ best"
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, save_path)
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{patience}"

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "best_val": best_val_loss})
        print(f"  {epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {status:>14}")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # ── save loss curve ──────────────────────────────────────────────────────
    curve_path = Path(save_path).with_name(Path(save_path).stem + "_losses.csv")
    with open(curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","best_val"])
        writer.writeheader()
        writer.writerows(history)

    best_epoch = min(history, key=lambda r: r["val_loss"])["epoch"]
    print(f"\n  Best val loss : {best_val_loss:.6f}  (epoch {best_epoch})")
    print(f"  Loss curve    → {curve_path}")
    print(f"  Checkpoint    → {save_path}")
    return history


# ── quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AudioPredictor()

    # Fake audio windows
    window_t  = np.random.randn(WINDOW_FRAMES).astype(np.float32) * 0.01
    window_t1 = np.random.randn(WINDOW_FRAMES).astype(np.float32) * 0.01

    # Test feature extraction
    feats = extract_audio_features(window_t)
    print(f"Feature vector shape: {feats.shape}  (expected 31)")
    print(f"  mfcc_mean  [{feats[0:13].min():.3f}, {feats[0:13].max():.3f}]")
    print(f"  mfcc_std   [{feats[13:26].min():.3f}, {feats[13:26].max():.3f}]")
    print(f"  rms_mean   {feats[26]:.6f}")
    print(f"  rms_max    {feats[27]:.6f}")
    print(f"  rms_var    {feats[28]:.6f}")
    print(f"  flux       {feats[29]:.6f}")
    print(f"  zcr        {feats[30]:.6f}")

    # Test with a loud event in window_t1
    window_loud = window_t1.copy()
    window_loud[4000:4100] = 0.9   # 100-sample loud burst
    feats_loud = extract_audio_features(window_loud)
    print(f"\nLoud event detection:")
    print(f"  quiet rms_max={feats[27]:.6f}  loud rms_max={feats_loud[27]:.6f}  "
          f"ratio={feats_loud[27]/(feats[27]+1e-9):.1f}x")
    print(f"  quiet flux   ={feats[29]:.6f}  loud flux   ={feats_loud[29]:.6f}  "
          f"ratio={feats_loud[29]/(feats[29]+1e-9):.1f}x")

    # Test prediction error
    err = audio_prediction_error(model, window_t, window_t1)
    print(f"\nSmoke test — prediction error (untrained): {err:.4f}")
    print(f"MLP params: {sum(p.numel() for p in model.parameters()):,}")