"""
Aria Dataset Loader — Sensory Spotlight
========================================
Loads egocentric visual frames and audio windows from the
Aria Everyday Activities dataset for training the predictive models.

Dataset structure assumed (standard Aria download layout):
    aria_data/
        <sequence_id>/
            video/          ← RGB frames as .jpg or .png
            audio/          ← .wav files (or single recording.wav)
            metadata.json   ← optional, ignored here

Adjust ARIA_ROOT to point to your download.

Usage:
    vis_loader, aud_loader = build_dataloaders("/path/to/aria_data")
    train_visual_predictor(visual_model, vis_loader)
    train_audio_predictor(audio_model, aud_loader)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

try:
    import librosa
except ImportError:
    raise ImportError("pip install librosa")

from models.visual_predictor import VISUAL_TRANSFORM
from models.audio_predictor  import (
    extract_mfcc, SAMPLE_RATE, WINDOW_FRAMES, N_MFCC
)


# ── visual dataset ─────────────────────────────────────────────────────────────
class AriaVisualDataset(Dataset):
    """
    Consecutive (frame_t, frame_t1) pairs from Aria video frames.
    Frames loaded from disk using DataLoader workers for prefetching.
    """

    def __init__(
        self,
        frame_dirs: List[str],
        transform=VISUAL_TRANSFORM,
        stride: int = 1,
    ):
        """
        Args:
            frame_dirs: list of directories, each containing sorted frame images
            transform:  torchvision transform applied to each frame
            stride:     gap between t and t+1 (1 = adjacent frames)
        """
        self.transform = transform
        self.pairs: List[Tuple[str, str]] = []

        for d in frame_dirs:
            frame_paths = sorted(
                glob.glob(os.path.join(d, "*.jpg")) +
                glob.glob(os.path.join(d, "*.png"))
            )
            for i in range(len(frame_paths) - stride):
                self.pairs.append((frame_paths[i], frame_paths[i + stride]))

        if len(self.pairs) == 0:
            raise ValueError(f"No frame pairs found. Check frame_dirs: {frame_dirs}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path_t, path_t1 = self.pairs[idx]
        frame_t  = self.transform(Image.open(path_t).convert("RGB"))
        frame_t1 = self.transform(Image.open(path_t1).convert("RGB"))
        return frame_t, frame_t1


# ── audio dataset ──────────────────────────────────────────────────────────────
class AriaAudioDataset(Dataset):
    """
    Consecutive (mfcc_t, mfcc_t1) MFCC pairs from Aria audio recordings.
    Accepts a list of .wav file paths; slices them into fixed-size windows.
    """

    def __init__(
        self,
        wav_paths: List[str],
        sr: int = SAMPLE_RATE,
        window_frames: int = WINDOW_FRAMES,
        hop_frames: Optional[int] = None,
    ):
        """
        Args:
            wav_paths:     list of .wav file paths
            sr:            target sample rate (audio resampled if needed)
            window_frames: samples per window
            hop_frames:    step between windows (default = window_frames, no overlap)
        """
        if hop_frames is None:
            hop_frames = window_frames

        self.mfccs: List[np.ndarray] = []

        for wav_path in wav_paths:
            audio, file_sr = librosa.load(wav_path, sr=sr, mono=True)
            # Slice into windows
            for start in range(0, len(audio) - window_frames, hop_frames):
                window = audio[start: start + window_frames]
                self.mfccs.append(extract_mfcc(window, sr))

        if len(self.mfccs) < 2:
            raise ValueError(f"Not enough audio windows found. Check wav_paths: {wav_paths}")

    def __len__(self) -> int:
        return len(self.mfccs) - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mfcc_t  = torch.tensor(self.mfccs[idx],     dtype=torch.float32)
        mfcc_t1 = torch.tensor(self.mfccs[idx + 1], dtype=torch.float32)
        return mfcc_t, mfcc_t1


# ── discovery helpers ──────────────────────────────────────────────────────────
def find_frame_dirs(aria_root: str) -> List[str]:
    """
    Walk the Aria dataset root and return all directories that contain
    image files (assumes frames are under a 'video/' or 'rgb/' subfolder).
    """
    candidates = []
    for subdir in ["video", "rgb", "frames"]:
        dirs = glob.glob(os.path.join(aria_root, "**", subdir), recursive=True)
        candidates.extend(dirs)

    # Fall back: any directory containing images
    if not candidates:
        for dirpath, _, filenames in os.walk(aria_root):
            if any(f.endswith((".jpg", ".png")) for f in filenames):
                candidates.append(dirpath)

    return sorted(set(candidates))


def find_wav_paths(aria_root: str) -> List[str]:
    """
    Recursively find all .wav files under aria_root.
    """
    return sorted(glob.glob(os.path.join(aria_root, "**", "*.wav"), recursive=True))


# ── dataloader factory ─────────────────────────────────────────────────────────
def build_dataloaders(
    aria_root: str,
    batch_size: int = 16,
    num_workers: int = 2,
    val_split: float = 0.1,
    visual_stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for both modalities.

    Returns:
        vis_train_loader, vis_val_loader,
        aud_train_loader, aud_val_loader
    """
    frame_dirs = find_frame_dirs(aria_root)
    wav_paths  = find_wav_paths(aria_root)

    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found under {aria_root}")
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found under {aria_root}")

    print(f"Found {len(frame_dirs)} frame dir(s) and {len(wav_paths)} wav file(s)")

    # ── visual ──
    vis_dataset = AriaVisualDataset(frame_dirs, stride=visual_stride)
    n_val_vis   = max(1, int(len(vis_dataset) * val_split))
    n_trn_vis   = len(vis_dataset) - n_val_vis
    vis_trn, vis_val = torch.utils.data.random_split(vis_dataset, [n_trn_vis, n_val_vis])

    vis_train_loader = DataLoader(
        vis_trn, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    vis_val_loader = DataLoader(
        vis_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    # ── audio ──
    aud_dataset = AriaAudioDataset(wav_paths)
    n_val_aud   = max(1, int(len(aud_dataset) * val_split))
    n_trn_aud   = len(aud_dataset) - n_val_aud
    aud_trn, aud_val = torch.utils.data.random_split(aud_dataset, [n_trn_aud, n_val_aud])

    aud_train_loader = DataLoader(
        aud_trn, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    aud_val_loader = DataLoader(
        aud_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return vis_train_loader, vis_val_loader, aud_train_loader, aud_val_loader