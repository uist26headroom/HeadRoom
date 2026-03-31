"""
Modality Availability Index — Sensory Spotlight
================================================
Combines visual and audio prediction errors into a single routing signal.

Core idea (from paper):
  - High visual error  → visual channel occupied → route through AUDIO
  - High audio error   → audio channel occupied  → route through VISUAL
  - The index is a soft score in [0, 1] per channel, not a hard threshold

Usage:
    index = ModalityAvailabilityIndex()
    route = index.route(visual_error=0.42, audio_error=0.11)
    # → 'audio'  (visual channel is occupied)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


# ── types ─────────────────────────────────────────────────────────────────────
class Channel(str, Enum):
    VISUAL = "visual"
    AUDIO  = "audio"
    EITHER = "either"   # both channels available — pick lowest cost delivery


@dataclass
class AvailabilityState:
    visual_error:       float   # raw MSE from visual predictor
    audio_error:        float   # raw MSE from audio predictor
    visual_availability: float  # normalised score in [0,1]; 1 = fully available
    audio_availability:  float  # normalised score in [0,1]; 1 = fully available
    recommended_channel: Channel
    confidence:          float  # |visual_avail - audio_avail|; 0 = tie, 1 = clear


# ── running normaliser ────────────────────────────────────────────────────────
class RunningNormaliser:
    """
    Online min-max normaliser with exponential moving statistics.
    Keeps the index meaningful as the environment changes over time.
    Avoids storing a growing buffer — Pi-safe.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha   = alpha      # EMA smoothing factor (smaller = slower adaptation)
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    def update_and_normalise(self, value: float) -> float:
        if self._min is None:
            self._min = value
            self._max = value
            return 0.5

        # Exponential moving min / max
        self._min = self.alpha * value + (1 - self.alpha) * self._min
        self._max = self.alpha * value + (1 - self.alpha) * self._max

        # Clamp to ensure min <= value <= max
        lo = min(self._min, value)
        hi = max(self._max, value)

        if hi - lo < 1e-8:
            return 0.5
        return (value - lo) / (hi - lo)   # 0 = lo error, 1 = hi error


# ── availability index ─────────────────────────────────────────────────────────
class ModalityAvailabilityIndex:
    """
    Converts raw prediction errors into a routing decision.

    Parameters
    ----------
    alpha : float
        EMA smoothing for running normalisation (default 0.01).
    threshold : float
        Minimum availability score to consider a channel "available" (default 0.3).
        Below this, avoid that channel even if it's the better of the two.
    """
    def __init__(self, alpha: float = 0.01, threshold: float = 0.3):
        self._vis_norm  = RunningNormaliser(alpha)
        self._aud_norm  = RunningNormaliser(alpha)
        self.threshold  = threshold

    def update(
        self,
        visual_error: float,
        audio_error:  float,
    ) -> AvailabilityState:
        """
        Feed new errors → get a routing decision.

        Availability is INVERSE of normalised error:
            availability = 1 - normalised_error
        So higher error → lower availability → avoid that channel.
        """
        vis_norm = self._vis_norm.update_and_normalise(visual_error)
        aud_norm = self._aud_norm.update_and_normalise(audio_error)

        vis_avail = 1.0 - vis_norm   # high error → low availability
        aud_avail = 1.0 - aud_norm

        channel, confidence = self._decide(vis_avail, aud_avail)

        return AvailabilityState(
            visual_error         = visual_error,
            audio_error          = audio_error,
            visual_availability  = round(vis_avail, 4),
            audio_availability   = round(aud_avail, 4),
            recommended_channel  = channel,
            confidence           = round(confidence, 4),
        )

    def _decide(self, vis_avail: float, aud_avail: float) -> tuple[Channel, float]:
        vis_ok = vis_avail >= self.threshold
        aud_ok = aud_avail >= self.threshold

        if not vis_ok and not aud_ok:
            # Both channels stressed — pick the less bad one
            channel = Channel.VISUAL if vis_avail > aud_avail else Channel.AUDIO
        elif not vis_ok:
            channel = Channel.AUDIO
        elif not aud_ok:
            channel = Channel.VISUAL
        elif abs(vis_avail - aud_avail) < 0.05:
            channel = Channel.EITHER   # effectively tied
        else:
            channel = Channel.VISUAL if vis_avail > aud_avail else Channel.AUDIO

        confidence = abs(vis_avail - aud_avail)
        return channel, confidence


# ── convenience: batch evaluation over a sequence ─────────────────────────────
def compute_availability_timeline(
    visual_errors: list[float],
    audio_errors:  list[float],
    alpha: float = 0.01,
    threshold: float = 0.3,
) -> list[AvailabilityState]:
    """
    Process parallel sequences of visual/audio errors and return
    a per-timestep list of AvailabilityState objects.
    Useful for offline analysis on Aria dataset recordings.
    """
    assert len(visual_errors) == len(audio_errors), \
        "visual and audio error sequences must have the same length"

    index  = ModalityAvailabilityIndex(alpha=alpha, threshold=threshold)
    return [
        index.update(ve, ae)
        for ve, ae in zip(visual_errors, audio_errors)
    ]


# ── quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate a sequence: visual gets busy mid-way, then audio gets busy
    n = 30
    vis_errors = np.concatenate([
        np.random.uniform(0.01, 0.05, 10),   # both calm
        np.random.uniform(0.20, 0.40, 10),   # visual busy
        np.random.uniform(0.01, 0.05, 10),   # calm again
    ])
    aud_errors = np.concatenate([
        np.random.uniform(0.01, 0.05, 10),
        np.random.uniform(0.01, 0.05, 10),
        np.random.uniform(0.20, 0.40, 10),   # audio busy
    ])

    timeline = compute_availability_timeline(vis_errors.tolist(), aud_errors.tolist())

    print(f"{'t':>3}  {'vis_err':>8}  {'aud_err':>8}  {'vis_avail':>10}  {'aud_avail':>10}  {'channel':<10}  {'conf':>6}")
    print("-" * 72)
    for t, state in enumerate(timeline):
        print(
            f"{t:>3}  {state.visual_error:>8.4f}  {state.audio_error:>8.4f}"
            f"  {state.visual_availability:>10.4f}  {state.audio_availability:>10.4f}"
            f"  {state.recommended_channel.value:<10}  {state.confidence:>6.4f}"
        )