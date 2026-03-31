"""
Visual Predictive Model — Sensory Spotlight
============================================
Architecture:
  - Frozen MobileNetV3-Small backbone extracts 576-dim embeddings from RGB frames
  - Tiny 2-layer MLP head predicts the *next* frame's embedding
  - Prediction error (MSE) is the availability signal:
      high error → visual channel occupied
      low error  → visual channel has headroom

Pi 4 target: ~30-40ms per forward pass (CPU, float32, 224×224 input)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path


# ── constants ────────────────────────────────────────────────────────────────
EMBED_DIM   = 576      # MobileNetV3-Small penultimate feature dim
HIDDEN_DIM  = 128      # MLP hidden layer width — kept tiny for Pi
INPUT_SIZE  = (224, 224)


# ── transforms (reuse across training and inference) ─────────────────────────
VISUAL_TRANSFORM = T.Compose([
    T.Resize(INPUT_SIZE),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── backbone ──────────────────────────────────────────────────────────────────
def build_backbone() -> nn.Module:
    """
    MobileNetV3-Small with the classification head removed.
    All weights are frozen — we never update them.
    Output: (B, 576) feature vector per frame.
    """
    backbone = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    # Drop classifier; keep features + adaptive pool
    backbone.classifier = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    return backbone


# ── MLP prediction head ───────────────────────────────────────────────────────
class EmbeddingPredictor(nn.Module):
    """
    Predicts the embedding of frame t+1 given the embedding of frame t.
    Tiny 2-layer MLP: 576 → 128 → 576
    ~110K parameters total.
    """
    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── full visual predictor ─────────────────────────────────────────────────────
class VisualPredictor(nn.Module):
    """
    End-to-end module: raw frame tensor → predicted next-frame embedding.
    During inference, only the MLP head runs gradients.
    """
    def __init__(self):
        super().__init__()
        self.backbone  = build_backbone()
        self.predictor = EmbeddingPredictor()

    @torch.no_grad()
    def extract(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a single frame (no grad, Pi-safe)."""
        return self.backbone(frame)

    def forward(self, frame_t: torch.Tensor) -> torch.Tensor:
        """
        frame_t: (B, 3, 224, 224) — current frame
        returns: (B, 576)         — predicted embedding of next frame
        """
        with torch.no_grad():
            emb_t = self.backbone(frame_t)
        return self.predictor(emb_t)


# ── availability signal ────────────────────────────────────────────────────────
def visual_prediction_error(
    model: VisualPredictor,
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
) -> float:
    """
    Compute MSE between predicted and actual next-frame embedding.
    Higher → visual channel is more occupied.

    Args:
        model:    trained VisualPredictor (eval mode)
        frame_t:  current frame tensor  (1, 3, H, W)
        frame_t1: next frame tensor     (1, 3, H, W)

    Returns:
        scalar error (float)
    """
    model.eval()
    with torch.no_grad():
        pred_emb   = model(frame_t)
        actual_emb = model.extract(frame_t1)
        error      = nn.functional.mse_loss(pred_emb, actual_emb)
    return error.item()


# ── training loop ──────────────────────────────────────────────────────────────
def train_visual_predictor(
    model: VisualPredictor,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: str = "checkpoints/visual_predictor.pt",
    device: str = "cpu",
    patience: int = 5,
):
    """
    Train only the MLP head; backbone stays frozen.
    Loss: MSE between predicted and actual next-frame embedding.

    Includes:
      - Per-epoch train + val loss logging
      - Best checkpoint saved by val loss (not final epoch)
      - Early stopping after `patience` epochs without val improvement
      - Loss curve written to <save_path stem>_losses.csv for paper figures
    """
    import csv
    model = model.to(device)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
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
        for frame_t, frame_t1 in train_loader:
            frame_t, frame_t1 = frame_t.to(device), frame_t1.to(device)
            pred_emb   = model(frame_t)
            with torch.no_grad():
                actual_emb = model.extract(frame_t1)
            loss = criterion(pred_emb, actual_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * frame_t.size(0)
        train_loss = total / len(train_loader.dataset)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        total = 0.0
        with torch.no_grad():
            for frame_t, frame_t1 in val_loader:
                frame_t, frame_t1 = frame_t.to(device), frame_t1.to(device)
                pred_emb   = model(frame_t)
                actual_emb = model.extract(frame_t1)
                loss = criterion(pred_emb, actual_emb)
                total += loss.item() * frame_t.size(0)
        val_loss = total / len(val_loader.dataset)

        scheduler.step(val_loss)

        # ── checkpoint + early stopping ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            status           = "✓ best"
            cpu_state = {k: v.cpu() for k, v in model.predictor.state_dict().items()}
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
    model = VisualPredictor()
    dummy_t  = torch.randn(1, 3, 224, 224)
    dummy_t1 = torch.randn(1, 3, 224, 224)

    err = visual_prediction_error(model, dummy_t, dummy_t1)
    print(f"Smoke test — visual prediction error (untrained): {err:.4f}")
    print(f"MLP params: {sum(p.numel() for p in model.predictor.parameters()):,}")