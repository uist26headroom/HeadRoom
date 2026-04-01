# HeadRoom 

HeadRoom predicts **visual** and **audio** channel availability from egocentric data (Aria-style frames + audio), then routes to the channel with more headroom.
Our companion website:
- [https://uist26headroom.github.io/HeadRoom/](https://uist26headroom.github.io/HeadRoom/)

## Project structure

- `train.py` — trains visual + audio predictors and runs a quick offline availability check.
- `export.py` — bundles checkpoints/config for deployment.
- `aria_loader.py` — dataset discovery + dataloaders.
- `models/` — model definitions and training loops.
- `inference/infergpu.py` — inference with auto device selection (`mps`/`cuda`/`cpu`).
- `inference/infercpu.py` — CPU-focused inference script.
- `checkpoints/` — trained model weights.

## Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- torch
- torchvision
- librosa
- numpy
- Pillow

## Training

```bash
python train.py --aria_root /path/to/aria_data --epochs 50
```

Useful flags:
- `--batch_size` (default: `1024`)
- `--lr` (default: `1e-3`)
- `--patience` (default: `5`)
- `--workers` (default: `24`)
- `--skip_visual`
- `--skip_audio`
- `--checkpoint_dir` (default: `checkpoints`)

Outputs:
- `checkpoints/visual_predictor.pt`
- `checkpoints/audio_predictor.pt`
- `checkpoints/*_losses.csv`

## Export for deployment

```bash
python export.py --checkpoint_dir checkpoints --out pi_deploy
```

This writes:
- `pi_deploy/visual_predictor.pt`
- `pi_deploy/audio_predictor.pt`
- `pi_deploy/config.json`
- `pi_deploy/README.txt`

## Inference

### macOS / GPU / MPS / CUDA (auto)

```bash
python inference/infergpu.py \
  --visual_ckpt checkpoints/visual_predictor.pt \
  --audio_ckpt checkpoints/audio_predictor.pt \
  --source files \
  --frame_dir /path/to/frames \
  --wav /path/to/audio.wav \
  --fps 10 \
  --threshold 0.5 \
  --out_csv /path/to/infer_results.csv
```

Optional: set `--device auto|mps|cuda|cpu`.

### CPU-only

```bash
python inference/infercpu.py \
  --visual_ckpt checkpoints/visual_predictor.pt \
  --audio_ckpt checkpoints/audio_predictor.pt \
  --source files \
  --frame_dir /path/to/frames \
  --wav /path/to/audio.wav \
  --fps 10 \
  --threshold 0.5 \
  --out_csv /path/to/infer_results.csv
```

### Unity
For our standalone headset deplyment we converted the MobileNetV3 backbone, Visual MLP and Audio MLP to ONNX and they are available at unity/assets/models.
