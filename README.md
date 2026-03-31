# HeadRoom 

HeadRoom predicts **visual** and **audio** channel availability from egocentric data (Aria-style frames + audio), then routes to the channel with more headroom.

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

## GitHub Pages (User Study Probes)

A static page is provided at `docs/index.html` with:
- Paper title header
- Code repository link
- Ordered user study probe videos: `Video 1`, `Video 2`, `Video 3`

Video files currently used by the page:
- `videos/video1_trimmed.mp4`
- `videos/video2_trimmed.mp4`
- `videos/video3_trimmed.mp4`

To deploy on GitHub Pages:

1. Push this repo to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, choose:
   - **Source**: `Deploy from a branch`
   - **Branch**: `main`
   - **Folder**: `/docs`
4. Save, then wait for Pages to publish.

Your site URL will be:
- `https://uist26headroom.github.io/HeadRoom/`

To update page text (paper title/probe descriptions), edit:
- `docs/index.html`

