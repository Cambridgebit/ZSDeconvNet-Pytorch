# AGENTS.md - Development Guidelines

## Project Overview
Zero-shot deconvolution (ZSDeconv) for microscopy image processing. Python 3.12 + PyTorch.

**Note:** Notebook-based project. No Python package or `main.py`.

## Environment
```bash
# Python 3.12 (uv-managed venv)
# Dependencies: torch>=2.0, numpy>=1.26, Pillow>=9.0, matplotlib>=3.5
# GPU required for training
source .venv/Scripts/activate  # Linux/Mac
```

## Notebooks (7 total)

| Notebook | Purpose | Training Time |
|----------|---------|---------------|
| `denoise_training.ipynb` | Denoise-only (self-supervised) | ~10 min |
| `deconv_training.ipynb` | Deconv-only (skip denoise) | ~10 min |
| `denoise_to_deconv.ipynb` | Two-stage pipeline (recommended) | ~20 min |
| `joint_denoise_deconv_training.ipynb` | Joint training (fixed weights) | ~15 min |
| `joint_training_two_stage.ipynb` | Joint training (dynamic weights) | ~20 min |
| `noise2void_training.ipynb` | Blind-spot denoising | ~25 min |
| `psf_visualization.ipynb` | PSF analysis & validation | ~2 min |

## Quick Start

```bash
# Activate venv
source .venv/Scripts/activate

# Run recommended two-stage pipeline
jupyter notebook denoise_to_deconv.ipynb
```

## Key Parameters

```python
# Paths (in notebooks)
DENOISE_CHECKPOINT = "./checkpoints/best_denoise.pt"
PSF_PATH = "datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif"
IMAGE_PATH = "datasets/Microtubule/train_data/01.tif"

# Training defaults
EPOCHS = 30
LR = 1e-4
UPSAMPLE = True  # 2x output resolution
```

## Data Pipeline

```python
img = percentile_norm(load_image(path))      # → float32 [0,1]
save_image(path, arr)                        # → 16-bit TIFF
img_pad, padinfo = reflect_pad(img, multiple=16, margin=16)
result = crop_pad(pred, padinfo, scale=1)    # scale=2 for upsample
```

## PSF Loading (CRITICAL)

```python
# ✓ CORRECT: Direct normalization
psf = load_image(psf_path)
psf /= psf.sum()

# ✗ WRONG: Don't use percentile_norm!
# psf = percentile_norm(psf)  # Truncates weak signal & peak
```

## Checkpoints

| Type | Structure |
|------|-----------|
| Denoise | `{"model": state_dict, "cfg": {"base_ch", "depth", "n_conv"}}` |
| Deconv | `{"model": state_dict, "psf": tensor, "cfg": {"...","upsample"}}` |

## Gotchas

- **PSF loading**: Never use `percentile_norm()` - breaks PSF energy distribution
- **Padding**: `multiple=16, margin=16` for inference (reflect mode)
- **Upsample**: Output is 2x, adjust `crop_pad` with `scale=2`
- **Hessian weight**: `hess_w=0.02~0.05` suppresses artifacts
- **Images**: Saved as 16-bit TIFF, not 8-bit PNG
- **Notebooks**: Execute cells sequentially (dependencies between cells)
- **output/**: Training outputs excluded from git (see `.gitignore`)

## Model Architectures

- **DenoiseUNet**: depth=4, base_ch=32, n_conv=3
- **DirectDeconvNet**: depth=4, base_ch=32, upsample=True
- **JointDenoiseDeconvNet**: Cascaded denoise → deconv
- **Noise2VoidUNet**: depth=3, base_ch=64 (blind-spot)
