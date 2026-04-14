# AGENTS.md - Development Guidelines

## Project Overview
Zero-shot deconvolution (ZSDeconv) for microscopy image processing. Python 3.12 + PyTorch.

**Note:** This is a notebook-based research project. No `zsdeconv/` package or `main.py`.

## Environment
```bash
# Python 3.12 (uv-managed venv), .venv in project root
# Dependencies: torch>=2.0, numpy>=1.26, Pillow>=9.0, matplotlib>=3.5
# GPU required for training (CUDA auto-detected)
# Activate: source .venv/Scripts/activate (Linux/Mac) or .venv\Scripts\activate (Windows)
```

## Commands

### Quick Validation
```bash
# Run denoise training notebook cell-by-cell in Jupyter
jupyter notebook denoise_training.ipynb

# Two-stage: denoise → deconv
jupyter notebook denoise_to_deconv.ipynb

# Deconv-only (skip denoise)
jupyter notebook deconv_training.ipynb
```

### Verify Syntax (notebooks)
```bash
python -m py_compile -c "import nbformat; nbformat.read('denoise_training.ipynb', as_version=4)"
```

### No Test Framework
Integration testing via notebook execution only.

## Project Structure
```
.
├── denoise_training.ipynb        # Stage 1: denoise-only training
├── deconv_training.ipynb         # Deconv-only training (skip denoise)
├── denoise_to_deconv.ipynb       # Two-stage: denoise → deconv (recommended)
├── checkpoints/                  # Pre-trained models (best_denoise.pt, best_deconv.pt)
└── datasets/
    ├── Microtubule/              # Microtubule dataset + PSF
    └── Lysosome/                 # Lysosome dataset + PSF
```

## Training Strategy

### Two-Stage (Recommended)
Use `denoise_to_deconv.ipynb`:
1. **Denoise inference**: Load pre-trained denoise model, process image
2. **Deconv training**: Use denoised image to train deconvolution model

Parameters (in notebook):
```python
DENOISE_CHECKPOINT = "./checkpoints/best_denoise.pt"
ORIGINAL_IMAGE_PATH = "datasets/Microtubule/train_data/01.tif"
PSF_PATH = "datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif"
EPOCHS_DECONV = 30
LR_DECONV = 1e-4
UPSAMPLE = True  # 2x output resolution
```

### Denoise-Only
Use `denoise_training.ipynb`:
- Self-supervised pseudo-pair training (adds Gaussian noise)
- UNet architecture (depth=4, base_ch=32)
- L1Loss, StepLR scheduler

### Deconv-Only
Use `deconv_training.ipynb`:
- Skips denoise, trains deconvolution directly
- DeconvLoss = PSF convolution reconstruction + Hessian regularization
- Real PSF loaded from file (25×25 crop)

## Code Conventions

### Imports (order matters)
```python
# 1. Standard library (os, math)
# 2. Third-party (numpy, torch, nn, F)
# 3. Local utilities (same notebook, earlier cells)
```

### Naming
| Type | Style | Example |
|------|-------|---------|
| Notebooks | snake_case | `denoise_training.ipynb` |
| Classes | PascalCase | `DenoiseUNet`, `DirectDeconvNet`, `ConvBlock` |
| Functions/Vars | snake_case | `train_denoise`, `patch_size` |
| Constants | UPPER_SNAKE_CASE | `IMAGE_PATH`, `SEED`, `OUT_DIR` |
| Private | `_prefix` | N/A (notebook-based) |

### PyTorch
- All models inherit from `nn.Module`
- Use `nn.ModuleList` for dynamic layers
- Use `self.register_buffer()` for non-parameter tensors (PSF)
- `optimizer.zero_grad(set_to_none=True)` for memory efficiency
- `@torch.no_grad()` for inference functions

### Data Pipeline
```python
img = percentile_norm(load_image(path))      # float32 [0,1]
save_image(path, arr)                        # (arr * 65535).astype(np.uint16)
img_pad, padinfo = reflect_pad(img, multiple=16, margin=16)
result = crop_pad(pred, padinfo, scale=1)    # or scale=2 for 2x upsample
```

## Checkpoints
| Type | Structure |
|------|-----------|
| Denoise | `{"model": state_dict, "cfg": {"base_ch", "depth", "n_conv"}}` |
| Deconv | `{"model": state_dict, "psf": tensor, "cfg": {"base_ch", "depth", "n_conv", "upsample"}}` |
| PSF | Saved as `psf_used.tif` alongside checkpoint |

## VSCode
- Environment manager: conda (`.vscode/settings.json`)
- Jupyter kernel: `rknn2.4.0` (Python 3.11.15) or project venv

## Gotchas
- Research code: minimal error handling
- Call `ensure_dir()` before writing files
- Images saved as 16-bit TIFF
- Padding: `multiple=16, margin=16` for inference (reflect mode)
- PSF modes: `"load"` (real from file), `"simulate"` (Gaussian σ=2.0)
- Deconv upsample: `scale=2` output, adjust crop_pad accordingly
- Hessian loss weight: `hess_w=0.02~0.05` suppresses artifacts
- Notebook execution order matters: run cells sequentially
