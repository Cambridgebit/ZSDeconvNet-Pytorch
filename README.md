# ZSDeconvNet PyTorch

Zero-shot deconvolution for microscopy image processing.

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install torch numpy Pillow matplotlib scipy
```

## Quick Start

### Denoise Training
```bash
python scripts/train_denoise.py --image datasets/Microtubule/train_data/01.tif --epochs 30
```

### Deconv Training (with denoised image)
```bash
python scripts/train_deconv.py --image datasets/Microtubule/train_data/01.tif --psf datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif --epochs 30
```

### Joint Training
```bash
python scripts/train_joint.py --image datasets/Microtubule/train_data/01.tif --psf datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif --epochs 30
```

### Two-Stage Joint Training
```bash
python scripts/train_joint_two_stage.py --image datasets/Microtubule/train_data/01.tif --psf datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif
```

### Inference
```bash
# Denoise only
python scripts/infer_denoise.py --image test.tif --checkpoint checkpoints/best_denoise.pt

# Deconv only
python scripts/infer_deconv.py --image test.tif --checkpoint checkpoints/best_deconv.pt --psf datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif

# Two-stage (denoise → deconv)
python scripts/infer_two_stage.py --image test.tif --denoise_ckpt checkpoints/best_denoise.pt --deconv_ckpt checkpoints/best_deconv.pt --psf datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif
```

## Project Structure

```
ZSDeconvNet/
├── zsdeconv/                 # Core package
│   ├── __init__.py
│   ├── models.py             # Model architectures
│   ├── data.py               # Datasets
│   ├── loss.py               # Loss functions
│   ├── utils.py              # Utilities
│   └── psf.py                # PSF loading
├── scripts/                  # Training & inference scripts
│   ├── train_denoise.py
│   ├── train_deconv.py
│   ├── train_joint.py
│   ├── train_joint_two_stage.py
│   ├── infer_denoise.py
│   ├── infer_deconv.py
│   └── infer_two_stage.py
├── checkpoints/              # Pre-trained models
├── datasets/                 # Example datasets
├── notebooks/                # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Notebooks

The original notebooks are available in `notebooks/` directory:
- `denoise_training.ipynb` - Denoise-only training
- `deconv_training.ipynb` - Deconv-only training
- `denoise_to_deconv.ipynb` - Two-stage pipeline
- `joint_denoise_deconv_training.ipynb` - Joint training
- `joint_training_two_stage.ipynb` - Two-stage joint training
- `noise2void_training.ipynb` - Noise2Void denoising
- `psf_visualization.ipynb` - PSF analysis

## Citation

```bibtex
@misc{zsdeconvnet,
  title={ZSDeconvNet: Zero-shot Deconvolution for Microscopy},
  year={2024},
  url={https://github.com/Cambridgebit/ZSDeconvNet-Pytorch}
}
```
