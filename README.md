# JDEC: JPEG Decoding via Enhanced Continuous Cosine Coefficient

Official implementation of **JDEC** from CVPR 2024: [JDEC: JPEG Decoding via Enhanced Continuous Cosine Coefficients](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_JDEC_JPEG_Decoding_via_Enhanced_Continuous_Cosine_Coefficients_CVPR_2024_paper.pdf) ([arXiv](https://arxiv.org/abs/2404.05558), [Project Page](https://wookyounghan.github.io/JDEC/)).

JDEC is a neural JPEG decoder that consumes **JPEG DCT coefficients + quantization tables**, not pixel-space images. It builds a continuous cosine representation and reconstructs RGB images with high PSNR and reduced artifacts.

![Overall Structure of Our JDEC](./static/images/Fig_4_ver_final_main.jpg)

---

## Contents

- [Key Ideas](#key-ideas)
- [Repository Structure](#repository-structure)
- [Environment & Dependencies](#environment--dependencies)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training Workflow](#training-workflow)
- [Testing / Evaluation Workflow](#testing--evaluation-workflow)
- [Project Workflow Deep Dive](#project-workflow-deep-dive)
- [Configuration Reference](#configuration-reference)
- [Model & Dataset Registration](#model--dataset-registration)
- [Customization Guide](#customization-guide)
- [How-to Recipes](#how-to-recipes)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Known Limitations & Inconsistencies](#known-limitations--inconsistencies)
- [Potential Next Steps](#potential-next-steps)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

---

## Key Ideas

- **Inputs are JPEG-domain**: JDEC uses compressed **DCT coefficients** and **quantization maps**, not RGB pixels.
- **Continuous cosine basis**: Features are mapped to a learnable cosine basis over block grids, enabling continuous-frequency reconstruction.
- **Two-stage decoding**:
  1. **Encoder** (SwinV2-based) embeds grouped DCT spectra.
  2. **Decoder** (MLP/1x1 conv) maps basis features to RGB.

---

## Repository Structure

```
.
├── configs/                  # YAML training configs
├── datasets/                 # Dataset definitions + wrappers
├── dct_manip/                # Custom libjpeg handler (needs compilation)
├── models/                   # JDEC model + encoder/decoder registries
├── utils/                    # DCT ops, transforms, helpers
├── train.py                  # Training entrypoint
├── test.py                   # Evaluation entrypoint
├── requirements.txt          # Pinned pip dependencies
└── environment.yaml          # Conda environment (recommended)
```

Key modules:
- **datasets/**: `image_folder_paired.py` and `wrappers_jpeg.py` define JPEG-coefficient datasets and input/GT wrappers.
- **models/**: `JDEC.py` (core model), `swinirv2.py` (encoder), `mlp.py` (decoder).
- **utils_.py**: training utilities, metrics (PSNR/PSNRB/SSIM), logging.

---

## Environment & Dependencies

This project was developed on **Ubuntu 20.04** with **Python 3.6**, **PyTorch 1.10**, and **CUDA 11.3**. See:
- `environment.yaml` for the full conda environment.
- `requirements.txt` for a minimal pip list.

Major dependencies:
- PyTorch, torchvision
- timm, einops
- opencv-python, Pillow
- jpegio, dct-manip (JPEG coefficient I/O)
- tensorboardX, tqdm

---

## Installation

### 1) Create environment

```bash
conda env create --file environment.yaml
conda activate jdec
```

### 2) Build `dct_manip`

`dct_manip` is a modified libjpeg handler required for DCT coefficient I/O:

1. Open `dct_manip/setup.py` and update:
   - `include_dirs` and `library_dirs`
   - `extra_objects` (path to `libjpeg.so`)
   - `headers` (path to `jpeglib.h`)
2. Build and install:

```bash
cd dct_manip
pip install .
```

---

## Data Preparation

The training/validation layout follows [FBCNN](https://github.com/jiaxi-jiang/FBCNN) with **paired JPEGs** at multiple qualities and **PNG ground-truth**.

Expected layout (quality-level folders):

```
jpeg_removal
├── train_paired
│   ├── train_10
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── train_20
│   │   ├── 0001.jpg
│   │   └── ...
│   ├── train_30
│   │   └── ...
│   ├── ...
│   └── train_100
│       └── ...
└── train
    ├── 0001.png
    └── ...
```

Notes:
- Each `train_<quality>` folder contains JPEG images (`.jpg`/`.jpeg`). GT images under `train/` are PNGs (`.png`). The loader reads all files in each folder, so keep only images there.
- Filenames (basenames) must match across qualities and GT (e.g., `train_paired/train_10/0001.jpg` ↔ `train/0001.png`) so that sorted ordering pairs the correct inputs/targets.

Validation uses a similar structure (single fixed quality):

```
valid_paired
├── valid_10
│   ├── 0001.jpg
│   └── ...
└── valid
    ├── 0001.png
    └── ...
```

> **Note:** Training randomly samples from JPEG qualities `[10, 20, ..., 100]` each iteration.

---

## Training Workflow

### Basic command

```bash
python train.py --config configs/train_JDEC.yaml --gpu 0
```

### What happens in training

1. **Dataset setup**
   - `train-paired-imageset` reads JPEGs at multiple qualities + GT PNGs.
   - `JDEC-decoder_toimage_rgb` wrapper:
     - Loads DCT coeffs + quant tables via `dct_manip.read_coefficients`.
     - Crops aligned JPEG blocks and GT patches.
     - Normalizes DCT values to `[-1, 1]`.

2. **Model forward pass**
   - Encoder: `swinv2_group_embedded` (SwinV2-based)
   - Decoder: `mlp_1dconv` (1x1 conv MLP)
   - Loss: L1 between predicted RGB and GT RGB.

3. **Outputs**
   - Checkpoints saved in `./save/<config_name>/`:
     - `epoch-last.pth` every epoch
     - `epoch-<N>.pth` for `epoch_save`
     - `epoch-best.pth` based on validation PSNR
   - Tensorboard logs in `./save/<config_name>/tensorboard/`

### Resume training

Set `resume` in YAML to the checkpoint:

```yaml
resume: ./save/_train_JDEC/epoch-last.pth
```

---

## Testing / Evaluation Workflow

`test.py` evaluates PSNR/PSNRB/SSIM on benchmark datasets (LIVE1, BSDS500, ICB).

### Basic command

```bash
python test.py
```

### Required edits in `test.py`

- Set dataset and paths:
  ```python
  setname = 'LIVE1'
  data_path = './PATH_TO_LIVE1'
  model_path = './PATH_TO_MODEL'
  ```
- By default the script evaluates at JPEG quality `q=30`.

### Outputs

- Prints averaged PSNR / PSNRB / SSIM.
- Saves decoded images under `./bin/<DATASET>/<QUALITY>/` when `save=True`.

---

## Project Workflow Deep Dive

This section follows the actual code paths to document invariants, data flow, and failure modes.

### End-to-end data flow map

1. **JPEG I/O (DCT domain)**  
   `dct_manip.read_coefficients` loads each JPEG and returns:
   - Quantization tables (`q_y`, `q_cbcr`)
   - Luma (Y) and chroma (CbCr) DCT coefficient blocks
2. **Data wrapping**  
   `JDEC-decoder_toimage_rgb`:
   - Randomly crops a block-aligned patch in **DCT space** and the corresponding **RGB GT**.
   - Dequantizes DCT by multiplying coefficients with the quantization tables.
   - Clamps coefficients to `[-1024, 1016]`, then normalizes to `[-1, 1]`.
   - Converts GT from BGR to RGB and shifts to `[-0.5, 0.5]`.
3. **Model forward**  
   `model(dct_y, dct_cbcr, q_map)` predicts RGB in the **same shifted range**.
4. **Loss / metrics**  
   - Training: `L1(pred_rgb, gt_rgb)`  
   - Validation: PSNR on the normalized/shifted tensors.

### Training chain (step-by-step)

1. **Config loading**: `train.py` reads YAML and sets `CUDA_VISIBLE_DEVICES`.
2. **Dataset + wrapper**:
   - `train-paired-imageset` selects a random JPEG quality for each sample.
   - `JDEC-decoder_toimage_rgb` performs block-aligned cropping and normalization.
3. **DataLoader**:
   - `shuffle=True` for train, `num_workers=8`, `pin_memory=True`.
4. **Forward pass**:
   - Inputs: `inp` (Y DCT), `chroma` (CbCr DCT), `dqt` (quant tables).
   - Output: predicted RGB patch in shifted range.
5. **Optimization**:
   - L1 loss; optimizer from `config.optimizer`.
   - Optional `MultiStepLR` scheduler.
6. **Checkpointing**:
   - `epoch-last.pth` every epoch.
   - `epoch-<N>.pth` every `epoch_save`.
   - `epoch-best.pth` on best validation PSNR.

### Validation chain (step-by-step)

1. Uses `valid-paired-dataset` (fixed JPEG quality) with the same wrapper.
2. Computes PSNR with `utils_.calc_psnr` on normalized tensors.
3. Writes metrics to Tensorboard under `psnr/valid`.

### Evaluation/testing chain (step-by-step)

`test.py` is a standalone evaluation script designed for benchmark datasets.

1. **Dataset selection**: `setname` chooses dataset and hard-coded path.
2. **Image conditioning**:
   - Pads input via symmetric flips to a fixed size (`size = 112*10`).
   - Writes a temporary JPEG (`./bin/temp_.jpg`) with quality `q`.
3. **DCT extraction**:
   - Reads DCT coefficients and quantization tables from the temp JPEG.
   - Dequantizes, clamps, and normalizes to `[-1, 1]`.
4. **Inference**:
   - Runs model, shifts output by `+0.5`, and crops to original size.
5. **Metrics + outputs**:
   - Calculates PSNR/PSNRB/SSIM vs. GT.
   - Saves predicted PNGs if `save=True`.

### Invariants & assumptions (critical for correctness)

- **Block alignment**: `inp_size` is in DCT block units (8x8), so images must be large enough for the sampled crop to be valid.
- **Range conventions**:
  - DCT coefficients are clamped to `[-1024, 1016]` and normalized to `[-1, 1]`.
  - GT RGB patches are in `[0, 1]` then shifted to `[-0.5, 0.5]`.
- **Data pairing**: JPEGs at all qualities must align with GT PNG filenames.
- **Path conventions**: `test.py` concatenates paths with `data_path + item`, so `data_path` must include a trailing `/`.

---

## Configuration Reference

The training config (`configs/train_JDEC.yaml`) drives the full pipeline.

### Top-level fields

| Field | Purpose |
|-------|---------|
| `train_dataset` | Training dataset + wrapper + batch size |
| `val_dataset` | Validation dataset + wrapper + batch size |
| `model` | Model architecture spec + encoder/decoder |
| `optimizer` | Optimizer name + hyperparameters |
| `epoch_max` | Total training epochs |
| `multi_step_lr` | LR scheduler settings |
| `epoch_val` | Validation frequency (epochs) |
| `epoch_save` | Checkpoint frequency (epochs) |
| `resume` | Path to checkpoint to resume from |

### Dataset spec

```yaml
train_dataset:
  dataset:
    name: train-paired-imageset
    args:
      root_path_inp: ./load/jpeg_removal/train_paired/train
      root_path_gt: ./load/jpeg_removal/train
      repeat: 5
      cache: bin
  wrapper:
    name: JDEC-decoder_toimage_rgb
    args:
      inp_size: 14
  batch_size: 16
```

**Key options**
- `repeat`: repeats the dataset (effective epoch length).
- `cache`: `none | bin | in_memory` (binary cache is recommended).
- `inp_size`: crop size in blocks (controls patch size).

### Model spec

```yaml
model:
  name: jdec
  args:
    encoder_spec:
      name: swinv2_group_embedded
      args:
        use_subblock: True
        emb_size: 256
        num_heads: [8,8,8,8,8]
    decoder_spec:
      name: mlp_1dconv
      args:
        out_dim: 3
        hidden_list: [512, 512, 512]
    hidden_dim: 512
```

**Important:** The model registry is keyed by `@register(...)` names. See [Model & Dataset Registration](#model--dataset-registration).

---

## Model & Dataset Registration

The repo uses lightweight registries for extensibility:

### Models
- Registry: `models/models.py` → `models` dict
- Registration: `@register('<name>')`
- Factory: `models.make(spec)`

Registered model names in code:
- `IPEC-decoder_dctform-rgb-share-size4` → JDEC core model
- `swinv2_group_embedded` → SwinV2 encoder
- `mlp` / `mlp_1dconv` → decoders

### Datasets & Wrappers
- Registry: `datasets/datasets.py`
- Registration: `@register('<name>')`
- Factory: `datasets.make(spec)`

Registered dataset/wrapper names in code:
- `train-paired-imageset`, `valid-paired-dataset`
- `image-folder-png`, `image-folder-embed-image`
- `JDEC-decoder_toimage_rgb`

---

## Customization Guide

### 1) Add a new dataset

```python
# datasets/my_dataset.py
from datasets import register
from torch.utils.data import Dataset

@register('my-dataset')
class MyDataset(Dataset):
    ...
```

Then update YAML:
```yaml
dataset:
  name: my-dataset
  args: { ... }
```

### 2) Add a new model

```python
# models/my_model.py
from models import register
import torch.nn as nn

@register('my-model')
class MyModel(nn.Module):
    ...
```

Then update YAML:
```yaml
model:
  name: my-model
  args: { ... }
```

### 3) Swap encoder / decoder

JDEC is defined as:
- `encoder_spec`: DCT encoder (SwinV2-based)
- `decoder_spec`: pixel-space decoder (MLP/Conv)

You can replace either spec with a registered model as long as input/output shapes are compatible.

---

## How-to Recipes

### Train on a new dataset (minimal steps)

1. Prepare paired data with the `train_paired/train_<quality>` and `train/` layout.  
2. Build and install `dct_manip`.
3. Update `configs/train_JDEC.yaml`:
   - `train_dataset.dataset.args.root_path_inp`
   - `train_dataset.dataset.args.root_path_gt`
   - `val_dataset` paths and `inp_size`
4. Launch training:
   ```bash
   python train.py --config configs/train_JDEC.yaml --gpu 0
   ```

### Evaluate a checkpoint on a benchmark

1. Edit `test.py`:
   - `setname = 'LIVE1'` (or `BSDS500`, `ICB`)
   - `data_path = './PATH_TO_LIVE1/'` (**must end with `/`**)
   - `model_path = './save/<run>/epoch-best.pth'`
2. Run:
   ```bash
   python test.py
   ```

### Change patch size / memory footprint

- **Increase speed / reduce memory**: lower `train_dataset.batch_size` or `wrapper.args.inp_size`.
- **Stability**: ensure `inp_size` does not exceed the valid DCT crop area, or random cropping can fail.

### Enable/disable caching

Set `cache` in dataset args:
- `none`: load from disk every time.
- `bin`: precompute `.pkl` caches (recommended for large datasets).
- `in_memory`: fastest but memory-heavy.

---

## FAQ / Troubleshooting

**Q: “Model name `jdec` not found in registry.”**
- The registry keys are derived from `@register(...)` names. The core JDEC model is registered as `IPEC-decoder_dctform-rgb-share-size4`. If you see a missing key error, update the YAML `model.name` to match that key, or add a `@register('jdec')` alias in code.

**Q: “`dct_manip` build fails.”**
- Ensure libjpeg headers and shared objects are correctly referenced in `dct_manip/setup.py`. You must provide valid `include_dirs`, `library_dirs`, and `extra_objects` paths.

**Q: “CUDA out of memory.”**
- Reduce `batch_size`, `inp_size`, or encoder embedding sizes.

**Q: “Validation PSNR is NaN/inf.”**
- Verify input normalization in the wrapper and check for invalid coefficients (e.g., corrupted JPEGs).

**Q: “Training finishes but tensorboard iterations look wrong.”**
- `train.py` and validation use hard-coded dataset sizes for the iteration counters (used only for tensorboard x-axis). If your dataset is not size 3450 (train) or 10 (val), the iteration index will be off. This does **not** affect training, but plots may look compressed or stretched.

**Q: “`test.py` can’t find images or errors with paths.”**
- `data_path` is concatenated with filenames (no `os.path.join`), so it **must end with a `/`**.

**Q: “`test.py` selects the wrong dataset path.”**
- The script uses `is` for string comparison in dataset selection; change it to `==` if you modify the script and see unexpected behavior.

**Q: “Cropping fails with a negative range error.”**
- Ensure training images are large enough for the chosen `inp_size` and that the image dimensions are multiples of 16 (because DCT blocks are 8x8 and chroma is subsampled).

**Q: “Metrics don’t match the paper.”**
- The validation loop computes PSNR on **normalized tensors** rather than on 8-bit RGB. For reproducibility vs. paper numbers, export predicted RGB to uint8 and re-evaluate externally.

---

## Known Limitations & Inconsistencies

- **Hard-coded dataset sizes** in training/validation for iteration counts (tensorboard only).  
- **`test.py` path concatenation** relies on a trailing slash.  
- **`test.py` string identity checks** (`is`) can be brittle.  
- **Single-quality evaluation**: `test.py` defaults to `q=30` only.  
- **Temporary JPEG file**: evaluation rewrites `./bin/temp_.jpg` for each image.

---

## Potential Next Steps

1. **Reproducibility upgrades**
   - Add deterministic seeding and explicit RNG controls.
   - Log DCT normalization ranges and quant tables per batch.
2. **Evaluation improvements**
   - Replace temporary JPEG file with in-memory encoding.
   - Add multi-quality evaluation and aggregate curves (PSNR vs. quality).
3. **Model/algorithm extensions**
   - Add uncertainty-aware decoding for ambiguous high-frequency coefficients.
   - Explore perceptual losses (LPIPS, DISTS) alongside L1.
4. **Engineering & scalability**
   - Replace hard-coded dataset sizes with `len(loader.dataset)`.
   - Add config-driven evaluation to eliminate code edits.
5. **Dataset robustness**
   - Expand paired datasets with diverse cameras and chroma subsampling modes.
   - Validate behavior under non-4:2:0 JPEGs.

---

## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif), [LTE](https://github.com/jaewon-lee-b/lte), [SwinIR](https://github.com/JingyunLiang/SwinIR), and [RGB No More](https://github.com/JeongsooP/RGB-no-more). We thank the authors for sharing their codes.

---

## BibTeX

```bibtex
@inproceedings{han2024jdec,
  title={JDEC: JPEG Decoding via Enhanced Continuous Cosine Coefficients},
  author={Han, Woo Kyoung and Im, Sunghoon and Kim, Jaedeok and Jin, Kyong Hwan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2784--2793},
  year={2024}
}
```
