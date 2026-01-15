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
- [Configuration Reference](#configuration-reference)
- [Model & Dataset Registration](#model--dataset-registration)
- [Customization Guide](#customization-guide)
- [FAQ / Troubleshooting](#faq--troubleshooting)
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
│   ├── train_20
│   ├── train_30
│   ├── ...
│   └── train_100
└── train
    ├── 0001.png
    └── ...
```

Validation uses a similar structure:

```
valid_paired
├── valid_10
└── valid
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

## FAQ / Troubleshooting

**Q: “Model name `jdec` not found in registry.”**
- The registry keys are derived from `@register(...)` names. The core JDEC model is registered as `IPEC-decoder_dctform-rgb-share-size4`. If you see a missing key error, update the YAML `model.name` to match that key, or add a `@register('jdec')` alias in code.

**Q: “`dct_manip` build fails.”**
- Ensure libjpeg headers and shared objects are correctly referenced in `dct_manip/setup.py`. You must provide valid `include_dirs`, `library_dirs`, and `extra_objects` paths.

**Q: “CUDA out of memory.”**
- Reduce `batch_size`, `inp_size`, or encoder embedding sizes.

**Q: “Validation PSNR is NaN/inf.”**
- Verify input normalization in the wrapper and check for invalid coefficients (e.g., corrupted JPEGs).

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
