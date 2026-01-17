# Agent Guidance for JDECoder (JDEC)

## Project snapshot (what matters most)
- **Goal**: JPEG-domain decoding + artifact reduction using DCT coefficients and quantization tables (not pixel-space). Core model is **JDEC** with SwinV2 DCT encoder and MLP/1×1 conv decoder. 【F:models/JDEC.py†L1-L63】【F:models/swinirv2.py†L715-L920】【F:models/mlp.py†L1-L44】
- **Entry points**:
  - Training: `train.py` (L1 loss, MultiStepLR, checkpointing). 【F:train.py†L1-L203】
  - Evaluation: `test.py` (PSNR/PSNRB/SSIM, temp JPEG, dataset path concatenation). 【F:test.py†L1-L133】
- **Registries**:
  - Models: `models.models` (`@register`, `make`). 【F:models/models.py†L1-L29】
  - Datasets: `datasets.datasets` (`@register`, `make`). 【F:datasets/datasets.py†L1-L25】

## Core architecture + data flow
- **JDEC forward**: `dct_y`, `cbcr`, `qmap` → SwinV2 encoder → learned cosine basis mixing → 1×1 conv MLP decoder → RGB. 【F:models/JDEC.py†L11-L63】
- **Default encoder**: `swinv2_group_embedded` (SwinV2 variant with DCT patch embedding and `out_dim`). 【F:models/swinirv2.py†L715-L920】
- **Default decoder**: `mlp_1dconv` for 1×1 conv MLP output head. 【F:models/mlp.py†L24-L44】
- **Wrapper + normalization**: `JDEC-decoder_toimage_rgb` crops block-aligned patches, dequantizes, clamps to `[-1024, 1016]`, normalizes to `[-1, 1]`; GT shifted to `[-0.5, 0.5]`. 【F:datasets/wrappers_jpeg.py†L16-L66】

## Datasets & caching
- JPEG coefficients + quantization: `ImageFolderJPEG_embed_image` via `dct_manip.read_coefficients`. 【F:datasets/image_folder_paired.py†L50-L147】
- Paired datasets:
  - `train-paired-imageset` (qualities 10–100) and `train-paired-imageset-small` (quality 90). 【F:datasets/image_folder_paired.py†L150-L229】
  - `valid-paired-dataset` for validation. 【F:datasets/image_folder_paired.py†L231-L247】
- Cache modes: `none`, `bin`, `in_memory`. 【F:datasets/image_folder_paired.py†L10-L147】

## Metrics, loss, and evaluation
- **Training loss**: L1 (`nn.L1Loss`). 【F:train.py†L71-L105】
- **Validation**: PSNR on normalized tensors. 【F:train.py†L109-L150】
- **Evaluation metrics**: PSNR, PSNRB (blocking effect), SSIM. 【F:utils_.py†L929-L1084】【F:test.py†L100-L133】

## Operational invariants & pitfalls
- **Block alignment**: DCT patches must align to 8×8 blocks; wrapper crops in block units. 【F:datasets/wrappers_jpeg.py†L34-L66】
- **Range conventions**: DCT clamp to `[-1024, 1016]` then normalize to `[-1, 1]`; GT RGB shifted by `-0.5`. 【F:datasets/wrappers_jpeg.py†L49-L66】
- **`test.py` path**: `data_path + item` (must include trailing `/`). 【F:test.py†L19-L27】【F:test.py†L50-L58】
- **Temp JPEG**: `test.py` writes `./bin/temp_.jpg` per image. 【F:test.py†L64-L83】

## Development workflow (preferred)
- **Train**: `python train.py --config configs/train_JDEC.yaml --gpu 0`. 【F:train.py†L1-L203】【F:configs/train_JDEC.yaml†L1-L48】
- **Eval**: Edit `test.py` dataset paths and run `python test.py`. 【F:test.py†L19-L133】
- **Dependencies**: PyTorch, timm, einops, jpegio/dct-manip, OpenCV, etc. 【F:requirements.txt†L1-L33】

## How to work in this repo (agent rules)
- Prefer **small diffs**, preserve data ranges, shapes, and block alignment.
- Verify **tensor semantics** (shape/dtype/device) when touching model or wrapper code.
- When changing registry names or configs, update both YAML and any registry aliasing (`models.make`). 【F:models/models.py†L1-L29】【F:configs/train_JDEC.yaml†L20-L35】
- Use evidence-based changes; cite file/line locations.
