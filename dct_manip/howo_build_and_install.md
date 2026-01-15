# Building and Installing `dct_manip` (Linux, x86_64)

This project ships a custom C++/PyTorch extension (`dct_manip`) that wraps **libjpeg** to expose JPEG DCT coefficient I/O and related helpers. It is required by the data pipeline (e.g., dataset loaders and transforms) and is imported by training/evaluation scripts, so the Python module **must build successfully** before you can train or run evaluation. 

> **What it does (indirectly)**
> - Dataset loaders call `dct_manip.read_coefficients(...)` to load DCT coefficients + quantization tables from JPEG files.
> - Other helpers (e.g., transforms) also rely on `dct_manip` for coefficient decoding.
>
> In short: if `dct_manip` is missing, the data pipeline cannot read JPEG-domain inputs.

---

## 1) Mental model: build invariants & failure modes

**Invariants (must hold):**
- A **C++17** toolchain is available (`g++`/`clang++`).
- **libjpeg** headers (`jpeglib.h`) and shared library (`libjpeg.so`) are installed and discoverable.
- **PyTorch** is installed in the same Python environment used to build the extension.
- The `setup.py` paths (`include_dirs`, `library_dirs`, `extra_objects`, `headers`) point to the actual libjpeg + PyTorch headers/libs.

**Common failure modes:**
- `fatal error: jpeglib.h: No such file or directory` → libjpeg headers not installed or wrong `include_dirs`.
- `ld: cannot find -ljpeg` or missing `libjpeg.so` → wrong `library_dirs`/`extra_objects` or missing libjpeg runtime.
- `undefined symbol: jpeg_*` at import time → wrong libjpeg ABI or mismatched libjpeg version.
- `fatal error: torch/extension.h` → PyTorch not installed or wrong Python env.

---

## 2) Prerequisites (Arch Linux, x86_64)

**System dependencies (recommended):**

```bash
sudo pacman -S --needed base-devel python python-pip python-setuptools python-wheel \
  libjpeg-turbo
```

- `base-devel` → compiler, make, etc.
- `libjpeg-turbo` → provides `jpeglib.h` and `libjpeg.so` on Arch.

**PyTorch (CPU or CUDA):**

- If you use the Arch package:
  ```bash
  sudo pacman -S --needed python-pytorch
  ```
- If you prefer pip/venv/conda, install PyTorch in that environment and use **the same Python** to build `dct_manip`.

> **Invariant:** the Python environment used to build `dct_manip` must be the same one used to import it later.

---

## 3) Locate libjpeg + PyTorch include/lib paths

You must update `dct_manip/setup.py` with actual paths. For Arch, defaults are usually:
- Headers: `/usr/include/jpeglib.h`
- Library: `/usr/lib/libjpeg.so`

To **verify paths**, run:

```bash
# libjpeg paths
pacman -Ql libjpeg-turbo | rg 'jpeglib.h|libjpeg.so'

# PyTorch include paths (same Python env you will use)
python - <<'PY'
import torch.utils.cpp_extension as ce
print("Torch include paths:", ce.include_paths())
print("Torch library paths:", ce.library_paths())
PY
```

You will use these paths in `setup.py`.

---

## 4) Edit `setup.py` (critical step)

Open `dct_manip/setup.py` and update the following fields:

```python
include_dirs=["/usr/include", ...],
library_dirs=["/usr/lib", ...],
extra_objects=["/usr/lib/libjpeg.so"],
headers=["/usr/include/jpeglib.h"],
```

**Notes:**
- You can add PyTorch include paths too, but `torch.utils.cpp_extension` usually injects them.
- `extra_objects` must be the **full path** to `libjpeg.so` (or `libjpeg.so.*`).
- If you use conda/venv, your paths may be under `.../envs/<name>/include` and `.../envs/<name>/lib`.

---

## 5) Build and install (recommended flow)

```bash
cd dct_manip
pip install .
```

This compiles and installs the extension into your active Python environment.

**Alternative (debug build):**
```bash
python setup.py build_ext --inplace
```

This leaves the compiled `.so` in the local `dct_manip/` directory.

---

## 6) Validate the install

```bash
python - <<'PY'
import dct_manip as dm
print("dct_manip loaded:", dm)
print("Available functions:", [f for f in dir(dm) if not f.startswith('_')])
PY
```

Optional runtime sanity check (requires a JPEG file):
```bash
python - <<'PY'
import dct_manip as dm
# Replace with a real JPEG path
path = "path/to/sample.jpg"
# read_coefficients returns (dimensions, quant_tables, Y_coeffs, CbCr_coeffs?)
print(dm.read_coefficients(path)[0])
PY
```

---

## 7) How `dct_manip` is used in this project

- Data loaders call `dct_manip.read_coefficients()` to fetch **DCT coefficients** and **quantization tables** from JPEG files.
- Training and evaluation depend on these JPEG-domain tensors; the model consumes them directly.

If `dct_manip` fails to build, dataset loading will fail before training starts.

---

## 8) FAQ / Troubleshooting

### Q1) `fatal error: jpeglib.h: No such file or directory`
**Cause:** libjpeg headers missing or `include_dirs` wrong.
**Fix:**
- On Arch: `sudo pacman -S libjpeg-turbo`.
- Ensure `setup.py` includes `/usr/include` and the correct `jpeglib.h` path.

### Q2) `ld: cannot find -ljpeg` or missing `libjpeg.so`
**Cause:** libjpeg shared library not found or wrong `library_dirs`/`extra_objects`.
**Fix:**
- Ensure `libjpeg.so` exists (`pacman -Ql libjpeg-turbo | rg libjpeg.so`).
- Set `extra_objects` to the full library path (e.g., `/usr/lib/libjpeg.so`).

### Q3) `ImportError: undefined symbol: jpeg_*`
**Cause:** ABI mismatch (wrong libjpeg version) or mixing environments.
**Fix:**
- Prefer **libjpeg-turbo** (Arch default) and rebuild the extension.
- Ensure you are not mixing conda’s libjpeg with system libjpeg.

### Q4) `fatal error: torch/extension.h` or `torch` not found
**Cause:** PyTorch not installed in the active Python environment.
**Fix:**
- Install PyTorch in the same environment and rebuild.

### Q5) `C++17` or compiler errors
**Cause:** old compiler or missing toolchain.
**Fix:**
- Install `base-devel` on Arch.
- Ensure `g++ --version` is recent enough (GCC ≥ 9 is safe).

### Q6) Build succeeds but import fails (`ModuleNotFoundError`)
**Cause:** Built under a different Python environment.
**Fix:**
- Activate the correct env before building.
- Rebuild with the same `python` that will import `dct_manip`.

---

## 9) Notes on configuring the build chain

- The extension uses **PyTorch C++ extensions** and **pybind11** via `torch.utils.cpp_extension`.
- `extra_compile_args=['-std=c++17']` enforces C++17.
- Linking is explicit via `extra_objects` (libjpeg), so the path must be correct.

If you want to customize the build (e.g., different libjpeg install prefix), update `setup.py` accordingly, then rebuild.



Invariants for a successful build:

1. `jpeglib.h` and `libjpeg.so` must match (same version family).
2. The include/library paths in `setup.py` must point to the environment where those files live.
3. The Python used to run `pip install .` must match the Python where PyTorch is installed.

If any of these invariants break, the build fails (most common: “cannot find jpeglib.h” or undefined libjpeg symbols at link time).

---

## Prerequisites (Arch Linux)

Install a build toolchain, Python dev headers, and libjpeg:

```bash
sudo pacman -S --needed base-devel python python-pip python-virtualenv \
  python-setuptools python-wheel cmake ninja git libjpeg-turbo
```

Notes:
- `base-devel` provides `gcc`, `g++`, `make`, and related tools.
- `libjpeg-turbo` is ABI-compatible with libjpeg and provides `jpeglib.h` + `libjpeg.so`.

### Optional: use Conda or venv

If you use conda/venv, install **PyTorch** in that environment *before* building the extension. The extension build uses the **active Python** and that Python must be the one with PyTorch installed.

---

## Step-by-step build (Arch Linux)

### 1) Create and activate a Python environment

Pick one:

**Option A: venv (lightweight)**
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

**Option B: conda**
```bash
conda env create --file environment.yaml
conda activate jdec
```

### 2) Install PyTorch

Use the PyTorch version that matches your CUDA setup. For CPU-only builds, install the CPU wheel.

Example (CUDA 11.8 on Arch):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3) Locate libjpeg headers/libs

On Arch with `libjpeg-turbo`, the defaults are typically:
- Headers: `/usr/include` (contains `jpeglib.h`)
- Library: `/usr/lib/libjpeg.so`

Confirm:
```bash
ls /usr/include/jpeglib.h
ls /usr/lib/libjpeg.so
```

### 4) Edit `dct_manip/setup.py`

You **must** set the correct include/lib paths for your environment. Update:

- `include_dirs`: folder containing `jpeglib.h`
- `library_dirs`: folder containing `libjpeg.so`
- `extra_objects`: full path to `libjpeg.so`
- `headers`: full path to `jpeglib.h`

Example for Arch system install:
```python
include_dirs=['/usr/include']
library_dirs=['/usr/lib']
extra_objects=['/usr/lib/libjpeg.so']
headers=['/usr/include/jpeglib.h']
```

Example for conda (replace `$CONDA_PREFIX`):
```python
include_dirs=['/path/to/conda/env/include']
library_dirs=['/path/to/conda/env/lib']
extra_objects=['/path/to/conda/env/lib/libjpeg.so']
headers=['/path/to/conda/env/include/jpeglib.h']
```
