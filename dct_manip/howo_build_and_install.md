### 2) Build `dct_manip`

[](#2-build-dct_manip)

`dct_manip` is a modified libjpeg handler required for DCT coefficient I/O:

1.  Open `dct_manip/setup.py` and update:
    - `include_dirs` and `library_dirs`
    - `extra_objects` (path to `libjpeg.so`)
    - `headers` (path to `jpeglib.h`)
2.  Build and install:

```bash
cd dct_manip
pip install .
```
