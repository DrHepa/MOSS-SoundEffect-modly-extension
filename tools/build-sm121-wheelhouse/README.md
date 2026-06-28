# Build SM121 Wheelhouse

This directory documents the **manual maintainer workflow** for preparing a local PyTorch/Triton wheelhouse for Linux `aarch64`, Python `3.12`, CUDA `13.x`, and NVIDIA `SM 12.1+` class GPUs.

These scripts are not run by `setup.py`. They do not install system packages, do not clone source repositories, and do not perform privileged operations. They are helper commands for maintainers who need to produce a prebuilt wheelhouse that setup can later consume.

## Goal

- Relative wheelhouse path from Modly root: `dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312`
- Required setup manifest inside that directory: `WHEELHOUSE.json`
- Setup consumes the wheelhouse only after it already exists and has been validated.

## Expected manual prerequisites

- Linux `aarch64`
- Python `3.12.x`
- CUDA `13.x` toolkit available through `CUDA_HOME`
- NVIDIA GPU class requiring explicit `SM 12.1+` source-built support
- `TORCH_CUDA_ARCH_LIST="12.1a"`
- Optional Triton override: `TRITON_OVERRIDE_ARCH=sm121`
- `ninja` available in the build environment
- A dedicated build virtual environment separate from Modly's runtime environment

## Recommended order

1. Run `00-check-host.sh`.
2. Prepare local source trees manually for Triton, PyTorch, torchaudio, and torchvision.
3. Run `10-build-triton.sh`.
4. Run `20-build-pytorch.sh`.
5. Copy or adapt `WHEELHOUSE.example.json` to `WHEELHOUSE.json`.
6. Run `30-validate-wheelhouse.sh`.

## Triton pin

- Recommended pin: `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`
- Fallback: a current upstream branch, only if the recommended pin fails in your local source tree.

## Important notes

- These scripts are for manual maintainer execution only.
- The extension setup never compiles PyTorch or Triton automatically.
- Do not rely on broad GPU-family aliases for this wheelhouse; use the explicit CUDA arch list required by the target class.
- If you create `WHEELHOUSE.json`, keep wheel filenames, package specifiers, and hashes synchronized with the actual files.

## Default source tree layout

The scripts use these defaults, all overrideable through environment variables:

- `MODLY_ROOT=${HOME}/Modly`
- `SM121_SOURCE_ROOT=${MODLY_ROOT}/dependencies/sources/pytorch-sm121`
- `TRITON_SRC=${SM121_SOURCE_ROOT}/triton`
- `PYTORCH_SRC=${SM121_SOURCE_ROOT}/pytorch`
- `TORCHAUDIO_SRC=${SM121_SOURCE_ROOT}/audio`
- `TORCHVISION_SRC=${SM121_SOURCE_ROOT}/vision`
- `WHEELHOUSE_DIR=${MODLY_ROOT}/dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312`

## Dedicated build environment

The scripts default to:

`PYTHON_BIN=${MODLY_ROOT}/dependencies/build-venvs/pytorch-sm121-cp312/bin/python`

That environment is only for building wheels. It is not Modly's runtime environment and is not the extension runtime environment.
