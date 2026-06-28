#!/usr/bin/env bash
set -euo pipefail

# Manual PyTorch/torchaudio/torchvision wheel build for the local SM121 wheelhouse.
# NO descarga repos; espera source trees locales ya preparados.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm121}"
MAX_JOBS="${MAX_JOBS:-4}"
MODLY_ROOT="${MODLY_ROOT:-$HOME/Modly}"
SM121_SOURCE_ROOT="${SM121_SOURCE_ROOT:-$MODLY_ROOT/dependencies/sources/pytorch-sm121}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$MODLY_ROOT/dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312}"
PYTORCH_SRC="${PYTORCH_SRC:-$SM121_SOURCE_ROOT/pytorch}"
TORCHAUDIO_SRC="${TORCHAUDIO_SRC:-$SM121_SOURCE_ROOT/audio}"
TORCHVISION_SRC="${TORCHVISION_SRC:-$SM121_SOURCE_ROOT/vision}"
PYTHON_BIN="${PYTHON_BIN:-$MODLY_ROOT/dependencies/build-venvs/pytorch-sm121-cp312/bin/python}"

export CUDA_HOME TORCH_CUDA_ARCH_LIST TRITON_OVERRIDE_ARCH MAX_JOBS

if [ ! -d "$PYTORCH_SRC" ]; then
  printf 'ERROR: PYTORCH_SRC does not exist: %s\n' "$PYTORCH_SRC" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  printf 'ERROR: ninja is not installed. This script does not install it.\n' >&2
  exit 1
fi

mkdir -p "$WHEELHOUSE_DIR"

printf 'TORCH_CUDA_ARCH_LIST=%s\n' "$TORCH_CUDA_ARCH_LIST"
printf 'TRITON_OVERRIDE_ARCH=%s\n' "$TRITON_OVERRIDE_ARCH"
printf 'MAX_JOBS=%s\n' "$MAX_JOBS"
printf 'Comandos sugeridos para ejecuci\u00f3n manual, uno por repo:\n'
printf '  %q -m pip wheel --no-deps --wheel-dir %q %q\n' "$PYTHON_BIN" "$WHEELHOUSE_DIR" "$PYTORCH_SRC"
printf '  %q -m pip wheel --no-deps --wheel-dir %q %q\n' "$PYTHON_BIN" "$WHEELHOUSE_DIR" "$TORCHAUDIO_SRC"
printf '  %q -m pip wheel --no-deps --wheel-dir %q %q\n' "$PYTHON_BIN" "$WHEELHOUSE_DIR" "$TORCHVISION_SRC"

if [ ! -d "$TORCHAUDIO_SRC" ] || [ ! -d "$TORCHVISION_SRC" ]; then
  printf 'WARN: TORCHAUDIO_SRC and/or TORCHVISION_SRC do not exist yet. Adjust variables before manual execution.\n' >&2
fi
