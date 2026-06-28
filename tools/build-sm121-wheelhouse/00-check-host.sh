#!/usr/bin/env bash
set -euo pipefail

# Verificacion previa del host para build manual.
# NO instala paquetes, NO descarga repos y NO ejecuta builds.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm121}"
MAX_JOBS="${MAX_JOBS:-4}"
MODLY_ROOT="${MODLY_ROOT:-$HOME/Modly}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$MODLY_ROOT/dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
printf 'TORCH_CUDA_ARCH_LIST=%s\n' "$TORCH_CUDA_ARCH_LIST"
printf 'TRITON_OVERRIDE_ARCH=%s\n' "$TRITON_OVERRIDE_ARCH"
printf 'MAX_JOBS=%s\n' "$MAX_JOBS"
printf 'WHEELHOUSE_DIR=%s\n' "$WHEELHOUSE_DIR"
printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"

uname -a
"$PYTHON_BIN" -V

if [ ! -d "$CUDA_HOME" ]; then
  printf 'ERROR: CUDA_HOME does not exist: %s\n' "$CUDA_HOME" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'ERROR: nvidia-smi is not available.\n' >&2
  exit 1
fi

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

if ! command -v ninja >/dev/null 2>&1; then
  printf 'WARN: ninja is not installed. Source builds remain blocked until this is resolved manually.\n' >&2
fi

mkdir -p "$WHEELHOUSE_DIR"
printf 'Host verificado. Revisa la salida antes de ejecutar los scripts de build.\n'
