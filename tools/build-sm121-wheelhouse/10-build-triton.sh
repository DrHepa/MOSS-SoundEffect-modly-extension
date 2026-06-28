#!/usr/bin/env bash
set -euo pipefail

# Manual Triton wheel build for the local SM121 wheelhouse.
# NO descarga repos; espera que TRITON_SRC ya exista localmente.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm121}"
MAX_JOBS="${MAX_JOBS:-4}"
MODLY_ROOT="${MODLY_ROOT:-$HOME/Modly}"
SM121_SOURCE_ROOT="${SM121_SOURCE_ROOT:-$MODLY_ROOT/dependencies/sources/pytorch-sm121}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$MODLY_ROOT/dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312}"
TRITON_SRC="${TRITON_SRC:-$SM121_SOURCE_ROOT/triton}"
TRITON_REF="${TRITON_REF:-f797708c0626e5f9840ca5b0a98790e2c7cb09ad}"
PYTHON_BIN="${PYTHON_BIN:-$MODLY_ROOT/dependencies/build-venvs/pytorch-sm121-cp312/bin/python}"

export CUDA_HOME TORCH_CUDA_ARCH_LIST TRITON_OVERRIDE_ARCH MAX_JOBS

if [ ! -d "$TRITON_SRC" ]; then
  printf 'ERROR: TRITON_SRC does not exist: %s\n' "$TRITON_SRC" >&2
  printf 'Prepare a local Triton source tree manually at the recommended pin %s or an explicitly chosen fallback.\n' "$TRITON_REF" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  printf 'ERROR: ninja is not installed. This script does not install it.\n' >&2
  exit 1
fi

mkdir -p "$WHEELHOUSE_DIR"

printf 'Manual Triton build command prepared.\n'
printf 'Source tree: %s\n' "$TRITON_SRC"
printf 'Pin recomendado: %s\n' "$TRITON_REF"
printf 'Comando sugerido:\n'
printf '  %q -m pip wheel --no-deps --wheel-dir %q %q\n' "$PYTHON_BIN" "$WHEELHOUSE_DIR" "$TRITON_SRC"
