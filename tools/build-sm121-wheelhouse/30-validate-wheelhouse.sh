#!/usr/bin/env bash
set -euo pipefail

# Statically validates the local wheelhouse expected by setup.py.
# NO instala nada y NO ejecuta inferencia.

MODLY_ROOT="${MODLY_ROOT:-$HOME/Modly}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$MODLY_ROOT/dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312}"
MANIFEST_PATH="${MANIFEST_PATH:-$WHEELHOUSE_DIR/WHEELHOUSE.json}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [ ! -d "$WHEELHOUSE_DIR" ]; then
  printf 'ERROR: wheelhouse does not exist: %s\n' "$WHEELHOUSE_DIR" >&2
  exit 1
fi

if [ ! -f "$MANIFEST_PATH" ]; then
  printf 'ERROR: falta el manifest WHEELHOUSE.json: %s\n' "$MANIFEST_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY' "$MANIFEST_PATH" "$WHEELHOUSE_DIR"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
wheelhouse_dir = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {"torch", "torchaudio", "torchvision", "triton"}
packages = manifest.get("packages") or []
package_map = {}
for item in packages:
    if isinstance(item, dict) and item.get("name"):
        package_map[str(item["name"])] = item
missing_packages = sorted(required - set(package_map))
missing_files = []
for item in package_map.values():
    filename = str(item.get("filename") or "").strip()
    if filename and not (wheelhouse_dir / filename).is_file():
        missing_files.append(filename)
if missing_packages or missing_files:
    raise SystemExit(
        "Manifest incompleto. missing_packages=%s missing_files=%s" % (missing_packages, sorted(missing_files))
    )
print(json.dumps({
    "status": "ok",
    "wheelhouse_dir": str(wheelhouse_dir),
    "manifest": str(manifest_path),
    "packages": sorted(package_map),
}, ensure_ascii=True))
PY
