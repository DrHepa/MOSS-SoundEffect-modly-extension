from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


EXTENSION_ID = "moss-soundeffect-v2-process-extension"
HF_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
UPSTREAM_GIT = "https://github.com/OpenMOSS/MOSS-TTS.git@main#subdirectory=moss_soundeffect_v2"
MODEL_OWNER_REL = Path(EXTENSION_ID) / "generate-soundeffect"
LEGACY_MODEL_ROOT_REL = Path("models/openmoss-team/moss-soundeffect-v2_0")
WORKFLOW_OUTPUT_REL = Path("Workflows/MOSS-SoundEffect")
SETUP_SENTINEL_REL = Path(".modly/setup-ready.json")
LOCAL_WHEELHOUSE_REL = Path("dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312")
LOCAL_WHEELHOUSE_MANIFEST = "WHEELHOUSE.json"
TEXT_ENCODER_INDEX_REL = Path("text_encoder/model.safetensors.index.json")
REQUIRED_SENTINELS = [
    Path("model_index.json"),
    Path("transformer/diffusion_pytorch_model.safetensors"),
    Path("vae/vae_128d_48k.pth"),
    Path("scheduler/scheduler_config.json"),
    Path("tokenizer/tokenizer.json"),
    TEXT_ENCODER_INDEX_REL,
]


class ExtensionError(RuntimeError):
    def __init__(self, message: str, code: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def emit_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def bool_from_any(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sanitize_filename(value: str, fallback: str = "moss-soundeffect") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip(".-_")
    if not text:
        text = fallback
    return text[:80]


def ensure_relative_child(root: Path, relative_path: str | Path) -> Path:
    if isinstance(relative_path, Path):
        candidate_rel = relative_path
    else:
        candidate_rel = Path(str(relative_path))
    if candidate_rel.is_absolute():
        raise ExtensionError(
            "Absolute paths are not allowed for workspace outputs.",
            code="unsafe_output_path",
            details={"path": str(candidate_rel)},
        )
    root_resolved = root.resolve()
    candidate_resolved = (root / candidate_rel).resolve()
    if os.path.commonpath([str(root_resolved), str(candidate_resolved)]) != str(root_resolved):
        raise ExtensionError(
            "Output path traversal is not allowed.",
            code="unsafe_output_path",
            details={"path": str(candidate_rel)},
        )
    return candidate_resolved


def extension_root() -> Path:
    return Path(__file__).resolve().parent.parent


def modly_root_from_extension(ext_root: Path | None = None) -> Path | None:
    root = (ext_root or extension_root()).resolve()
    modly_root_env = os.environ.get("MODLY_ROOT")
    if modly_root_env:
        return Path(modly_root_env).expanduser().resolve()
    if root.parent.name == "extensions":
        return root.parent.parent
    return None


def global_models_root(ext_root: Path | None = None) -> Path | None:
    modly_root = modly_root_from_extension(ext_root)
    if modly_root is None:
        return None
    return modly_root / "models"


def dependencies_root(ext_root: Path | None = None) -> Path | None:
    override = os.environ.get("MODLY_DEPENDENCIES_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    modly_root = modly_root_from_extension(ext_root)
    if modly_root is not None:
        return modly_root / "dependencies"
    return None


def local_wheelhouse_dir(ext_root: Path | None = None) -> Path:
    deps_root = dependencies_root(ext_root)
    if deps_root is not None:
        return deps_root / "wheelhouse" / LOCAL_WHEELHOUSE_REL.name
    return extension_root() / LOCAL_WHEELHOUSE_REL


def local_wheelhouse_manifest_path(ext_root: Path | None = None) -> Path:
    return local_wheelhouse_dir(ext_root) / LOCAL_WHEELHOUSE_MANIFEST


def model_root() -> Path:
    ext_root = extension_root()
    models_root = global_models_root(ext_root)
    if models_root is not None:
        return models_root / MODEL_OWNER_REL
    return ext_root / LEGACY_MODEL_ROOT_REL


def legacy_model_root() -> Path:
    return extension_root() / LEGACY_MODEL_ROOT_REL


def setup_sentinel_path() -> Path:
    return extension_root() / SETUP_SENTINEL_REL


def read_request_stdin() -> dict[str, Any]:
    raw = sys.stdin.readline()
    if not raw:
        raise ExtensionError("No JSON request received on stdin.", code="missing_request")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtensionError(
            "Invalid JSON request on stdin.",
            code="invalid_request_json",
            details={"error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise ExtensionError("Request payload must be a JSON object.", code="invalid_request")
    return payload


def validate_model_files(root: Path) -> list[str]:
    missing = [str(path) for path in REQUIRED_SENTINELS if not (root / path).is_file()]
    index_path = root / TEXT_ENCODER_INDEX_REL
    shard_matches = list((root / "text_encoder").glob("model-*.safetensors"))
    if index_path.is_file() and not shard_matches:
        missing.append("text_encoder/model-*.safetensors")
    return missing
