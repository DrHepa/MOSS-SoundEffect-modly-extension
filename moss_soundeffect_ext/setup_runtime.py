from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

from .common import (
    EXTENSION_ID,
    HF_REPO,
    MODEL_OWNER_REL,
    REQUIRED_SENTINELS,
    UPSTREAM_GIT,
    emit_json,
    extension_root,
    global_models_root,
    legacy_model_root,
    validate_model_files,
)
from .host_compat import resolve_host_compat


PINNED_PACKAGES = [
    "filelock>=3.0",
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2>=3.0",
    "fsspec>=0.8.5",
    "numpy==1.26.4",
    "einops==0.8.2",
    "pillow==12.2.0",
    "tqdm==4.67.3",
    "safetensors==0.7.0",
    "transformers==4.57.1",
    "diffusers==0.37.1",
    "ftfy==6.3.1",
    "regex==2026.4.4",
    "soundfile==0.13.1",
    "imageio==2.37.3",
    "descript-audiotools==0.7.2",
    "huggingface_hub>=0.30.0",
]

SETUP_SCHEMA_VERSION = 2
SETUP_BEHAVIOR_VERSION = 1


def _read_wheelhouse_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Wheelhouse manifest is missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Wheelhouse manifest is invalid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Wheelhouse manifest must be a JSON object: {manifest_path}")
    return payload


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_requirements_hash(lane: dict[str, Any]) -> str:
    payload = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "behavior_version": SETUP_BEHAVIOR_VERSION,
        "lane_id": str(lane.get("id") or ""),
        "install_source": str(lane.get("install_source") or ""),
        "torch_index_url": str(lane.get("torch_index_url") or ""),
        "torch_packages": [str(item) for item in lane.get("torch_packages") or []],
        "required_wheels": [str(item) for item in lane.get("required_wheels") or []],
        "cuda_variant": str(lane.get("cuda_variant") or ""),
        "pinned_packages": list(PINNED_PACKAGES),
        "upstream_git": UPSTREAM_GIT,
    }
    return _hash_payload(payload)


def _model_assets_signature() -> str:
    payload = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "hf_repo": HF_REPO,
        "required_sentinels": [str(path) for path in REQUIRED_SENTINELS],
    }
    return _hash_payload(payload)


def _read_previous_sentinel(ext_dir: Path) -> dict[str, Any] | None:
    sentinel_path = ext_dir / ".modly/setup-ready.json"
    if not sentinel_path.is_file():
        return None
    try:
        payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _runtime_reuse_decision(
    previous: dict[str, Any] | None,
    lane: dict[str, Any],
    runtime_requirements_hash: str,
    venv_python: Path,
    *,
    force_reinstall: bool,
) -> dict[str, Any]:
    decision = {
        "skip_runtime_install": False,
        "reason": "no_previous_ready_sentinel",
        "previous_status": None,
    }
    if force_reinstall:
        decision["reason"] = "force_reinstall"
        return decision
    if previous is None:
        return decision

    previous_status = str(previous.get("status") or "")
    decision["previous_status"] = previous_status
    if previous_status not in {"ready", "runtime_ready_model_missing"}:
        decision["reason"] = f"previous_status_{previous_status or 'unknown'}"
        return decision

    try:
        previous_schema_version = int(previous.get("setup_schema_version") or 0)
    except (TypeError, ValueError):
        previous_schema_version = 0
    previous_lane = previous.get("lane") if isinstance(previous.get("lane"), dict) else {}
    if str(previous_lane.get("id") or "") != str(lane.get("id") or ""):
        decision["reason"] = "lane_changed"
        return decision
    if not venv_python.exists():
        decision["reason"] = "venv_python_missing"
        return decision

    # Migration path for sentinels written before setup hashes existed. Reuse
    # only when the previous state was runnable, the selected lane still matches,
    # and the venv python exists. The post-install probe still runs immediately;
    # if that probe fails, setup falls back to a normal reinstall/repair.
    if previous_schema_version == 0 and not previous.get("runtime_requirements_hash"):
        decision["skip_runtime_install"] = True
        decision["reason"] = "reused_legacy_ready_runtime_probe_required"
        return decision

    if previous_schema_version != SETUP_SCHEMA_VERSION:
        decision["reason"] = "schema_version_changed"
        return decision
    if str(previous.get("runtime_requirements_hash") or "") != runtime_requirements_hash:
        decision["reason"] = "runtime_requirements_hash_changed"
        return decision

    decision["skip_runtime_install"] = True
    decision["reason"] = "reused_ready_runtime"
    return decision


def _verify_wheelhouse(lane: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
    wheelhouse_dir = Path(str(lane.get("wheelhouse_dir") or "")).expanduser()
    manifest_path = Path(str(lane.get("wheelhouse_manifest") or "")).expanduser()
    if not wheelhouse_dir.is_dir():
        raise RuntimeError(f"Wheelhouse directory does not exist: {wheelhouse_dir}")
    if not manifest_path.is_file():
        raise RuntimeError(f"Wheelhouse manifest does not exist: {manifest_path}")

    manifest = _read_wheelhouse_manifest(manifest_path)
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError(f"Wheelhouse manifest has no packages array: {manifest_path}")

    package_map: dict[str, dict[str, Any]] = {}
    missing_files: list[str] = []
    for item in packages:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        filename = str(item.get("filename") or "").strip()
        if not name:
            continue
        package_map[name] = item
        if filename and not (wheelhouse_dir / filename).is_file():
            missing_files.append(filename)

    required_wheels = [str(item) for item in lane.get("required_wheels") or []]
    missing_packages = [name for name in required_wheels if name not in package_map]
    if missing_packages:
        raise RuntimeError(f"Wheelhouse manifest is missing required packages: {', '.join(missing_packages)}")
    if missing_files:
        raise RuntimeError(f"Wheelhouse files are missing from {wheelhouse_dir}: {', '.join(sorted(set(missing_files)))}")

    expected_python_tag = str(host.get("python", {}).get("python_tag") or "")
    manifest_python_tag = str(manifest.get("python_tag") or "")
    if expected_python_tag and manifest_python_tag and manifest_python_tag != expected_python_tag:
        raise RuntimeError(f"Wheelhouse python_tag mismatch: manifest={manifest_python_tag} host={expected_python_tag}")

    manifest_platform_tag = str(manifest.get("platform_tag") or "")
    expected_platform_tag = "linux_aarch64" if host.get("os") == "linux" and host.get("arch") == "aarch64" else ""
    if expected_platform_tag and manifest_platform_tag and manifest_platform_tag != expected_platform_tag:
        raise RuntimeError(f"Wheelhouse platform_tag mismatch: manifest={manifest_platform_tag} host={expected_platform_tag}")

    manifest_cuda_variant = str(manifest.get("cuda_variant") or "")
    if lane.get("cuda_variant") and manifest_cuda_variant and manifest_cuda_variant != lane["cuda_variant"]:
        raise RuntimeError(f"Wheelhouse cuda_variant mismatch: manifest={manifest_cuda_variant} lane={lane['cuda_variant']}")

    manifest_min_sm = manifest.get("min_sm")
    if lane.get("min_sm") is not None and manifest_min_sm is not None and int(manifest_min_sm) > int(host.get("gpu_sm") or 0):
        raise RuntimeError(f"Wheelhouse requires SM >= {manifest_min_sm}, but host reports SM {host.get('gpu_sm')}")

    return {
        "wheelhouse_dir": str(wheelhouse_dir),
        "wheelhouse_manifest": str(manifest_path),
        "manifest": manifest,
        "package_map": package_map,
    }


def _build_torch_install_cmd(venv_python: Path, lane: dict[str, Any], wheelhouse_info: dict[str, Any] | None = None) -> list[str]:
    cmd = [str(venv_python), "-m", "pip", "install"]
    if lane.get("install_source") == "wheelhouse":
        if wheelhouse_info is None:
            raise RuntimeError("Wheelhouse install requested without verified wheelhouse info.")
        cmd.extend(["--no-index", "--no-deps", "--find-links", str(wheelhouse_info["wheelhouse_dir"])])
        required_wheels = [str(item) for item in lane.get("required_wheels") or []]
        package_specs: list[str] = []
        for name in required_wheels:
            package = wheelhouse_info["package_map"].get(name) or {}
            specifier = str(package.get("specifier") or "").strip()
            package_specs.append(specifier or name)
        cmd.extend(package_specs)
        return cmd

    if lane.get("torch_index_url"):
        cmd.extend(["--index-url", str(lane["torch_index_url"])])
    cmd.extend([str(item) for item in lane.get("torch_packages") or []])
    return cmd

def _load_context() -> dict[str, Any]:
    if len(sys.argv) < 2:
        return {}
    try:
        return json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        emit_json(
            {
                "status": "error",
                "code": "invalid_setup_context",
                "message": "setup.py expects a single JSON argument.",
                "details": {"error": str(exc)},
            }
        )
        raise SystemExit(2) from exc

def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts/python.exe"
    return venv_dir / "bin/python"


def _run(cmd: list[str], label: str, env: dict[str, str] | None = None) -> None:
    emit_json({"status": "running", "step": label, "command": cmd})
    subprocess.run(cmd, check=True, env=env)


def _run_pip_check(venv_python: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    cmd = [str(venv_python), "-m", "pip", "check"]
    emit_json({"status": "running", "step": "pip-check", "command": cmd})
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output = proc.stdout.strip()
    if output:
        emit_json({"status": "log", "step": "pip-check", "message": output})

    allowed_fragments = [
        # Upstream package declares Gradio for demos/UI; the Modly runtime imports the pipeline directly.
        "moss-soundeffect-v2 0.1.0 requires gradio, which is not installed",
        # PyTorch cu128 aarch64 currently reports this through pip check even when torch imports and CUDA probes pass.
        "nvidia-cusparselt-cu12",
        # PyTorch cu130 aarch64 reports the same metadata-level issue; post-install CUDA probes are authoritative.
        "nvidia-cusparselt-cu13",
    ]
    unexpected_lines = [
        line
        for line in output.splitlines()
        if line.strip() and not any(fragment in line for fragment in allowed_fragments)
    ]
    if proc.returncode != 0 and unexpected_lines:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)
    return {
        "returncode": proc.returncode,
        "allowed_issues": [line for line in output.splitlines() if line.strip() and line not in unexpected_lines],
        "unexpected_issues": unexpected_lines,
    }


def _install_runtime(
    venv_python: Path,
    lane: dict[str, Any],
    host: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any] | None:
    wheelhouse_info: dict[str, Any] | None = None
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel"], "bootstrap-pip", env=env)
    if lane.get("install_torch"):
        if lane.get("install_source") == "wheelhouse":
            wheelhouse_info = _verify_wheelhouse(lane, host)
            emit_json(
                {
                    "status": "log",
                    "step": "verify-wheelhouse",
                    "wheelhouse_dir": wheelhouse_info["wheelhouse_dir"],
                    "wheelhouse_manifest": wheelhouse_info["wheelhouse_manifest"],
                    "required_wheels": lane.get("required_wheels") or [],
                }
            )
        torch_cmd = _build_torch_install_cmd(venv_python, lane, wheelhouse_info)
        _run(torch_cmd, f"install-torch-{lane.get('cuda_variant') or 'default'}", env=env)
    _run([str(venv_python), "-m", "pip", "install", *PINNED_PACKAGES], "install-runtime-dependencies", env=env)
    _run(
        [str(venv_python), "-m", "pip", "install", "--ignore-requires-python", "--no-deps", f"git+{UPSTREAM_GIT}"],
        "install-upstream-runtime",
        env=env,
    )
    return wheelhouse_info


def _safe_relative_dir(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"Unsafe venv_dir value: {value}")
    return candidate


def _write_sentinel(ext_dir: Path, payload: dict[str, Any]) -> None:
    sentinel = ext_dir / ".modly/setup-ready.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _bool_from_context(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _download_model_assets(venv_python: Path, model_root: Path, env: dict[str, str]) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    script = (
        "from huggingface_hub import snapshot_download; "
        "import sys; "
        f"snapshot_download(repo_id={HF_REPO!r}, local_dir=sys.argv[1], local_dir_use_symlinks=False)"
    )
    _run([str(venv_python), "-c", script, str(model_root)], "download-model-assets", env=env)


def _resolve_model_root(ext_dir: Path) -> Path:
    models_root = global_models_root(ext_dir)
    if models_root is not None:
        return models_root / MODEL_OWNER_REL
    return ext_dir / "models/openmoss-team/moss-soundeffect-v2_0"


def _parse_torch_arch_list(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        digits = "".join(character for character in str(value) if character.isdigit())
        if len(digits) >= 2:
            try:
                result.append(int(digits))
            except ValueError:
                continue
    return sorted(set(result))


def _run_json_probe(venv_python: Path, label: str, script: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    cmd = [str(venv_python), "-c", script]
    emit_json({"status": "running", "step": label, "command": cmd})
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Probe '{label}' failed with return code {completed.returncode}: {completed.stderr.strip() or stdout}")
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"Probe '{label}' did not emit valid JSON.")


def _post_install_probe(venv_python: Path, lane: dict[str, Any], host: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    script = (
        "import json\n"
        "payload = {'ok': False}\n"
        "try:\n"
        " import torch, torchaudio, torchvision, diffusers, transformers, soundfile\n"
        " from moss_soundeffect_v2 import MossSoundEffectPipeline\n"
        " payload.update({\n"
        "  'ok': True,\n"
        "  'torch': getattr(torch, '__version__', None),\n"
        "  'torchaudio': getattr(torchaudio, '__version__', None),\n"
        "  'torchvision': getattr(torchvision, '__version__', None),\n"
        "  'torch_cuda': getattr(getattr(torch, 'version', None), 'cuda', None),\n"
        "  'cuda_available': bool(torch.cuda.is_available()),\n"
        "  'pipeline': MossSoundEffectPipeline.__name__,\n"
        " })\n"
        " try:\n"
        "  payload['torch_arch_list'] = list(torch.cuda.get_arch_list())\n"
        " except Exception as exc:\n"
        "  payload['torch_arch_list_error'] = f'{type(exc).__name__}: {exc}'\n"
        " if payload['cuda_available']:\n"
        "  payload['device_name'] = torch.cuda.get_device_name(0)\n"
        "  payload['capability'] = list(torch.cuda.get_device_capability(0))\n"
        "except Exception as exc:\n"
        " payload['error_type'] = type(exc).__name__\n"
        " payload['error'] = str(exc)\n"
        "print(json.dumps(payload, sort_keys=True))"
    )
    payload = _run_json_probe(venv_python, "post-install-probe", script, env=env)
    if not payload.get("ok"):
        raise RuntimeError(f"Post-install probe failed: {payload.get('error_type') or 'RuntimeError'}: {payload.get('error') or 'unknown error'}")

    if lane.get("accelerator") == "nvidia":
        if not payload.get("cuda_available"):
            raise RuntimeError("Installed torch lane is NVIDIA/CUDA, but torch.cuda.is_available() is false.")
        capability = payload.get("capability") or []
        gpu_sm = None
        if isinstance(capability, list) and len(capability) >= 2:
            gpu_sm = int(capability[0]) * 10 + int(capability[1])
        else:
            gpu_sm = host.get("gpu_sm")
        if lane.get("min_sm") is not None and gpu_sm is not None and int(gpu_sm) < int(lane["min_sm"]):
            raise RuntimeError(f"Resolved lane requires SM >= {lane['min_sm']}, but post-install probe observed SM {gpu_sm}.")
        if lane.get("max_sm") is not None and gpu_sm is not None and int(gpu_sm) > int(lane["max_sm"]):
            raise RuntimeError(f"Resolved lane supports up to SM {lane['max_sm']}, but post-install probe observed SM {gpu_sm}.")
        arch_list = _parse_torch_arch_list(list(payload.get("torch_arch_list") or []))
        if arch_list and gpu_sm is not None and max(arch_list) < int(gpu_sm):
            raise RuntimeError(
                f"Torch arch list {payload.get('torch_arch_list')} does not cover observed GPU SM {gpu_sm}; runtime compile/probe failures are expected."
            )
        torch_cuda = str(payload.get("torch_cuda") or "")
        if gpu_sm is not None and int(gpu_sm) >= 121 and torch_cuda.startswith("12.8"):
            raise RuntimeError("SM 12.1+ detected with an incompatible CUDA 12.8 torch runtime; this lane is blocked for this host class.")
    return payload


def _wheelhouse_payload(lane: dict[str, Any], wheelhouse_info: dict[str, Any] | None, previous: dict[str, Any] | None) -> dict[str, Any]:
    previous_wheelhouse = previous.get("wheelhouse") if isinstance(previous, dict) and isinstance(previous.get("wheelhouse"), dict) else {}
    return {
        "install_source": lane.get("install_source"),
        "wheelhouse_dir": wheelhouse_info.get("wheelhouse_dir") if wheelhouse_info else previous_wheelhouse.get("wheelhouse_dir", lane.get("wheelhouse_dir")),
        "wheelhouse_manifest": wheelhouse_info.get("wheelhouse_manifest") if wheelhouse_info else previous_wheelhouse.get("wheelhouse_manifest", lane.get("wheelhouse_manifest")),
        "manifest": wheelhouse_info.get("manifest") if wheelhouse_info else previous_wheelhouse.get("manifest"),
    }


def main() -> int:
    context = _load_context()
    ext_dir = Path(context.get("ext_dir") or extension_root())
    python_exe = str(context.get("python_exe") or sys.executable)
    venv_rel = _safe_relative_dir(str(context.get("venv_dir") or "venv"))
    venv_dir = ext_dir / venv_rel
    force_reinstall = _bool_from_context(context.get("force_reinstall"), False)
    force_model_download = _bool_from_context(context.get("force_model_download"), False)
    skip_probe = _bool_from_context(context.get("skip_probe"), False)
    download_model_assets = force_model_download or (
        _bool_from_context(context.get("download_model_assets"), True)
        and not _bool_from_context(context.get("skip_model_download"), False)
    )
    pip_env = os.environ.copy()
    pip_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    pip_env.setdefault("PYTHONUTF8", "1")
    host: dict[str, Any] | None = None
    lane: dict[str, Any] | None = None
    wheelhouse_info: dict[str, Any] | None = None
    pip_check_result: dict[str, Any] | None = None
    post_install_probe: dict[str, Any] | None = None
    runtime_requirements_hash = ""
    model_assets_signature = _model_assets_signature()
    previous_sentinel = _read_previous_sentinel(ext_dir)
    install_plan: dict[str, Any] = {
        "force_reinstall": force_reinstall,
        "force_model_download": force_model_download,
        "skip_probe": skip_probe,
    }
    skipped_steps: list[str] = []
    runtime_install_skipped = False
    downloads_started = False

    try:
        compatibility = resolve_host_compat({**context, "python_exe": python_exe, "ext_dir": str(ext_dir)})
        host = compatibility["host"]
        lane = compatibility["lane"]
        runtime_requirements_hash = _runtime_requirements_hash(lane)

        emit_json(
            {
                "status": "starting",
                "extension_id": EXTENSION_ID,
                "python_exe": python_exe,
                "venv_dir": str(venv_dir),
                "model_root": str(_resolve_model_root(ext_dir)),
                "downloads_started": False,
                "download_model_assets": download_model_assets,
                "installs_started": False,
                "host": host,
                "lane": lane,
                "setup_schema_version": SETUP_SCHEMA_VERSION,
                "runtime_requirements_hash": runtime_requirements_hash,
                "model_assets_signature": model_assets_signature,
            }
        )

        if lane["status"] == "unsupported":
            emit_json(
                {
                    "status": "error",
                    "code": "unsupported_host_matrix",
                    "message": lane["reason"],
                    "details": {"host": host, "lane": lane},
                    "downloads_started": False,
                    "installs_started": False,
                    "next_steps": lane.get("guidance") or [],
                }
            )
            return 1

        if not venv_dir.exists():
            emit_json({"status": "running", "step": "create-venv", "message": "Creating extension-owned virtual environment."})
            builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt")
            builder.create(venv_dir)

        venv_python = _venv_python(venv_dir)
        if not venv_python.exists():
            raise RuntimeError(f"Virtual environment python not found at {venv_python}")

        runtime_reuse = _runtime_reuse_decision(
            previous_sentinel,
            lane,
            runtime_requirements_hash,
            venv_python,
            force_reinstall=force_reinstall,
        )
        runtime_install_skipped = bool(runtime_reuse["skip_runtime_install"])
        install_plan.update(
            {
                "runtime_reuse": runtime_reuse,
                "runtime_requirements_hash": runtime_requirements_hash,
                "selected_lane": str(lane.get("id") or ""),
            }
        )

        if runtime_install_skipped:
            skipped_steps.extend(["bootstrap-pip", "install-torch", "install-runtime-dependencies", "install-upstream-runtime"])
            emit_json(
                {
                    "status": "runtime-install-skipped",
                    "reason": runtime_reuse["reason"],
                    "previous_status": runtime_reuse["previous_status"],
                    "venv_dir": str(venv_dir),
                    "selected_lane": str(lane.get("id") or ""),
                }
            )
        else:
            wheelhouse_info = _install_runtime(venv_python, lane, host, pip_env)

        pip_check_result = _run_pip_check(venv_python, env=pip_env)
        if skip_probe:
            skipped_steps.append("post-install-probe")
            post_install_probe = {"ok": None, "skipped": True, "reason": "skip_probe"}
            emit_json({"status": "probe-skipped", "step": "post-install-probe", "reason": "skip_probe"})
        else:
            try:
                post_install_probe = _post_install_probe(venv_python, lane, host, pip_env)
            except Exception:
                if runtime_install_skipped:
                    emit_json(
                        {
                            "status": "log",
                            "step": "runtime-repair",
                            "message": "Skipped runtime failed the health probe; reinstalling dependencies to repair the environment.",
                            "selected_lane": str(lane.get("id") or ""),
                        }
                    )
                    runtime_install_skipped = False
                    skipped_steps = [step for step in skipped_steps if step not in {"bootstrap-pip", "install-torch", "install-runtime-dependencies", "install-upstream-runtime"}]
                    wheelhouse_info = _install_runtime(venv_python, lane, host, pip_env)
                    pip_check_result = _run_pip_check(venv_python, env=pip_env)
                    post_install_probe = _post_install_probe(venv_python, lane, host, pip_env)
                    install_plan["runtime_reuse"]["repaired_after_probe_failure"] = True
                else:
                    raise

        current_model_root = _resolve_model_root(ext_dir)
        legacy_root = legacy_model_root()
        if not validate_model_files(legacy_root) and validate_model_files(current_model_root):
            emit_json(
                {
                    "status": "log",
                    "step": "model-root",
                    "message": "Legacy extension-local model files detected; copy or move them to the global Modly models directory for UI ownership alignment.",
                    "legacy_model_root": str(legacy_root),
                    "model_root": str(current_model_root),
                }
            )
        missing_before_download = validate_model_files(current_model_root)
        if force_model_download:
            downloads_started = True
            _download_model_assets(venv_python, current_model_root, pip_env)
        elif not missing_before_download:
            skipped_steps.append("download-model-assets")
            emit_json(
                {
                    "status": "model-assets-skipped",
                    "model_root": str(current_model_root),
                    "sentinel_count": len(REQUIRED_SENTINELS),
                    "reason": "required_sentinels_present",
                }
            )
        elif download_model_assets:
            if missing_before_download:
                downloads_started = True
                _download_model_assets(venv_python, current_model_root, pip_env)
        else:
            skipped_steps.append("download-model-assets")

        missing = validate_model_files(current_model_root)
        if download_model_assets and missing:
            raise RuntimeError(f"Model asset download completed but required sentinels are still missing: {missing}")
        sentinel_payload = {
            "status": "ready" if not missing else "runtime_ready_model_missing",
            "setup_schema_version": SETUP_SCHEMA_VERSION,
            "runtime_requirements_hash": runtime_requirements_hash,
            "model_assets_signature": model_assets_signature,
            "extension_id": EXTENSION_ID,
            "python_exe": str(venv_python),
            "venv_dir": str(venv_dir),
            "model_root": str(current_model_root),
            "hf_repo": HF_REPO,
            "downloads_started": downloads_started,
            "download_model_assets": download_model_assets,
            "installs_started": not runtime_install_skipped,
            "runtime_install_skipped": runtime_install_skipped,
            "host": host,
            "lane": lane,
            "selected_lane": str(lane.get("id") or ""),
            "pip_check": pip_check_result,
            "post_install_probe": post_install_probe,
            "wheelhouse": _wheelhouse_payload(lane, wheelhouse_info, previous_sentinel),
            "install_plan": install_plan,
            "skipped_steps": skipped_steps,
            "missing_model_files": missing,
            "setup_contract": "python-root-setup-py-json-observable",
            "next_steps": [
                "If model files are missing, rerun setup with download_model_assets=true or place the Hugging Face snapshot in the logical model directory.",
                "Run a first workflow smoke test only on a host matrix marked supported or explicitly accepted as experimental.",
            ],
        }
        _write_sentinel(ext_dir, sentinel_payload)
        emit_json(sentinel_payload)
        return 0
    except subprocess.CalledProcessError as exc:
        payload = {
            "status": "error",
            "code": "setup_command_failed",
            "message": f"Setup command failed during step '{exc.cmd}'.",
            "details": {"returncode": exc.returncode, "command": exc.cmd},
            "setup_schema_version": SETUP_SCHEMA_VERSION,
            "runtime_requirements_hash": runtime_requirements_hash,
            "model_assets_signature": model_assets_signature,
            "host": host,
            "lane": lane,
            "downloads_started": False,
            "installs_started": not runtime_install_skipped,
            "install_plan": install_plan,
            "skipped_steps": skipped_steps,
            "next_steps": [
                "Inspect Electron setup logs for the failed pip command.",
                "Verify Python 3.12 and NVIDIA CUDA compatibility before retrying.",
            ],
        }
        _write_sentinel(ext_dir, payload)
        emit_json(payload)
        return exc.returncode or 1
    except Exception as exc:
        payload = {
            "status": "error",
            "code": "setup_failed",
            "message": str(exc),
            "setup_schema_version": SETUP_SCHEMA_VERSION,
            "runtime_requirements_hash": runtime_requirements_hash,
            "model_assets_signature": model_assets_signature,
            "host": host,
            "lane": lane,
            "downloads_started": False,
            "installs_started": not runtime_install_skipped,
            "install_plan": install_plan,
            "skipped_steps": skipped_steps,
            "next_steps": [
                "Verify the injected python_exe is Python 3.12+.",
                "Retry setup inside Modly on a host matrix that resolves to a supported or explicitly accepted experimental CUDA lane.",
            ],
        }
        _write_sentinel(ext_dir, payload)
        emit_json(payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
