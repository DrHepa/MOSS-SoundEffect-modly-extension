from __future__ import annotations

import json
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from .common import local_wheelhouse_dir, local_wheelhouse_manifest_path


def _bool_from_any(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_os(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("linux"):
        return "linux"
    if raw.startswith("win"):
        return "windows"
    if raw.startswith("darwin") or raw.startswith("mac"):
        return "darwin"
    return raw or "unknown"


def normalize_arch(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"x86_64", "amd64"}:
        return "x86_64"
    if raw in {"aarch64", "arm64"}:
        return "aarch64"
    return raw or "unknown"


def normalize_gpu_sm(value: Any) -> int | None:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    if len(digits) < 2:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def normalize_cuda_version(value: Any) -> str | None:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    if not digits:
        return None
    if len(digits) == 2:
        return f"{digits}0"
    return digits[:3]


def _driver_major(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    major = text.split(".", 1)[0]
    try:
        return int(major)
    except ValueError:
        return None


def _parse_python_identity(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    version = [int(item) for item in payload["version"]]
    python_tag = f"cp{version[0]}{version[1]}"
    return {
        "version": version,
        "version_text": ".".join(str(item) for item in version),
        "python_tag": python_tag,
        "implementation": payload.get("implementation") or "cpython",
        "platform": payload.get("platform"),
        "soabi": payload.get("soabi"),
    }


def probe_python_identity(python_exe: str) -> dict[str, Any]:
    script = (
        "import json, platform, sys, sysconfig; "
        "print(json.dumps({"
        "'version': list(sys.version_info[:3]), "
        "'implementation': getattr(sys.implementation, 'name', 'cpython'), "
        "'platform': sysconfig.get_platform(), "
        "'soabi': sysconfig.get_config_var('SOABI')"
        "}))"
    )
    raw = subprocess.check_output([python_exe, "-c", script], text=True).strip()
    return _parse_python_identity(raw)


def probe_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"available": False, "reason": "nvidia-smi-not-found", "command": command}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "command": command}

    result: dict[str, Any] = {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        result["reason"] = "nvidia-smi-failed"
        return result

    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) >= 3:
        result["device_name"] = parts[0]
        result["gpu_sm"] = normalize_gpu_sm(parts[1])
        result["driver_version"] = parts[2]
    return result


def detect_host_facts(context: dict[str, Any]) -> dict[str, Any]:
    python_exe = str(context.get("python_exe") or sys.executable)
    ext_dir = str(Path(context.get("ext_dir") or Path(__file__).resolve().parent.parent))
    os_name = normalize_os(platform.system())
    arch = normalize_arch(platform.machine() or sysconfig.get_platform())
    python_identity = probe_python_identity(python_exe)
    smi = probe_nvidia_smi()
    gpu_sm = normalize_gpu_sm(context.get("gpu_sm"))
    if gpu_sm is None:
        gpu_sm = smi.get("gpu_sm")
    cuda_version = normalize_cuda_version(context.get("cuda_version"))
    return {
        "os": os_name,
        "arch": arch,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "sysconfig_platform": sysconfig.get_platform(),
        "python_exe": python_exe,
        "python": python_identity,
        "ext_dir": ext_dir,
        "gpu_sm": gpu_sm,
        "cuda_version": cuda_version,
        "driver_version": smi.get("driver_version"),
        "nvidia_smi": smi,
        "allow_cpu_setup": _bool_from_any(context.get("allow_cpu_setup"), False),
    }


def _lane(
    *,
    lane_id: str,
    status: str,
    accelerator: str,
    reason: str,
    guidance: list[str],
    torch_packages: list[str] | None = None,
    torch_index_url: str | None = None,
    cuda_variant: str | None = None,
    min_sm: int | None = None,
    max_sm: int | None = None,
    install_torch: bool = True,
    install_source: str = "index",
    wheelhouse_dir: str | None = None,
    wheelhouse_manifest: str | None = None,
    required_wheels: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": lane_id,
        "status": status,
        "accelerator": accelerator,
        "reason": reason,
        "guidance": guidance,
        "torch_packages": list(torch_packages or []),
        "torch_index_url": torch_index_url,
        "cuda_variant": cuda_variant,
        "min_sm": min_sm,
        "max_sm": max_sm,
        "install_torch": install_torch,
        "install_source": install_source,
        "wheelhouse_dir": wheelhouse_dir,
        "wheelhouse_manifest": wheelhouse_manifest,
        "required_wheels": list(required_wheels or []),
    }


def _read_manifest_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_package_names(manifest: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in manifest.get("packages") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _local_wheelhouse_state(ext_dir: str) -> dict[str, Any]:
    wheelhouse_dir = local_wheelhouse_dir(Path(ext_dir))
    manifest_path = local_wheelhouse_manifest_path(Path(ext_dir))
    state: dict[str, Any] = {
        "wheelhouse_dir": str(wheelhouse_dir),
        "wheelhouse_manifest": str(manifest_path),
        "exists": wheelhouse_dir.is_dir(),
        "manifest_exists": manifest_path.is_file(),
        "valid": False,
        "manifest": None,
        "missing_required_wheels": [],
    }
    if not manifest_path.is_file():
        return state
    manifest = _read_manifest_json(manifest_path)
    if manifest is None:
        return state
    state["manifest"] = manifest
    required_wheels = ["torch", "torchaudio", "torchvision", "triton"]
    package_names = _manifest_package_names(manifest)
    missing_required_wheels = [name for name in required_wheels if name not in package_names]
    state["missing_required_wheels"] = missing_required_wheels
    state["valid"] = not missing_required_wheels
    return state


def resolve_host_compat(context: dict[str, Any]) -> dict[str, Any]:
    facts = detect_host_facts(context)
    python_version = tuple(facts["python"]["version"])
    python_tag = str(facts["python"].get("python_tag") or "")
    guidance = ["MOSS-SoundEffect v2 requires a CUDA-capable PyTorch runtime for practical inference. Python support is lane-specific."]
    requested_lane = str(context.get("torch_lane") or "").strip().lower()
    require_local_wheelhouse = _bool_from_any(context.get("require_local_wheelhouse"), False) or requested_lane == "local-wheelhouse-sm121"

    if python_version < (3, 12, 0) and not (facts["os"] == "windows" and python_version >= (3, 11, 0)):
        return {
            "host": facts,
            "lane": _lane(
                lane_id=f"{facts['os']}-{facts['arch']}/unsupported-python",
                status="unsupported",
                accelerator="unknown",
                reason=f"Python {facts['python']['version_text']} is below the minimum supported setup baseline for this host.",
                guidance=["Use a Python 3.12+ interpreter, or add an explicitly verified lane for this host matrix."],
                install_torch=False,
            ),
        }

    if facts["os"] == "darwin":
        return {
            "host": facts,
            "lane": _lane(
                lane_id=f"{facts['os']}-{facts['arch']}/mps-unsupported",
                status="unsupported",
                accelerator="mps",
                reason="macOS/MPS is not evidenced for upstream MOSS-SoundEffect inference.",
                guidance=["Use Linux with NVIDIA CUDA.", "Do not attempt a silent CPU or MPS fallback for this extension."],
                install_torch=False,
            ),
        }

    if facts["gpu_sm"] is None:
        cpu_guidance = [
            "No NVIDIA GPU compute capability was supplied or detected.",
            "This extension intentionally fails fast instead of attempting impractical CPU inference.",
        ]
        if facts["allow_cpu_setup"]:
            cpu_guidance.append("allow_cpu_setup=true was provided, but no CPU dry-run lane is managed by this setup contract yet.")
        return {
            "host": facts,
            "lane": _lane(
                lane_id=f"{facts['os']}-{facts['arch']}/cpu-unsupported",
                status="unsupported",
                accelerator="cpu",
                reason="No supported NVIDIA CUDA device was detected for MOSS-SoundEffect runtime setup.",
                guidance=cpu_guidance,
                install_torch=False,
            ),
        }

    gpu_sm = int(facts["gpu_sm"])
    driver_major = _driver_major(facts["driver_version"])
    cuda_version = facts["cuda_version"] or "unknown"

    if facts["os"] == "linux" and facts["arch"] in {"x86_64", "aarch64"}:
        if gpu_sm >= 121:
            wheelhouse_state = _local_wheelhouse_state(facts["ext_dir"])
            if facts["arch"] == "aarch64" and python_tag == "cp312":
                if wheelhouse_state["exists"] and wheelhouse_state["manifest_exists"] and wheelhouse_state["valid"]:
                    return {
                        "host": facts,
                        "lane": _lane(
                            lane_id="linux-aarch64/local-wheelhouse-sm121",
                            status="supported",
                            accelerator="nvidia",
                            reason="SM 12.1+ on Linux aarch64/cp312 can use the local cu130 wheelhouse built for this host class.",
                            guidance=[
                                "This lane uses a local wheelhouse instead of public PyTorch indexes.",
                                "Source builds remain manual; setup only consumes a prevalidated WHEELHOUSE.json manifest.",
                            ],
                            cuda_variant="cu130",
                            min_sm=121,
                            install_source="wheelhouse",
                            wheelhouse_dir=str(wheelhouse_state["wheelhouse_dir"]),
                            wheelhouse_manifest=str(wheelhouse_state["wheelhouse_manifest"]),
                            required_wheels=["torch", "torchaudio", "torchvision", "triton"],
                        ),
                    }
                if require_local_wheelhouse:
                    missing_reason = "Local SM121 wheelhouse is required but not ready."
                    if not wheelhouse_state["exists"]:
                        missing_reason = f"Local SM121 wheelhouse directory is missing: {wheelhouse_state['wheelhouse_dir']}"
                    elif not wheelhouse_state["manifest_exists"]:
                        missing_reason = f"Local SM121 wheelhouse manifest is missing: {wheelhouse_state['wheelhouse_manifest']}"
                    elif wheelhouse_state["missing_required_wheels"]:
                        missing_reason = (
                            "Local SM121 wheelhouse manifest is incomplete; missing required packages: "
                            + ", ".join(wheelhouse_state["missing_required_wheels"])
                        )
                    return {
                        "host": facts,
                        "lane": _lane(
                            lane_id="linux-aarch64/local-wheelhouse-sm121-missing",
                            status="unsupported",
                            accelerator="nvidia",
                            reason=missing_reason,
                            guidance=[
                                f"Prepare {wheelhouse_state['wheelhouse_dir']} with WHEELHOUSE.json and the torch/triton wheels before rerunning setup.",
                                "Do not force the public nightly lane when the setup context explicitly requires the local wheelhouse.",
                            ],
                            cuda_variant="cu130",
                            min_sm=121,
                            install_torch=False,
                            install_source="wheelhouse",
                            wheelhouse_dir=str(wheelhouse_state["wheelhouse_dir"]),
                            wheelhouse_manifest=str(wheelhouse_state["wheelhouse_manifest"]),
                            required_wheels=["torch", "torchaudio", "torchvision", "triton"],
                        ),
                    }
                    missing_reason = "SM 12.1+ requires a validated local source-built wheelhouse or another explicitly verified CUDA 13 lane."
            lane_guidance = [
                f"Prepare {wheelhouse_state['wheelhouse_dir']} with WHEELHOUSE.json and wheels built with TORCH_CUDA_ARCH_LIST=\"12.1a\".",
                "Setup will not compile PyTorch/Triton automatically and will not guess a public wheel lane for SM 12.1+.",
            ]
            if driver_major is not None and driver_major < 580:
                lane_guidance.append(f"Detected NVIDIA driver {facts['driver_version']}; upgrade to a CUDA 13-capable driver for reliable runtime behavior.")
            return {
                "host": facts,
                "lane": _lane(
                    lane_id=f"linux-{facts['arch']}/sm121-wheelhouse-required",
                    status="unsupported",
                    accelerator="nvidia",
                    reason=missing_reason,
                    guidance=lane_guidance,
                    cuda_variant="cu130",
                    min_sm=121,
                    install_torch=False,
                    install_source="wheelhouse",
                    wheelhouse_dir=str(wheelhouse_state["wheelhouse_dir"]),
                    wheelhouse_manifest=str(wheelhouse_state["wheelhouse_manifest"]),
                    required_wheels=["torch", "torchaudio", "torchvision", "triton"],
                ),
            }

        if gpu_sm == 120:
            if driver_major is not None and driver_major >= 580:
                return {
                    "host": facts,
                    "lane": _lane(
                        lane_id=f"linux-{facts['arch']}/cu130",
                        status="supported",
                        accelerator="nvidia",
                        reason="SM 12.0 prefers the modern cu130 lane when the installed driver can support it.",
                        guidance=["Selected cu130 because the host appears compatible with the newer CUDA lane."],
                        torch_packages=["torch==2.12.1+cu130", "torchaudio==2.11.0+cu130", "torchvision==0.27.1+cu130"],
                        torch_index_url="https://download.pytorch.org/whl/cu130",
                        cuda_variant="cu130",
                        min_sm=120,
                    ),
                }
            if driver_major is not None and driver_major < 580:
                return {
                    "host": facts,
                    "lane": _lane(
                        lane_id=f"linux-{facts['arch']}/cu128",
                        status="experimental",
                        accelerator="nvidia",
                        reason="SM 12.0 can fall back to cu128 on older drivers, but this is less future-proof than cu130.",
                        guidance=[
                            f"Detected NVIDIA driver {facts['driver_version']}; using cu128 as an experimental fallback.",
                            "Upgrade to a CUDA 13-capable driver to move this host onto the preferred cu130 lane.",
                        ],
                        torch_packages=["torch==2.9.0+cu128", "torchaudio==2.9.0", "torchvision==0.24.0"],
                        torch_index_url="https://download.pytorch.org/whl/cu128",
                        cuda_variant="cu128",
                        min_sm=80,
                        max_sm=120,
                    ),
                }
            return {
                "host": facts,
                "lane": _lane(
                    lane_id=f"linux-{facts['arch']}/cu130",
                    status="experimental",
                    accelerator="nvidia",
                    reason="SM 12.0 prefers cu130, but no driver evidence was available to confirm the host can satisfy that lane.",
                    guidance=[
                        "Verify an NVIDIA driver compatible with CUDA 13 before running setup.",
                        "If the host is pinned below that driver baseline, a manual cu128 fallback may be needed instead.",
                    ],
                    torch_packages=["torch==2.12.1+cu130", "torchaudio==2.11.0+cu130", "torchvision==0.27.1+cu130"],
                    torch_index_url="https://download.pytorch.org/whl/cu130",
                    cuda_variant="cu130",
                    min_sm=120,
                ),
            }

        if gpu_sm < 120:
            return {
                "host": facts,
                "lane": _lane(
                    lane_id=f"linux-{facts['arch']}/cu128",
                    status="supported",
                    accelerator="nvidia",
                    reason="SM below 12.0 stays on the established cu128 lane to avoid breaking existing Linux CUDA users without verified cu130 need.",
                    guidance=[
                        "cu128 remains the conservative default for pre-SM12.0 Linux users in this extension.",
                        "This lane is intentionally blocked for SM 12.1+ hosts.",
                    ],
                    torch_packages=["torch==2.9.0+cu128", "torchaudio==2.9.0", "torchvision==0.24.0"],
                    torch_index_url="https://download.pytorch.org/whl/cu128",
                    cuda_variant="cu128",
                    min_sm=0,
                    max_sm=120,
                ),
            }

    if facts["os"] == "windows":
        if python_tag == "cp311" and 70 <= gpu_sm < 120:
            return {
                "host": facts,
                "lane": _lane(
                    lane_id="windows-x86_64/cp311-cu126-legacy-nvidia",
                    status="experimental",
                    accelerator="nvidia",
                    reason="Windows cp311 with pre-SM12 NVIDIA GPUs can use the cu126 PyTorch lane for setup/import probes, but pre-SM80 GPUs are not considered viable for practical MOSS generation in current tests.",
                    guidance=[
                        "This lane bypasses upstream MOSS Requires-Python >=3.12 metadata and is kept for setup/import diagnostics, not as a practical inference target.",
                        "Do not present pre-SM80 Windows GPUs as supported for useful generation until a practical end-to-end runtime benchmark is validated.",
                        "Audio saving uses soundfile instead of torchaudio.save/TorchCodec to avoid requiring FFmpeg full-shared DLLs on Windows.",
                    ],
                    torch_packages=["torch==2.9.1+cu126", "torchaudio==2.9.1+cu126", "torchvision==0.24.1+cu126"],
                    torch_index_url="https://download.pytorch.org/whl/cu126",
                    cuda_variant="cu126",
                    min_sm=70,
                    max_sm=120,
                ),
            }
        if gpu_sm >= 121:
            return {
                "host": facts,
                "lane": _lane(
                    lane_id="windows-x86_64/cu130",
                    status="experimental",
                    accelerator="nvidia",
                    reason="Windows SM 12.1+ is modeled as a tentative cu130 lane, but this extension has not verified that matrix end-to-end.",
                    guidance=[
                        "Treat Windows as experimental until a full setup and generation smoke test is captured.",
                        "Do not use the older cu128 lane on SM 12.1+.",
                    ],
                    torch_packages=["torch==2.12.1+cu130", "torchaudio==2.11.0+cu130", "torchvision==0.27.1+cu130"],
                    torch_index_url="https://download.pytorch.org/whl/cu130",
                    cuda_variant="cu130",
                    min_sm=121,
                ),
            }
        return {
            "host": facts,
            "lane": _lane(
                lane_id=f"windows-{facts['arch']}/unverified",
                status="unsupported",
                accelerator="nvidia",
                reason="Windows lanes below SM 12.1 are not evidence-backed for this extension.",
                guidance=["Use Linux + NVIDIA for the supported path.", "Add a verified Windows dependency matrix before enabling setup installs."],
                install_torch=False,
            ),
        }

    return {
        "host": facts,
        "lane": _lane(
            lane_id=f"{facts['os']}-{facts['arch']}/unsupported",
            status="unsupported",
            accelerator="unknown",
            reason="No evidence-backed MOSS-SoundEffect runtime lane exists for this host matrix.",
            guidance=guidance,
            install_torch=False,
        ),
    }
