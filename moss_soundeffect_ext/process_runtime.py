from __future__ import annotations

import os
import random
import traceback
import json
import contextlib
import io
import warnings
from pathlib import Path
from typing import Any

from .common import (
    HF_REPO,
    WORKFLOW_OUTPUT_REL,
    ExtensionError,
    bool_from_any,
    emit_json,
    ensure_relative_child,
    model_root,
    read_request_stdin,
    sanitize_filename,
    setup_sentinel_path,
    validate_model_files,
)


def _log(message: str, **extra: Any) -> None:
    payload: dict[str, Any] = {"type": "log", "message": message}
    payload.update(extra)
    emit_json(payload)


def _progress(percent: int, label: str, **extra: Any) -> None:
    payload: dict[str, Any] = {"type": "progress", "percent": percent, "label": label}
    payload.update(extra)
    emit_json(payload)


def _error(message: str, code: str, details: dict[str, Any] | None = None) -> None:
    emit_json({"type": "error", "message": message, "code": code, "details": details or {}})


def _extract_prompt(input_payload: Any, params: dict[str, Any]) -> str:
    if isinstance(params.get("prompt"), str) and params.get("prompt").strip():
        return params["prompt"].strip()
    if isinstance(input_payload, str) and input_payload.strip():
        return input_payload.strip()
    if isinstance(input_payload, dict):
        for key in ("prompt", "text", "value"):
            value = input_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ExtensionError(
        "A non-empty text prompt is required either in params.prompt or the node input.",
        code="missing_prompt",
    )


def _int_param(params: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    raw = params.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ExtensionError(f"Parameter '{key}' must be an integer.", code="invalid_param", details={"param": key}) from exc
    if value < minimum or value > maximum:
        raise ExtensionError(
            f"Parameter '{key}' must be between {minimum} and {maximum}.",
            code="invalid_param_range",
            details={"param": key, "value": value, "min": minimum, "max": maximum},
        )
    return value


def _float_param(params: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    raw = params.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ExtensionError(f"Parameter '{key}' must be a number.", code="invalid_param", details={"param": key}) from exc
    if value < minimum or value > maximum:
        raise ExtensionError(
            f"Parameter '{key}' must be between {minimum} and {maximum}.",
            code="invalid_param_range",
            details={"param": key, "value": value, "min": minimum, "max": maximum},
        )
    return value


def _resolve_seed(params: dict[str, Any]) -> int:
    seed = _int_param(params, "seed", 0, -1, 2147483647)
    if seed == -1:
        return random.SystemRandom().randint(0, 2147483647)
    return seed


def _resolve_output_path(workspace_dir: Path, prompt: str, params: dict[str, Any]) -> tuple[Path, str]:
    output_name = params.get("output_name")
    if isinstance(output_name, str) and output_name.strip():
        stem = sanitize_filename(output_name, fallback="moss-soundeffect")
    else:
        stem = sanitize_filename(prompt[:60], fallback="moss-soundeffect")
    relative_output = WORKFLOW_OUTPUT_REL / f"{stem}.wav"
    output_path = ensure_relative_child(workspace_dir, relative_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = output_path.relative_to(workspace_dir).as_posix()
    return output_path, relative_path


def _readiness_guard(current_model_root: Path) -> None:
    sentinel = setup_sentinel_path()
    if not sentinel.is_file():
        raise ExtensionError(
            "Setup readiness sentinel is missing. Run Modly extension setup first.",
            code="setup_not_ready",
            details={"sentinel": str(sentinel)},
        )
    try:
        sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtensionError(
            "Setup readiness sentinel is invalid JSON. Rerun Modly extension setup.",
            code="setup_not_ready",
            details={"sentinel": str(sentinel), "error": str(exc)},
        ) from exc
    if not isinstance(sentinel_payload, dict):
        raise ExtensionError(
            "Setup readiness sentinel is malformed. Rerun Modly extension setup.",
            code="setup_not_ready",
            details={"sentinel": str(sentinel)},
        )
    sentinel_status = str(sentinel_payload.get("status") or "")
    allowed_statuses = {"ready", "runtime_ready_model_missing"}
    if sentinel_status == "error" or sentinel_status not in allowed_statuses:
        raise ExtensionError(
            "Setup readiness sentinel is not in a runnable state. Rerun or fix extension setup first.",
            code="setup_not_ready",
            details={
                "sentinel": str(sentinel),
                "status": sentinel_status,
                "next_steps": sentinel_payload.get("next_steps") or [],
            },
        )
    missing = validate_model_files(current_model_root)
    if missing:
        raise ExtensionError(
            "Required model files are missing from the logical model directory.",
            code="missing_model_files",
            details={"model_root": str(current_model_root), "missing": missing},
        )


def _runtime_env(disable_compile: bool) -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if disable_compile:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["TORCHINDUCTOR_DISABLE"] = "1"


@contextlib.contextmanager
def _suppress_library_stdio():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _configure_warning_filters() -> None:
    warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
    warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
    warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")


def _save_audio_with_soundfile(audio: Any, output_path: Path, sample_rate: int) -> Path:
    import numpy as np
    import soundfile as sf
    import torch

    wav = audio.detach().cpu() if isinstance(audio, torch.Tensor) else audio
    if isinstance(wav, torch.Tensor):
        if wav.ndim == 3:
            wav = wav[0]
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32).numpy()
    else:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 3:
            wav = wav[0]
        elif wav.ndim == 1:
            wav = wav[None, :]

    # Upstream uses torchaudio.save([channels, samples]), but modern torchaudio
    # routes WAV writing through TorchCodec/FFmpeg on some platforms. Writing the
    # tensor directly with soundfile avoids a fragile FFmpeg DLL requirement on
    # native Windows while preserving the same WAV artifact contract.
    if wav.ndim == 2:
        wav = wav.T
    wav = np.clip(wav, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav, int(sample_rate), subtype="PCM_16", format="WAV")
    return output_path


def main() -> int:
    try:
        request = read_request_stdin()
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise ExtensionError("params must be a JSON object.", code="invalid_params")

        workspace_dir_raw = request.get("workspaceDir")
        if not isinstance(workspace_dir_raw, str) or not workspace_dir_raw.strip():
            raise ExtensionError("workspaceDir is required.", code="missing_workspace")
        workspace_dir = Path(workspace_dir_raw).expanduser()
        if not workspace_dir.is_absolute():
            raise ExtensionError("workspaceDir must be an absolute path provided by Modly.", code="invalid_workspace")

        prompt = _extract_prompt(request.get("input"), params)
        seconds = _int_param(params, "seconds", 10, 1, 30)
        num_inference_steps = _int_param(params, "num_inference_steps", 100, 1, 300)
        cfg_scale = _float_param(params, "cfg_scale", 4.0, 0.1, 20.0)
        sigma_shift = _float_param(params, "sigma_shift", 5.0, 0.1, 20.0)
        seed = _resolve_seed(params)
        torch_dtype_name = str(params.get("torch_dtype") or "bfloat16")
        if torch_dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ExtensionError(
                "torch_dtype must be one of: bfloat16, float16, float32.",
                code="invalid_param",
                details={"param": "torch_dtype", "value": torch_dtype_name},
            )
        disable_compile = bool_from_any(params.get("disable_compile"), True)
        output_path, relative_output = _resolve_output_path(workspace_dir, prompt, params)
        current_model_root = model_root()

        _progress(5, "validate-input")
        _log(
            "Validated process request.",
            nodeId=request.get("nodeId"),
            seconds=seconds,
            num_inference_steps=num_inference_steps,
            torch_dtype=torch_dtype_name,
            disable_compile=disable_compile,
        )
        _progress(10, "readiness")
        _runtime_env(disable_compile)
        _configure_warning_filters()
        _readiness_guard(current_model_root)

        _progress(40, "load-model")
        _log("Importing torch runtime and loading MOSS-SoundEffect pipeline.")
        with _suppress_library_stdio():
            import torch
            from moss_soundeffect_v2 import MossSoundEffectPipeline

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)
            _log("CUDA device detected.", device=device_name, capability=list(capability))
            if torch_dtype_name == "bfloat16" and capability < (8, 0):
                raise ExtensionError(
                    "bfloat16 is not supported for this GPU architecture. Use torch_dtype=float16.",
                    code="unsupported_dtype_for_gpu",
                    details={
                        "device": device_name,
                        "capability": list(capability),
                        "torch_dtype": torch_dtype_name,
                        "recommended_torch_dtype": "float16",
                    },
                )
        else:
            raise ExtensionError(
                "CUDA is not available in the runtime process.",
                code="cuda_unavailable",
                details={"torch_dtype": torch_dtype_name},
            )

        torch_dtype = getattr(torch, torch_dtype_name)
        with _suppress_library_stdio():
            pipe = MossSoundEffectPipeline.from_pretrained(
                str(current_model_root),
                torch_dtype=torch_dtype,
                device="cuda",
                local_files_only=True,
            )

        _progress(70, "generate-audio")
        with _suppress_library_stdio():
            audio = pipe(
                prompt=prompt,
                seconds=seconds,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                seed=seed,
            )

        _progress(90, "write-output")
        saved_path = _save_audio_with_soundfile(audio, output_path, sample_rate=48000)
        if not output_path.is_file():
            raise ExtensionError(
                "Pipeline completed but the declared WAV output file was not found.",
                code="missing_output_file",
                details={"declared_filePath": relative_output, "saved_name": saved_path.name},
            )
        if output_path.stat().st_size <= 0:
            raise ExtensionError(
                "Pipeline completed but the declared WAV output file is empty.",
                code="empty_output_file",
                details={"declared_filePath": relative_output},
            )

        _progress(100, "done")
        result_metadata = {
            "kind": "audio-artifact",
            "filePath": relative_output,
            "format": "wav",
            "sample_rate": 48000,
            "seconds": seconds,
            "seed": seed,
            "prompt": prompt,
            "model_id": HF_REPO,
            "nodeId": request.get("nodeId"),
            "modly_contract": "text metadata output plus workspace WAV filePath; audio is not first-class in current Modly",
        }
        emit_json(
            {
                "type": "done",
                "result": {
                    "filePath": relative_output,
                    "text": json.dumps(result_metadata, ensure_ascii=False),
                    "sample_rate": 48000,
                    "seconds": seconds,
                    "seed": seed,
                    "prompt": prompt,
                    "model_id": HF_REPO,
                    "nodeId": request.get("nodeId"),
                },
            }
        )
        return 0
    except ExtensionError as exc:
        _error(str(exc), code=exc.code, details=exc.details)
        return 1
    except Exception as exc:
        _error(
            str(exc),
            code="runtime_failed",
            details={"traceback_type": type(exc).__name__, "traceback_tail": traceback.format_exc(limit=3).splitlines()[-3:]},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
