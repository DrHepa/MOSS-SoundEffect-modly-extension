# MOSS-SoundEffect Modly Extension

Process extension workspace for **MOSS-SoundEffect v2.0** using the upstream runtime from `OpenMOSS/MOSS-TTS` and weights from `OpenMOSS-Team/MOSS-SoundEffect-v2.0`.

This repository is an MIT-licensed Modly integration wrapper. MOSS-SoundEffect v2.0 itself, including the upstream runtime and model weights, belongs to OpenMOSS/OpenMOSS-Team. See `NOTICE` and the upstream model/runtime links before redistributing weights or using the model commercially.

## What this extension does

- Accepts a text prompt in English or Chinese.
- Runs the upstream `MossSoundEffectPipeline`.
- Writes a `.wav` artifact under `Workflows/MOSS-SoundEffect/` inside the workflow `workspaceDir`.
- Emits Modly process-runner JSONL messages: `progress`, `log`, `done`, `error`.

## Modly contract posture

- Bucket: `process-extension`
- Setup seam: root `setup.py`
- Runtime seam: root `moss_soundeffect_process.py`
- Model ownership: Modly global model assets under `models/moss-soundeffect-v2-process-extension/generate-soundeffect`
- Output contract: `done.result.filePath` points to a workspace `.wav` audio artifact; `done.result.text` contains JSON metadata fallback

This extension expects Modly builds with `audio` as a first-class `ArtifactKind`. For older Modly builds that only support `image`, `text`, `mesh`, and `scene`, the process still emits JSON metadata in `done.result.text` and the generated `.wav` in `done.result.filePath`, but the node manifest may need to be downgraded to `output: "text"` until audio support is available.

## Platform matrix

This extension now resolves a host/backend lane before any `pip install` happens and fails fast on unsupported matrices.

Implemented matrix:

- Linux `aarch64` + NVIDIA `SM >= 121` + Python `3.12`: use local wheelhouse lane `linux-aarch64/local-wheelhouse-sm121` when `dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312/WHEELHOUSE.json` exists
- Linux `x86_64` or `aarch64` + NVIDIA `SM >= 121` without a validated local wheelhouse: fail fast. Public wheels are not assumed compatible with this host class unless explicitly verified.
- Linux `x86_64` or `aarch64` + NVIDIA `SM == 120`: prefer `cu130` when driver evidence is new enough; otherwise fallback to experimental `cu128`
- Linux `x86_64` or `aarch64` + NVIDIA `SM < 120`: `cu128` lane preserved to avoid breaking established users
- Windows + NVIDIA `SM >= 121`: `cu130` as **experimental only**
- Windows `x86_64` + Python `3.11` + pre-`SM 8.0` NVIDIA GPUs: setup and model loading can be validated with `cu126`, but practical generation is **not supported** for this GPU class. Treat this lane as setup/import probe only.

Fail-fast / unsupported:

- macOS / MPS
- CPU-only
- Windows lanes below `SM 121` for practical inference unless explicitly validated on the target GPU class
- Any host where setup cannot resolve an evidence-backed lane

Critical rule: legacy `cu128` wheels are blocked for `SM 12.1+` host classes. Those hosts require a validated CUDA 13 wheelhouse or another explicitly verified lane.

For Linux `aarch64` + `cp312` + `SM 12.1+`, the preferred path is a local wheelhouse built from source with `TORCH_CUDA_ARCH_LIST="12.1a"`. Public indexes are not assumed to cover that host class unless explicitly verified.

## Install flow in Modly UI

1. Install/apply the extension from this repository/workspace.
2. Run extension setup from Modly so Electron executes root `setup.py`.
3. Let extension setup download model assets into Modly's global `models/moss-soundeffect-v2-process-extension/generate-soundeffect`, or place an equivalent Hugging Face snapshot there before running.
4. Run the workflow node with a text prompt.

## What setup does

`setup.py` is designed for Modly's injected JSON context and will:

- use `ext_dir` and `python_exe` from the setup payload when provided
- create an extension-owned `venv/`
- apply lane-specific Python support. The local SM121 wheelhouse lane currently targets Python `3.12`; the Windows pre-SM80 probe lane can use Modly's Python `3.11` but is not considered viable for practical inference.
- resolve the host lane from `gpu_sm`, `cuda_version`, `python_exe`, `ext_dir`, local platform facts, and a light `nvidia-smi` probe when available
- emit the detected host facts plus selected lane before dependency installation
- fail fast before installation when the host matrix is unsupported
- install lane-specific torch packages instead of hardcoding a single `cu128` stack
- prefer a local `wheelhouse` lane for Linux `aarch64` + `cp312` + `SM 12.1+` when `WHEELHOUSE.json` is present under `dependencies/wheelhouse/pytorch-sm121-cu130-aarch64-cp312`
- install the upstream Python runtime package from public GitHub
- run `pip check`
- run a post-install probe for `torch`, `torch.version.cuda`, CUDA availability, device capability, torch arch coverage, and `MossSoundEffectPipeline`
- download the Hugging Face model snapshot into the extension-owned logical model directory by default
- validate required model sentinels after download
- write `.modly/setup-ready.json`

## Model asset provisioning

Important: Modly's UI checks model readiness under the global `modelsDir`, not inside the extension folder. For this extension, setup provisions the same owner-scoped global model directory that the UI checks.

- expected model root: `models/moss-soundeffect-v2-process-extension/generate-soundeffect`
- default setup behavior: download `OpenMOSS-Team/MOSS-SoundEffect-v2.0` into that root
- optional setup escape hatch: pass `skip_model_download=true` / `download_model_assets=false` only for dry setup or manually pre-seeded assets
- runtime behavior: fail closed if sentinels are missing; runtime generation never downloads weights

This is deliberate. Downloading from the process runtime during generation creates ambiguous failures and bad UX. Setup owns model asset provisioning and records `downloads_started`, `missing_model_files`, and readiness status in `.modly/setup-ready.json`.

The node manifest intentionally does not expose `hf_repo`/`download_check` as a UI model-download button until Modly's process-extension ownership support is available in the running app. Otherwise the current UI can show a false Download state for process extensions even when setup has already provisioned the global model directory.

## Required model sentinels

- `model_index.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `vae/vae_128d_48k.pth`
- `scheduler/scheduler_config.json`
- `tokenizer/tokenizer.json`
- `text_encoder/model.safetensors.index.json`
- at least one `text_encoder/model-*.safetensors` shard

## Runtime parameters

- `prompt`: text prompt; English or Chinese
- `seconds`: default `10`, max `30`
- `num_inference_steps`: default `100`
- `cfg_scale`: default `4.0`
- `sigma_shift`: default `5.0`
- `seed`: default `0`, `-1` means random
- `torch_dtype`: `bfloat16`, `float16`, `float32`
- `disable_compile`: default `true`; if enabled, runtime sets `TORCHDYNAMO_DISABLE=1` and `TORCHINDUCTOR_DISABLE=1` before torch-heavy imports. This is now an explicit runtime choice again, not something silently forced by an override env path.
- `output_name`: optional custom filename stem; sanitized and forced to `.wav`

### Parameter notes by architecture

- `SM 12.1+`: use a validated local wheelhouse or another explicitly verified CUDA 13 lane. Do not rely on legacy CUDA 12.8 wheels for this GPU class.
- Pre-`SM 8.0` NVIDIA GPUs: do **not** use `bfloat16`; the runtime rejects it and recommends `float16`. Even with `float16`, this GPU class is not considered viable for useful MOSS-SoundEffect v2 generation based on current tests.
- For any unvalidated GPU class, start with a short smoke test before attempting long/high-step generation.

## Output behavior

- output directory: `Workflows/MOSS-SoundEffect/`
- output filename: sanitized from `output_name` or prompt text
- output type: WAV, 48 kHz
- manifest workflow output: `audio`
- `done.result.text`: JSON metadata containing the WAV path and generation parameters
- `done.result.filePath`: workspace-relative WAV artifact path

Absolute output paths and traversal are rejected.

## Validation ladder

1. Static review of `manifest.json`, `setup.py`, and runtime path safety
2. Modly setup run on Linux + NVIDIA CUDA with Python 3.12
3. Confirm `.modly/setup-ready.json` exists
4. Confirm logical model directory contains all sentinels
5. Smoke test with a short prompt and low-risk parameters
6. Confirm `done.result.filePath` resolves to a real workspace WAV file

## Known upstream/runtime gaps

- Native Modly audio output handling requires a build that includes the audio ArtifactKind/workflow/preview changes.
- For older Modly builds without audio support, temporarily change the manifest node to `output: "text"` and rely on `done.result.text` metadata plus `done.result.filePath`.
- First run may be slow due to `torch.compile` / Triton graph compilation.
- `disable_compile=true` is still a valid escape hatch for fragile compile stacks, but it is NOT a substitute for the correct setup lane.
- If the host is `SM 12.1+`, setup must land on a validated CUDA 13-compatible lane. Legacy `cu128` is intentionally rejected for that host class.
- Source builds for Triton/PyTorch stay OUTSIDE automatic Modly setup. This repository now includes `tools/build-sm121-wheelhouse/` with manual, reproducible helper scripts and a `WHEELHOUSE.example.json` template, but setup only consumes an already prepared local wheelhouse.
- Windows pre-`SM 8.0` NVIDIA GPUs can complete setup/import probes in some cases, but they are not supported practical inference targets for MOSS-SoundEffect v2.

## Files in this workspace

- `manifest.json`: Modly extension contract
- `setup.py`: setup entrypoint wrapper
- `moss_soundeffect_process.py`: process runner entrypoint wrapper
- `moss_soundeffect_ext/setup_runtime.py`: setup implementation
- `moss_soundeffect_ext/host_compat.py`: host detection and lane resolution
- `moss_soundeffect_ext/process_runtime.py`: process JSONL implementation
- `moss_soundeffect_ext/common.py`: shared constants and safety helpers
- `tools/build-sm121-wheelhouse/`: manual docs/scripts to build a local SM121 wheelhouse without making setup.py compile PyTorch or Triton
