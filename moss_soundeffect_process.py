from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _venv_python() -> Path:
    if os.name == "nt":
        return ROOT / "venv" / "Scripts" / "python.exe"
    return ROOT / "venv" / "bin" / "python"


def _ensure_extension_venv() -> None:
    # Native Windows venvs created from Modly's embedded Python can report the
    # base embedded interpreter in sys.executable even when launched through
    # venv\Scripts\python.exe. Re-execing there creates a second process and can
    # break the JSONL stdin/stdout contract, leaving Modly stuck in "generating".
    # On Windows, trust the launcher/manifest/setup to provide the correct env.
    if os.name == "nt":
        return
    venv_python = _venv_python()
    if os.environ.get("MOSS_SFX_PROCESS_IN_VENV") == "1":
        return
    if not venv_python.exists():
        return
    current = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if current == target:
        return
    env = os.environ.copy()
    env["MOSS_SFX_PROCESS_IN_VENV"] = "1"
    os.execve(str(target), [str(target), str(Path(__file__).resolve())], env)


_ensure_extension_venv()

from moss_soundeffect_ext.process_runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
