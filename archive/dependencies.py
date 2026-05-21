"""
dependencies.py

One-command setup for this project.

Run:
  python dependencies.py

It will install (or confirm) the packages needed to run:
- new_app.py (UI webcam detection + optional Arduino serial)
- servo_tracker.py (optional)
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable


# Keep this short and practical for Windows users.
REQUIRED_PIP_PACKAGES: list[str] = [
    "ultralytics",
    "opencv-python",
    "pyside6",
    "pyserial",
]


# Map import name -> pip package name (when different)
IMPORT_CHECKS: dict[str, str] = {
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "PySide6": "pyside6",
    "serial": "pyserial",
}


def _missing_imports() -> list[str]:
    missing: list[str] = []
    for import_name, pip_name in IMPORT_CHECKS.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(pip_name)
    # preserve order, unique
    seen: set[str] = set()
    ordered: list[str] = []
    for p in REQUIRED_PIP_PACKAGES:
        if p in missing and p not in seen:
            ordered.append(p)
            seen.add(p)
    # include any extras not in REQUIRED_PIP_PACKAGES
    for p in missing:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def _pip_install(packages: Iterable[str]) -> None:
    pkgs = list(packages)
    if not pkgs:
        return
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    print("Installing:", " ".join(pkgs))
    subprocess.check_call(cmd)


def main() -> None:
    missing = _missing_imports()
    if not missing:
        print("All dependencies already installed.")
        return
    _pip_install(missing)
    still_missing = _missing_imports()
    if still_missing:
        raise SystemExit(f"Some dependencies still missing: {still_missing}")
    print("Done. You can now run: python new_app.py")


if __name__ == "__main__":
    main()

