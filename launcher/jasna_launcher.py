"""Start the Jasna runtime that matches the installed GPU.

One download carries three complete runtimes, because no single
PyTorch/TensorRT/CUDA build covers the whole supported range:

* ``rt/cu130`` — CUDA 13, ``sm_75 .. sm_120`` (Turing through Blackwell),
  full TensorRT 10.16.
* ``rt/cu124`` — CUDA 12.4, ``sm_70`` (Volta), torch 2.5 + torch_tensorrt 2.5
  + TensorRT 10.3 — the newest TensorRT line that still accepts Volta.
* ``rt/cu126`` — CUDA 12.6, ``sm_50 .. sm_90`` (Maxwell through Hopper), no
  TensorRT — the 10.x line cannot compile for Pascal at all, so the Pascal
  tier runs the PyTorch path.

CUDA 13 removed offline compilation below compute capability 7.5, and the CUDA
runtime cannot be picked inside a single process (torch's DLL names are fixed),
so the choice has to be made *before* either interpreter starts. That is this
program: it reads the compute capability straight from the CUDA driver, then
replaces itself with the matching ``jasna.exe``.

Being wrong costs a fallback, never a crash: an unreadable capability, a missing
tree or a failed exec all fall through to whichever tree is present, and the reason
goes to stderr.

``JASNA_RUNTIME=cu126|cu124|cu130`` forces a tree (used by the build's own tests).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

MODERN = "cu130"
VOLTA = "cu124"
LEGACY = "cu126"

# The capability at which CUDA 13 starts. Everything below it needs a CUDA 12.x
# tree; Volta (7.0) additionally wants the last TensorRT that accepts it.
CUDA13_FLOOR = (7, 5)
VOLTA_CAPABILITY = (7, 0)

_TREES = (MODERN, VOLTA, LEGACY)
_FORCED_KEYS = {
    MODERN: {MODERN, "cu13", "modern", "new"},
    VOLTA: {VOLTA, "cu124", "volta"},
    LEGACY: {LEGACY, "cu12", "cu126", "legacy", "old", "pascal"},
}


def _root() -> Path:
    """The package root: next to this launcher, or the PyInstaller extraction dir."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _capability_via_driver() -> tuple[int, int] | None:
    """Read compute capability through the CUDA driver, with no torch involved."""
    try:
        nvcuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    try:
        if nvcuda.cuInit(0) != 0:
            return None
        device = ctypes.c_int()
        if nvcuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return None
        major, minor = ctypes.c_int(), ctypes.c_int()
        if nvcuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device) != 0:
            return None
        return int(major.value), int(minor.value)
    except Exception:
        return None


def _capability_via_smi() -> tuple[int, int] | None:
    """Fallback for a driver whose nvcuda entry points are not importable."""
    smi = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    if not smi.is_file():
        return None
    try:
        out = subprocess.run(
            [str(smi), "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except OSError:
        return None
    text = out.strip().splitlines()[0].strip() if out.strip() else ""
    major, _, minor = text.partition(".")
    if major.isdigit() and minor.isdigit():
        return int(major), int(minor)
    return None


def detect_capability() -> tuple[int, int] | None:
    return _capability_via_driver() or _capability_via_smi()


def choose_runtime() -> tuple[str, str]:
    """``(tree name, why)`` for the GPU that is installed."""
    forced = os.environ.get("JASNA_RUNTIME", "").strip().lower()
    for name, keys in _FORCED_KEYS.items():
        if forced in keys:
            return name, f"{forced} forced by JASNA_RUNTIME"

    capability = detect_capability()
    if capability is None:
        # No readable NVIDIA device: the modern tree is the better default, and a
        # non-NVIDIA build does not go through here at all.
        return MODERN, "no readable CUDA device; defaulting to the CUDA 13 runtime"
    if capability >= CUDA13_FLOOR:
        return MODERN, f"compute capability {capability[0]}.{capability[1]} >= {CUDA13_FLOOR[0]}.{CUDA13_FLOOR[1]}"
    if capability == VOLTA_CAPABILITY:
        return VOLTA, (
            f"compute capability 7.0 (Volta): the CUDA 13 tree has no Volta kernels "
            f"and CUDA 13's TensorRT dropped Volta, so this uses the dedicated "
            f"torch 2.5 + TensorRT 10.3 runtime"
        )
    return LEGACY, (
        f"compute capability {capability[0]}.{capability[1]}: TensorRT cannot compile "
        f"for this architecture, so this uses the CUDA 12.6 PyTorch runtime"
    )


def resolve_target(root: Path, preferred: str) -> tuple[Path, str]:
    """The exe to start, falling back through the other trees when one is missing.

    A tree is either ``rt/<name>/jasna.exe`` or ``rt/<name>/jasna/jasna.exe``: the
    PyInstaller build drops the exe in a folder named after the spec, and whether
    that folder is kept or flattened depends on how the package was assembled.
    """
    exe = "jasna.exe" if os.name == "nt" else "jasna"
    order = [preferred, *(_ for _ in _TREES if _ != preferred)]
    for name in order:
        for candidate in (root / "rt" / name / exe, root / "rt" / name / "jasna" / exe):
            if candidate.is_file():
                note = "" if name == preferred else f" (falling back: the {preferred} tree is missing)"
                return candidate, note
    raise SystemExit(
        f"Jasna is not installed correctly: none of "
        f"{', '.join('rt/' + name for name in _TREES)} was found under {root}"
    )


def runtime_name_of(target: Path) -> str:
    """The tree a target belongs to: ``rt/<name>/...`` -> ``<name>``."""
    parts = target.parts
    if "rt" in parts:
        index = len(parts) - 1 - parts[::-1].index("rt")
        if index + 1 < len(parts):
            return parts[index + 1]
    return target.parent.name


def main() -> int:
    root = _root()
    preferred, why = choose_runtime()
    target, note = resolve_target(root, preferred)
    print(f"[jasna] {runtime_name_of(target)}: {why}{note}", file=sys.stderr, flush=True)

    argv = [str(target), *sys.argv[1:]]
    try:
        # Replacing the process (rather than waiting on a child) keeps the console
        # handling identical to running the tree directly: the app detaches the
        # console itself for a GUI launch, and that has to release the window the
        # launcher inherited.
        os.execv(str(target), argv)
    except OSError as exc:
        print(f"[jasna] could not start {target}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
