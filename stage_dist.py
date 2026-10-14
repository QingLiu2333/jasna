"""Assemble the final self-contained Jasna dist directory.

PyInstaller cannot put data at the *dist root* for a onedir build, but the code
resolves several things relative to ``sys.executable``:

    <exe_dir>/tools/ffmpeg.exe, ffprobe.exe   os_utils._find_bundled_executable
    <exe_dir>/model_weights/                  engine_paths.model_weights_dir
    <exe_dir>/assets/                         gui.branding.brand_asset_path
    <exe_dir>/vlc/                            gui.raw_player._configure_bundled_vlc
    <exe_dir>/*.fatbin                        media.cuda_kernel.fatbin_path

Run after PyInstaller:  python D:\\jasna2\\stage_dist.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\jasna2")
REPO = ROOT / "src"                 # upstream clone: assets/, docs/, licence files
WORK = ROOT / "work"                # patched source (the ``jasna`` package lives here)
BUILD = ROOT / "build"              # tools/, vlc/ and the model_weights/ tree
STAGING = BUILD
# ``jasna.spec`` writes its onedir output to <distpath>/jasna. Override the root
# with JASNA_DIST_DIR to stage a build made next to a locked/older install.
DIST_ROOT = Path(os.environ.get("JASNA_DIST_DIR", str(ROOT / "dist")))
DIST = DIST_ROOT / "jasna"

# VLC ships far more than libvlc needs: browser plugins, Lua scripts, skins,
# locales and the installer payload can all go.
VLC_DROP_DIRS = {"locale", "lua", "skins", "hrtfs", "msi", "sdk", "include", "lib"}
VLC_DROP_FILES = {"axvlc.dll", "npvlc.dll", "vlc.exe", "vlc-cache-gen.exe"}
VLC_KEEP_FILES = {"libvlc.dll", "libvlccore.dll", "COPYING.txt", "AUTHORS.txt", "NEWS.txt", "README.txt"}


def log(msg: str) -> None:
    print(msg, flush=True)


def copy_tree(src: Path, dst: Path, *, skip_names: set[str] | None = None) -> int:
    if not src.is_dir():
        log(f"  ! missing source dir {src}")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    skip_names = skip_names or set()
    for item in src.iterdir():
        if item.name in skip_names:
            continue
        target = dst / item.name
        if item.is_dir():
            count += copy_tree(item, target, skip_names=skip_names)
        else:
            shutil.copy2(item, target)
            count += 1
    return count


def trim_vlc(vlc_dir: Path) -> None:
    removed = 0
    for child in list(vlc_dir.iterdir()):
        if child.is_dir() and child.name in VLC_DROP_DIRS:
            shutil.rmtree(child)
            removed += 1
        elif child.is_file() and child.name in VLC_DROP_FILES:
            child.unlink()
            removed += 1
    log(f"  vlc: dropped {removed} unneeded entries")


def dir_size(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


ENGINES_README = """\
Engines in this folder are stored per graphics card.

    engines/<family>/<gpu-token>/<engine files>

The token describes the exact card and build the engine belongs to (compute
capability, SM count, memory, driver and TensorRT version), and the matching
gpu_identity.txt next to the engines says which card that is.

A TensorRT engine only runs on the card it was built for, so a card that has no
folder here simply compiles its own set on first use - that is expected, not an
error. Engine sets that turn out to be unusable on the installed card are moved
aside as "<token>.unusable-<date>" and rebuilt.

JASNA_ENGINE_PER_GPU=0 falls back to the old flat layout, and
JASNA_ADOPT_LEGACY_ENGINES=1 lets engines from an older install be reused (same
card only).
"""


def prune_legacy_engines(model_weights: Path) -> None:
    """Remove engine files that are not filed under a per-GPU archive.

    A TensorRT plan only deserializes on the card that built it. Engines left in
    the historic flat layout (``model_weights/*.engine``, ``*_sub_engines/``)
    carry no record of their origin, so an install on another card would find
    them, try to use them and fail with a misleading "checkpoint is corrupted"
    error. Only the token-keyed ``engines/`` archive may be shipped.
    """
    if not model_weights.is_dir():
        return

    removed_files = 0
    for stray in sorted(model_weights.glob("*.engine")):
        stray.unlink()
        removed_files += 1
    removed_dirs = 0
    for folder in sorted(model_weights.glob("*_sub_engines")):
        if folder.is_dir():
            removed_files += sum(1 for f in folder.rglob("*") if f.is_file())
            shutil.rmtree(folder)
            removed_dirs += 1
    log(f"  pruned {removed_files} flat engine file(s) and {removed_dirs} legacy "
        "engine folder(s) - only per-GPU archives are shipped")

    engines = model_weights / "engines"
    if engines.is_dir():
        (engines / "README.txt").write_text(ENGINES_README, encoding="utf-8")
        log("  wrote engines/README.txt")


def main() -> int:
    if not (DIST / "jasna.exe").is_file():
        log(f"ERROR: {DIST / 'jasna.exe'} not found; run PyInstaller first.")
        return 1

    log(f"Staging into {DIST}")

    log("- tools/ (ffmpeg + ffprobe and their DLLs)")
    log(f"  copied {copy_tree(STAGING / 'tools', DIST / 'tools')} files")

    log("- vlc/")
    n = copy_tree(STAGING / "vlc", DIST / "vlc")
    trim_vlc(DIST / "vlc")
    log(f"  copied {n} files")

    log("- assets/")
    log(f"  copied {copy_tree(REPO / 'assets', DIST / 'assets')} files")

    log("- model_weights/ (incl. per-GPU engine archives under engines/)")
    log(f"  copied {copy_tree(BUILD / 'model_weights', DIST / 'model_weights')} files")
    prune_legacy_engines(DIST / "model_weights")

    log("- *.fatbin (CUDA kernels expected at the dist root when frozen)")
    media = WORK / "jasna" / "media"
    fatbins = sorted(media.glob("*.fatbin"))
    for fatbin in fatbins:
        shutil.copy2(fatbin, DIST / fatbin.name)
    log(f"  copied {len(fatbins)} fatbins: {', '.join(f.name for f in fatbins)}")

    log("- licence and source documents")
    docs = [
        "LICENSE",
        "LICENSING.md",
        "NOTICE",
        "RELEASE_SOURCES.md",
        "README.md",
        "README.zh.md",
    ]
    copied = 0
    for name in docs:
        src = REPO / name
        if src.is_file():
            shutil.copy2(src, DIST / name)
            copied += 1
    docs_dir = REPO / "docs"
    if docs_dir.is_dir():
        copy_tree(docs_dir, DIST / "docs")
    log(f"  copied {copied} documents plus docs/")

    log("")
    log(f"Total dist size: {dir_size(DIST):.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
