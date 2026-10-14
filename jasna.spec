# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the self-contained Jasna bundle.

Layout produced in ``D:\\jasna2\\dist\\jasna`` (matching what the code expects
from a frozen build):

    jasna.exe                 entry: jasna/__main__.py (CLI + GUI dispatch)
    _internal/                everything PyInstaller bundles
    tools/ffmpeg.exe          find_executable("ffmpeg")  -> <exe_dir>/tools
    tools/ffprobe.exe         find_executable("ffprobe") -> <exe_dir>/tools
    model_weights/            model_weights_dir()        -> <exe_dir>/model_weights
    assets/                   brand_asset_path()         -> <exe_dir>/assets
    vlc/                      libvlc for the native player
    *.fatbin                  fatbin_path() when frozen  -> <exe_dir>/<name>

``tools``, ``model_weights``, ``assets``, ``vlc`` and the fatbins are staged
outside the bundle by ``stage_dist.py`` (PyInstaller cannot place data at the
dist root for a onedir build).

HARDWARE-ARCHITECTURE NOTE
--------------------------
The default ``torch``/``torch_tensorrt`` wheels here are compiled for
``sm_75 sm_80 sm_86 sm_90 sm_100 sm_120``. That means the frozen exe covers
Turing through Blackwell. Volta (sm_70) and Pascal (sm_61) need a matching
torch build; see README-BUILD.md.

The same spec builds both runtime trees of the dual-runtime package: pass
``--distpath`` for the target tree and set ``JASNA_DIST_DIR`` for
``stage_dist.py``. The only difference between the trees is the venv
(``requirements-legacy.txt`` for ``rt/cu126``).
"""

import pathlib
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

PACKAGE_ROOT = pathlib.Path(r"D:\jasna2\work")          # parent of the ``jasna`` package
HIDDEN_LIST = pathlib.Path(r"D:\jasna2\hiddenimports.txt")

# 1. ``D:\jasna2\work\jasna\packaging`` is jasna's *own* sub-package: putting
#    the package directory itself (``...\work\jasna``) on sys.path makes
#    ``import packaging`` resolve to that stub instead of the real library, and
#    PyInstaller's own hooks then fail with
#    ``No module named 'packaging.requirements'``.
# 2. Anything on sys.path that is the *parent* (``...\work``) is fine.
#
# An older checkout of the repository (``D:\jasna2\src``, or the directory of an
# earlier build) would shadow the package the same way, so both are dropped from
# sys.path before analysis starts.
_SHADOWING = {p.resolve() for p in (pathlib.Path(r"D:\jasna2\src"), PACKAGE_ROOT)}
for _entry in list(sys.path):
    try:
        _resolved = pathlib.Path(_entry or ".").resolve()
    except OSError:
        continue
    if _resolved in _SHADOWING:
        sys.path.remove(_entry)
sys.path.insert(0, str(PACKAGE_ROOT))

# Packages with data files, binaries or lazy submodule imports that static
# analysis misses.
COLLECT_ALL = [
    "customtkinter",
    "tkinterdnd2",
    "av",
    "ultralytics",
    "mmengine",
    "diffusers",
    "huggingface_hub",
    "safetensors",
    "tensorrt",
    "tensorrt_libs",
    "tensorrt_bindings",
    "nvvfx",
    "torch_tensorrt",
    "cv2",
    "onnx",
    "cryptography",
    "polars",
    "matplotlib",
    "yapf",
    "dllist",
    "addict",
    "rich",
]

datas = []
binaries = []
hiddenimports = []

for pkg in COLLECT_ALL:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_all({pkg!r}) failed: {exc}")
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

if HIDDEN_LIST.is_file():
    for line in HIDDEN_LIST.read_text(encoding="utf-8").split():
        if line and line not in hiddenimports:
            hiddenimports.append(line)

# Submodules jasna imports dynamically (registry / platform-dispatch style).
# ``jasna.gpu_identity`` and ``jasna.arch_compat`` are new: they live behind
# lazy imports from the GUI/engine paths, so collect_submodules is what keeps
# them in the bundle.
for pkg in ("jasna", "torch", "torchvision", "tensorrt", "nvvfx", "vlc"):
    try:
        for name in collect_submodules(pkg):
            if name not in hiddenimports:
                hiddenimports.append(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_submodules({pkg!r}) failed: {exc}")

hiddenimports += [
    "vlc",
    "PIL._tkinter_finder",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
]

a = Analysis(
    [str(PACKAGE_ROOT / "jasna" / "__main__.py")],
    pathex=[str(PACKAGE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Not installed / not used by this build.
        "python_vali",
        "sageattention",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
        "nuitka",
        "PyInstaller",
        # The ``rfdetr`` torch package is only needed where the detector runs as a
        # torch model instead of a TensorRT engine. Neither tree can use it today:
        # the modern tree always has TensorRT, and the legacy tree ships no
        # ``rfdetr-*.pt`` checkpoint (only the ONNX export), so bundling ~150 MiB
        # of it would buy nothing. Listed here so a future tree that ships the .pt
        # has to make that decision explicitly.
        "rfdetr",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="jasna",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(pathlib.Path(r"D:\jasna2\src\assets\jasna-logo.ico")),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="jasna",
)
