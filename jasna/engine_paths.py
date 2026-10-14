"""Lightweight engine-path helpers — no torch / tensorrt imports apart from
:mod:`jasna.gpu_identity`, which itself is import-safe without a CUDA device.

Every TensorRT engine path here is *device-qualified*: the compiled plan lives
in a directory keyed by the identity token of the GPU it was built for (see
:mod:`jasna.gpu_identity`). Engines for different cards therefore never collide
and never shadow each other, and swapping the physical card makes the
application compile a fresh set for the new card instead of failing to
deserialize an engine built for the old one.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from jasna._frozen import is_frozen

log = logging.getLogger(__name__)

# The device the engine paths are resolved for. ``None`` means "the CUDA device
# this process is currently working with", which is what every normal launch
# wants. The CLI/engine-compiler subprocess pins it explicitly so a compile
# targeted at device N cannot be mis-filed under device M.
_engine_device_index: int | None = None

# Whether to write into a per-GPU subdirectory. Disable with
# ``JASNA_ENGINE_PER_GPU=0`` to get the historic flat layout back (diagnostics
# and existing installs that already carry a hand-built engine).
_PER_GPU_ENV = "JASNA_ENGINE_PER_GPU"

# Name of the small text file written next to an engine archive describing the
# GPU it belongs to. Kept for humans reading the folder and for the GUI's
# engine-status panel.
ARCHIVE_INFO_FILENAME = "gpu_identity.txt"

# Suffix given to an archive directory that turned out to be unusable on this
# card. The directory is renamed rather than deleted so nothing is destroyed and
# every existence check reports the engines as missing.
UNUSABLE_SUFFIX = ".unusable"

# Opt-in for reusing engine files from an install that predates the per-GPU
# archive (or from another machine). Off by default: a TensorRT plan is only
# valid on the card it was built on, so reusing an unknown set usually costs
# more (a failed load that looks like a corrupt checkpoint) than it saves.
_ADOPT_ENV = "JASNA_ADOPT_LEGACY_ENGINES"


def set_engine_device_index(index: int | None) -> None:
    """Pin the CUDA device the engine paths resolve for."""
    global _engine_device_index
    _engine_device_index = None if index is None else int(index)


def engine_device_index() -> int | None:
    return _engine_device_index


def per_gpu_engines_enabled() -> bool:
    return os.environ.get(_PER_GPU_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _resolve_index() -> int:
    if _engine_device_index is not None:
        return _engine_device_index
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:
        pass
    return 0


def device_identity_token() -> str:
    """Identity token of the GPU engine paths are currently keyed by.

    Empty string when the archive is disabled or no GPU can be identified, in
    which case the historic flat layout is used.
    """
    if not per_gpu_engines_enabled():
        return ""
    try:
        from jasna.gpu_identity import current_identity_token

        return current_identity_token(_resolve_index())
    except Exception:
        return ""


def device_identity():
    """The full :class:`~jasna.gpu_identity.GpuIdentity`, or ``None``."""
    try:
        from jasna.gpu_identity import describe_current_device

        return describe_current_device(_resolve_index())
    except Exception:
        return None


def model_weights_dir() -> Path:
    """Resolve the ``model_weights`` directory.

    When running as a frozen executable, models are bundled next to the executable.
    In a dev / source checkout we fall back to a CWD-relative path so existing dev
    workflows keep working.
    """
    if is_frozen():
        return Path(sys.executable).parent / "model_weights"
    return Path("model_weights")


def engines_root() -> Path:
    """Directory holding every per-GPU engine archive."""
    return model_weights_dir() / "engines"


_DEFAULT_RESTORATION_MODELS = {
    "basicvsrpp": "lada_mosaic_restoration_model_generic_v1.2.pth",
    "ltx": "ltx-restore",
}


def default_restoration_model_path(model_name: str) -> Path:
    return model_weights_dir() / _DEFAULT_RESTORATION_MODELS[model_name]


def typed_engine_dir(kind: str, *, create: bool = False) -> Path | None:
    """Per-GPU archive directory for one engine family, or ``None`` when the
    archive is off (callers then use the flat layout).

    ``kind`` names the engine family, e.g. ``basicvsrpp`` or ``detection``.
    """
    token = device_identity_token()
    if not token:
        return None
    path = engines_root() / kind / token
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_archive_identity(kind: str, directory: Path) -> None:
    """Drop a human-readable description of the owning GPU into an archive."""
    identity = device_identity()
    if identity is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        text = (
            f"{identity.describe()}\n"
            f"token           = {identity.token}\n"
            f"identity string = {identity.hash_inputs()}\n"
            f"name            = {identity.name}\n"
        )
        (directory / ARCHIVE_INFO_FILENAME).write_text(text, encoding="utf-8")
    except OSError:
        pass


def other_archives(kind: str) -> list[dict]:
    """Every *other* GPU archive present for one engine family.

    Used by the GUI to tell the user "engines already exist for card X, this
    launch will build its own".
    """
    root = engines_root() / kind
    if not root.is_dir():
        return []
    current = device_identity_token()
    found: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == current or UNUSABLE_SUFFIX in child.name:
            continue
        info = child / ARCHIVE_INFO_FILENAME
        description = ""
        if info.is_file():
            try:
                description = info.read_text(encoding="utf-8").strip().splitlines()[0]
            except OSError:
                description = ""
        found.append(
            {
                "token": child.name,
                "path": str(child),
                "description": description or child.name,
                "engine_count": sum(1 for p in child.rglob("*.engine")),
            }
        )
    return found


def _rename_archive_dir(directory: Path, *, reason: str = "", kind: str = "") -> Path | None:
    """Rename an engine directory to ``<name>.unusable-<timestamp>``."""
    import time

    try:
        if not directory.is_dir():
            return None
    except OSError:
        return None
    if UNUSABLE_SUFFIX in directory.name:
        return None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = directory.with_name(f"{directory.name}{UNUSABLE_SUFFIX}-{stamp}")
    counter = 1
    while target.exists():
        target = directory.with_name(f"{directory.name}{UNUSABLE_SUFFIX}-{stamp}-{counter}")
        counter += 1
    try:
        directory.rename(target)
    except OSError:
        log.warning("Could not move the unusable engine archive %s", directory, exc_info=True)
        return None

    try:
        identity = device_identity()
        (target / "unusable_reason.txt").write_text(
            f"reason   = {reason or 'engine could not be loaded on this GPU'}\n"
            f"gpu      = {identity.describe() if identity else '?'}\n"
            f"moved at = {stamp}\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    log.warning(
        "Moved the unusable %s engine archive to %s - this GPU will compile fresh engines.",
        kind or directory.parent.name,
        target,
    )
    return target


def quarantine_engine_archive(kind: str, *, reason: str = "", token: str | None = None) -> Path | None:
    """Move this card's archive for ``kind`` aside because it cannot be used.

    Returns the new location, or ``None`` when there was nothing to move.

    Renaming (instead of deleting) keeps the files for inspection and is
    atomic; more importantly every ``*_exist`` check now reports the engine set
    as missing, so the normal compile path rebuilds it for the card that is
    actually installed.
    """
    token = token if token is not None else device_identity_token()
    if not token:
        return None
    return _rename_archive_dir(engines_root() / kind / token, reason=reason, kind=kind)


def quarantine_engine_archive_for_path(engine_path: str | Path, *, reason: str = "") -> Path | None:
    """Quarantine the archive directory that owns ``engine_path``.

    Works for both layouts: ``<model_weights>/engines/<kind>/<token>/<file>``
    (the per-GPU archive) and ``<model_weights>/<model>_sub_engines/<file>``
    (the historic flat layout). A file that does not live in one of those is
    left alone - it is not ours to move.
    """
    directory = Path(engine_path).parent
    looks_like_archive = (
        directory.name.endswith("_sub_engines")
        or directory.parent.parent.name == "engines"  # engines/<kind>/<token>/file
        or (directory / ARCHIVE_INFO_FILENAME).is_file()
    )
    if not looks_like_archive:
        return None
    return _rename_archive_dir(directory, reason=reason)


def engine_system_suffix() -> str:
    return ".win" if os.name == "nt" else ".linux"


def legacy_engine_files() -> list[Path]:
    """Engine files sitting in a *pre-archive* layout, for diagnostics/logging.

    That is: ``<model_weights>/*.engine`` and ``<model>_sub_engines/*.engine``.
    """
    model_dir = model_weights_dir()
    if not model_dir.is_dir():
        return []
    found = list(model_dir.glob("*.engine"))
    for candidate in model_dir.glob("*_sub_engines"):
        if candidate.is_dir():
            found.extend(candidate.glob("*.engine"))
    return sorted(found)


def legacy_engines_adoption_enabled() -> bool:
    return os.environ.get(_ADOPT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def adopt_legacy_engines(*, reference_path: Path | None = None, note: str = "") -> int:
    """Move engines from a pre-archive install into this GPU's archive.

    A TensorRT plan only deserializes on the card it was built for, and the old
    flat layout carries no record of which card produced it. Adopting those
    files blindly therefore poisons this card's archive with engines that fail
    to load later (reported as a corrupt checkpoint), so adoption is now
    opt-in: set ``JASNA_ADOPT_LEGACY_ENGINES=1`` when you know the flat engines
    were built on the card that is installed now. By default they are left
    alone and the card compiles its own set.

    Returns the number of engine files adopted.
    """
    import shutil

    token = device_identity_token()
    if not token:
        return 0
    model_dir = model_weights_dir()
    if not model_dir.is_dir():
        return 0

    files = legacy_engine_files()
    if not files:
        return 0
    if not legacy_engines_adoption_enabled():
        log.info(
            "Ignoring %d engine file(s) from a pre-archive install in %s: a TensorRT "
            "engine is only valid on the card it was built for, so this GPU compiles "
            "its own. Set %s=1 to reuse them anyway.",
            len(files),
            model_dir,
            _ADOPT_ENV,
        )
        return 0

    # A ``.engine`` file directly in model_weights/ was the old per-build
    # location for detection/unet engines.
    adopted = 0
    for kind in ("detection", "yolo", "unet4x", "onnx"):
        archive = typed_engine_dir(kind, create=True)
        if archive is None:
            continue
        for engine in model_dir.glob("*.engine"):
            target = archive / engine.name
            if target.exists():
                continue
            try:
                shutil.move(str(engine), str(target))
                adopted += 1
            except OSError:
                continue

    # BasicVSR++ sub-engines lived in a sibling ``<model>_sub_engines`` folder.
    for candidate in model_dir.glob("*_sub_engines"):
        if not candidate.is_dir():
            continue
        engines = list(candidate.glob("*.engine"))
        if not engines:
            continue
        archive = typed_engine_dir("basicvsrpp", create=True)
        if archive is None:
            continue
        for engine in engines:
            target = archive / engine.name
            if target.exists():
                continue
            try:
                shutil.move(str(engine), str(target))
                adopted += 1
            except OSError:
                continue
        try:
            if not any(candidate.iterdir()):
                candidate.rmdir()
        except OSError:
            pass

    if adopted:
        write_archive_identity(note or "basicvsrpp", typed_engine_dir("basicvsrpp", create=True) or model_dir)
        log.info("Adopted %d existing engine file(s) into the %s archive", adopted, token)
    return adopted


def engine_precision_name(*, fp16: bool) -> str:
    return "fp16" if bool(fp16) else "fp32"


def _engine_output_path(onnx_path: Path, suffix: str, kind: str) -> Path:
    """Where an engine built from ``onnx_path`` should live, honouring archiving.

    The per-GPU directory keeps the original stem so the file stays recognisable,
    and two different GPUs produce two different directories rather than two
    differently-suffixed files in one folder.
    """
    archive = typed_engine_dir(kind, create=True)
    if archive is None:
        return onnx_path.with_suffix(suffix)
    return archive / (onnx_path.stem + suffix)


def get_onnx_tensorrt_engine_path(
    onnx_path: str | Path,
    *,
    batch_size: int | None = None,
    fp16: bool = True,
    dynamic_batch: bool = False,
    kind: str = "onnx",
) -> Path:
    onnx_path = Path(onnx_path)
    suffix = ""
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        suffix += f".bs1-{batch_size}" if dynamic_batch else f".bs{batch_size}"
    suffix += ".fp16" if bool(fp16) else ""
    suffix += engine_system_suffix()
    suffix += ".engine"
    return _engine_output_path(onnx_path, suffix, kind)


def get_yolo_tensorrt_engine_path(model_path: str | Path, *, fp16: bool) -> Path:
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt YOLO checkpoint, got: {model_path}")
    onnx_path = model_path.with_suffix(".onnx")
    return get_onnx_tensorrt_engine_path(
        onnx_path, batch_size=None, fp16=bool(fp16), kind="yolo"
    )


def get_detection_engine_path(
    onnx_path: str | Path,
    *,
    batch_size: int | None = None,
    fp16: bool = True,
    dynamic_batch: bool = False,
) -> Path:
    """Engine path for a detection model that is exported to ONNX (RF-DETR).

    Lives in the ``detection`` archive so detection engines for different cards
    stay together and never collide with a restoration or super-resolution
    engine built from the same ONNX stem. Every detection consumer must go
    through this helper — calling :func:`get_onnx_tensorrt_engine_path`
    directly would look in the ``onnx`` archive and miss the file.
    """
    return get_onnx_tensorrt_engine_path(
        onnx_path,
        batch_size=batch_size,
        fp16=fp16,
        dynamic_batch=dynamic_batch,
        kind="detection",
    )


UNET4X_ONNX_PATH = model_weights_dir() / "unet-4x.onnx"
UNET4X_ONNX_ENC_PATH = model_weights_dir() / "unet-4x.onnx.enc"

SD15_HF_REPO = "Kruk2/sd-15-jav"
SD15_DIR = model_weights_dir() / "sd-15-jav"
SD15_CKPT_PATH = SD15_DIR / "sd15-200000.ckpt"
SD15_CKPT_ENC_PATH = SD15_DIR / "sd15-200000.ckpt.enc"


def get_unet4x_engine_path(onnx_path: str | Path | None = None, fp16: bool = True) -> Path:
    if onnx_path is None:
        onnx_path = UNET4X_ONNX_PATH
    return get_onnx_tensorrt_engine_path(onnx_path, batch_size=None, fp16=fp16, kind="unet4x")


def get_unet4x_encrypted_engine_path(fp16: bool = True) -> Path:
    plain = get_unet4x_engine_path(UNET4X_ONNX_PATH, fp16=fp16)
    return plain.parent / (plain.name + ".enc")


def unet4x_plaintext_available() -> bool:
    return UNET4X_ONNX_PATH.exists() and not is_frozen()


def expected_unet4x_engine_path(fp16: bool = True) -> Path:
    """The unet-4x engine file that should exist on disk for this build: the plaintext
    engine in source/dev, the encrypted engine in frozen builds. Single source of truth
    for "is unet-4x already compiled?" used by both the compiler and the GUI preflight."""
    if unet4x_plaintext_available():
        return get_unet4x_engine_path(fp16=fp16)
    return get_unet4x_encrypted_engine_path(fp16=fp16)


BASICVSRPP_DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")

# The preprocess and upsample stages run in fixed-size batches instead of one
# call per clip. TensorRT sizes an engine's scratch for its largest profile
# shape and that scratch scales linearly with the batch, so a clip-sized engine
# reserves gigabytes the batched one does not: at 180 frames, b180 upsample
# costs 3.2 GB more than b30 (identical speed) and b180 preprocess 0.8 GB more
# than b60 (+2 ms per clip). Both are also independent of the clip length, so
# changing the max clip size no longer recompiles any engine.
BASICVSRPP_UPSAMPLE_BATCH = 30
BASICVSRPP_PREPROCESS_BATCH = 60


def basicvsrpp_sub_engine_dir(model_weights_path: str, *, create: bool = False) -> str:
    """Per-GPU directory holding the BasicVSR++ sub-engines.

    Falls back to the historic sibling directory (``<model>_sub_engines``) when
    the archive is disabled.
    """
    stem = os.path.splitext(os.path.basename(model_weights_path))[0]
    archive = typed_engine_dir("basicvsrpp", create=create)
    if archive is not None:
        return str(archive)
    return os.path.join(os.path.dirname(model_weights_path), f"{stem}_sub_engines")


def get_basicvsrpp_sub_engine_paths(model_weights_path: str, fp16: bool) -> dict[str, str]:
    engine_dir = basicvsrpp_sub_engine_dir(model_weights_path)
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    paths: dict[str, str] = {}
    for d in BASICVSRPP_DIRECTIONS:
        paths[f"loop_body_{d}"] = os.path.join(engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine")
    paths["preprocess"] = os.path.join(
        engine_dir, f"preprocess_b{BASICVSRPP_PREPROCESS_BATCH}.trt_{prec}{suf}.engine"
    )
    paths["upsample"] = os.path.join(
        engine_dir, f"upsample_dyn_b{BASICVSRPP_UPSAMPLE_BATCH}.trt_{prec}{suf}.engine"
    )
    return paths


def all_basicvsrpp_sub_engines_exist(model_weights_path: str, fp16: bool) -> bool:
    return all(os.path.isfile(p) for p in get_basicvsrpp_sub_engine_paths(model_weights_path, fp16).values())
