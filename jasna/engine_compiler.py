"""Compile TensorRT engines in a subprocess to guarantee full VRAM release."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

from jasna._frozen import is_frozen
from jasna.engine_paths import (
    all_basicvsrpp_sub_engines_exist,
    set_engine_device_index,
    write_archive_identity,
)
from jasna.os_utils import subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30 * 60


def _engine_device_index(device: str) -> int:
    """CUDA index of the device an engine is being compiled for."""
    text = str(device).strip()
    if text in {"", "cuda"}:
        return 0
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1]
    try:
        return int(text)
    except ValueError:
        return 0


@dataclass
class EngineCompilationRequest:
    device: str
    fp16: bool

    basicvsrpp: bool = False
    basicvsrpp_model_path: str = ""

    detection: bool = False
    detection_model_name: str = ""
    detection_model_path: str = ""
    detection_batch_size: int = 4

    unet4x: bool = False

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> EngineCompilationRequest:
        return EngineCompilationRequest(**json.loads(s))


@dataclass
class EngineCompilationResult:
    use_basicvsrpp_tensorrt: bool = False


def _detection_engine_exists(
    detection_model_name: str,
    detection_model_path: str,
    batch_size: int,
    fp16: bool,
    device: str,
) -> bool:
    import torch

    from jasna.accelerator import is_amd_device
    from jasna.arch_compat import tensorrt_supports_device
    from jasna.mosaic.detection_registry import (
        is_rfdetr_model,
        is_yolo_model,
        rfdetr_model_config,
    )

    resolved_device = torch.device(device)
    if is_amd_device(resolved_device) or not tensorrt_supports_device(resolved_device):
        # AMD runs both RF-DETR (rfdetr torch model) and YOLO through PyTorch. So
        # does an NVIDIA card the installed TensorRT cannot target — Pascal is below
        # the 10.x floor of 7.5. Either way there is no compiled engine artifact to
        # check for, and asking for one would trigger a compile that cannot succeed.
        return True

    from jasna.engine_paths import (
        get_detection_engine_path,
        get_yolo_tensorrt_engine_path,
    )

    if is_rfdetr_model(detection_model_name):
        config = rfdetr_model_config(detection_model_name)
        return get_detection_engine_path(
            detection_model_path,
            batch_size=config.engine_batch_size(batch_size),
            fp16=fp16,
            dynamic_batch=config.dynamic_batch,
        ).exists()
    if is_yolo_model(detection_model_name):
        return get_yolo_tensorrt_engine_path(detection_model_path, fp16=fp16).exists()
    return True


def _unet4x_engine_exists(fp16: bool) -> bool:
    from jasna.engine_paths import (
        expected_unet4x_engine_path,
        get_unet4x_encrypted_engine_path,
        unet4x_plaintext_available,
    )

    if unet4x_plaintext_available():
        return expected_unet4x_engine_path(fp16=fp16).exists()

    engine_path = get_unet4x_encrypted_engine_path(fp16=fp16)
    if not engine_path.exists():
        return False

    from jasna._licensing import protection_error_types
    from jasna._licensing import protection_module

    module = protection_module()
    if module is None:
        return False
    try:
        module.protected_model.decrypt_engine_bytes("unet-4x", engine_path.read_bytes())
    except protection_error_types():
        return False
    return True


class _MissingEngines(typing.NamedTuple):
    basicvsrpp: bool
    detection: bool
    unet4x: bool


def _missing_engines(req: EngineCompilationRequest, *, trt: bool) -> _MissingEngines:
    """Which engines still have to be built.

    ``trt`` means "the installed TensorRT can target this card", not merely "the card
    is NVIDIA": on Pascal TensorRT has no path at all, so nothing is missing — the
    pipeline runs the PyTorch models instead, and a compile could only fail.
    """
    return _MissingEngines(
        basicvsrpp=trt and req.basicvsrpp and req.fp16
        and not all_basicvsrpp_sub_engines_exist(req.basicvsrpp_model_path, req.fp16),
        detection=req.detection and not _detection_engine_exists(
            req.detection_model_name,
            req.detection_model_path,
            req.detection_batch_size,
            req.fp16,
            req.device,
        ),
        unet4x=trt and req.unet4x and not _unet4x_engine_exists(req.fp16),
    )


def ensure_engines_compiled(
    req: EngineCompilationRequest,
    log_callback: typing.Callable[[str], None] | None = None,
) -> EngineCompilationResult:
    import torch

    from jasna.accelerator import is_nvidia_device
    from jasna.arch_compat import tensorrt_supports_device

    result = EngineCompilationResult()
    device = torch.device(req.device)
    nvidia = is_nvidia_device(device)
    # TensorRT has to be able to target the card before it is worth trying: on Pascal
    # (below the 10.x floor of 7.5) there is no engine path, so the run uses the
    # PyTorch restorer and detector without a compile attempt that cannot succeed.
    trt = nvidia and tensorrt_supports_device(device)
    if nvidia and not trt:
        logger.info(
            "TensorRT cannot target this GPU (compute capability %s); running the PyTorch "
            "models instead of compiling engines",
            ".".join(str(part) for part in torch.cuda.get_device_capability(device)),
        )

    if req.unet4x and not nvidia:
        raise RuntimeError("unet-4x currently requires the NVIDIA TensorRT build")

    missing = _missing_engines(req, trt=trt)
    if missing.unet4x:
        from jasna._licensing import license_store
        from jasna.engine_paths import unet4x_plaintext_available
        if not unet4x_plaintext_available() and not license_store().is_licensed():
            raise RuntimeError("unet-4x is a supporter feature. Enter your license to enable it.")

    if req.basicvsrpp and trt:
        if not req.fp16:
            result.use_basicvsrpp_tensorrt = False
        elif not missing.basicvsrpp:
            result.use_basicvsrpp_tensorrt = True

    if not any(missing):
        return result

    logger.info("Spawning GPU model compilation subprocess...")
    start_msg = "Compiling TensorRT engines (this may take several minutes)..."
    # The frozen GUI drops its console (FreeConsole), leaving stdout invalid — an
    # unconditional print() there raises WinError 6. Print only on the CLI (no callback).
    if log_callback:
        log_callback(start_msg)
    else:
        print(start_msg)

    if is_frozen():
        cmd = [sys.executable, "--compile-engines", req.to_json()]
    else:
        cmd = [sys.executable, "-m", "jasna.engine_compiler", req.to_json()]

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,  # don't inherit the GUI's detached (invalid) stdin
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    proc = subprocess.Popen(cmd, **kwargs, **subprocess_no_window_kwargs())
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        if line:
            if log_callback:
                log_callback(line)
                logger.debug("[compiler] %s", line)
            else:
                print(line)
                logger.info("[compiler] %s", line)
    returncode = proc.wait(timeout=_TIMEOUT_SECONDS)

    if returncode != 0:
        raise RuntimeError(f"Engine compilation subprocess failed (exit code {returncode})")

    if req.basicvsrpp and nvidia:
        result.use_basicvsrpp_tensorrt = all_basicvsrpp_sub_engines_exist(
            req.basicvsrpp_model_path, req.fp16
        )

    return result


def _subprocess_compile(req: EngineCompilationRequest) -> None:
    import logging as _logging
    import warnings
    warnings.filterwarnings("ignore")
    _logging.disable(_logging.WARNING)

    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    import torch
    from jasna.accelerator import is_nvidia_device

    # The compile subprocess imports torch_tensorrt (-> torch._inductor) directly, without
    # going through jasna.pipeline, so the source-introspection shims aren't installed yet.
    # In the compiled (Nuitka) binary that introspection raises; patch before any such import.
    from jasna._frozen import patch_frozen_torch
    patch_frozen_torch()
    from jasna.accelerator import configure_rocm_process_env
    configure_rocm_process_env()

    device = torch.device(req.device)
    # Pin the per-GPU engine archive before any path is resolved, then decide
    # what is missing: this makes a compile targeted at device N land in device
    # N's archive and never satisfy itself from another card's engines.
    set_engine_device_index(_engine_device_index(req.device))
    from jasna.arch_compat import tensorrt_supports_device

    missing = _missing_engines(req, trt=is_nvidia_device(device) and tensorrt_supports_device(device))

    if missing.basicvsrpp:
        from jasna.restorer.basicvsrpp_sub_engines import compile_basicvsrpp_engines
        print("Compiling BasicVSR++ sub-engines...")
        compile_basicvsrpp_engines(req.basicvsrpp_model_path, device, req.fp16)
        print("BasicVSR++ sub-engines compiled.")

    if missing.detection:
        from jasna.mosaic.detection_registry import precompile_detection_engine
        print(f"Compiling detection engine ({req.detection_model_name})...")
        precompile_detection_engine(
            detection_model_name=req.detection_model_name,
            detection_model_path=Path(req.detection_model_path),
            batch_size=req.detection_batch_size,
            device=device,
            fp16=req.fp16,
        )
        print("Detection engine compiled.")

    if missing.unet4x:
        from jasna.restorer.unet4x_secondary_restorer import compile_unet4x_engine
        print("Compiling Unet4x engine...")
        compile_unet4x_engine(device, fp16=req.fp16)
        print("Unet4x engine compiled.")

    _stamp_engine_archives()


def _stamp_engine_archives() -> None:
    """Write the owning GPU's description into every archive touched by a build.

    The stamp is what lets a later launch tell "this archive belongs to another
    card" apart from "this archive is incomplete", and it is what the GUI shows
    when it reports which GPUs already have engines ready.
    """
    from jasna.engine_paths import typed_engine_dir

    for kind in ("basicvsrpp", "detection", "yolo", "unet4x", "onnx"):
        directory = typed_engine_dir(kind)
        if directory is not None and directory.is_dir():
            write_archive_identity(kind, directory)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m jasna.engine_compiler <json_request>", file=sys.stderr)
        sys.exit(1)
    req = EngineCompilationRequest.from_json(sys.argv[1])
    _subprocess_compile(req)
