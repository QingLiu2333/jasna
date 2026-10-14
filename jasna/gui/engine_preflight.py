from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jasna.gui.models import AppSettings


@dataclass(frozen=True)
class EngineRequirement:
    key: str
    label: str
    paths: tuple[Path, ...]
    exists: bool
    missing_paths: tuple[Path, ...]


@dataclass(frozen=True)
class EnginePreflightResult:
    requirements: tuple[EngineRequirement, ...]
    should_warn_first_run_slow: bool
    # Which GPU this launch will compile for, and whether engines already exist
    # for other cards. Empty when the information could not be read.
    device_summary: str = ""
    archive_token: str = ""
    other_gpu_archives: tuple[str, ...] = ()

    @property
    def missing(self) -> tuple[EngineRequirement, ...]:
        return tuple(r for r in self.requirements if not r.exists)

    @property
    def switches_gpu(self) -> bool:
        """True when engines exist for other cards but none for this one yet.

        That is the "you changed the physical card" case: the job will spend a
        while compiling instead of silently reusing an incompatible engine.
        """
        return bool(self.other_gpu_archives) and any(not r.exists for r in self.requirements)


def _detection_weights_path(settings: AppSettings) -> Path:
    from jasna.mosaic.detection_registry import coerce_detection_model_name, detection_model_weights_path

    return detection_model_weights_path(coerce_detection_model_name(str(settings.detection_model)))


def run_engine_preflight(settings: AppSettings) -> EnginePreflightResult:
    import torch

    from jasna.accelerator import is_amd_device
    from jasna.engine_paths import (
        default_restoration_model_path,
        expected_unet4x_engine_path,
        get_basicvsrpp_sub_engine_paths,
        get_detection_engine_path,
        get_yolo_tensorrt_engine_path,
    )
    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        is_rfdetr_model,
        is_yolo_model,
        rfdetr_model_config,
    )

    reqs: list[EngineRequirement] = []
    from jasna import os_utils

    # The selected GPU, not index 0: the preflight builds/validates the archive
    # the job will actually load from.
    device = os_utils.compute_device()
    amd = is_amd_device(device)

    # A card this TensorRT cannot target at all (Pascal and other pre-7.5 parts
    # on the CUDA 13 build, or any card on a tree with no TensorRT) runs every
    # model through its PyTorch path. No engine will ever be compiled, so none
    # may be reported as "missing" - the first-run warning listing engines that
    # cannot exist was pure noise on a GTX 1060.
    from jasna.arch_compat import tensorrt_supports_device

    tensorrt_possible = (not amd) and tensorrt_supports_device(device)

    det_name = coerce_detection_model_name(str(settings.detection_model))
    det_weights = _detection_weights_path(settings)
    if is_rfdetr_model(det_name):
        if amd or not tensorrt_possible:
            # AMD runs RF-DETR through the rfdetr torch model; a non-TRT NVIDIA
            # card cannot run it at all, so there is nothing to build either way.
            det_engine = None
            det_exists = True
        else:
            config = rfdetr_model_config(det_name)
            det_engine = get_detection_engine_path(
                det_weights,
                batch_size=config.engine_batch_size(settings.batch_size),
                fp16=bool(settings.fp16_mode),
                dynamic_batch=config.dynamic_batch,
            )
            det_exists = det_engine.is_file()
        if det_engine is not None:
            reqs.append(
                EngineRequirement(
                    key="rfdetr",
                    label=f"RF-DETR ({det_weights.name})",
                    paths=(det_engine,),
                    exists=det_exists,
                    missing_paths=() if det_exists else (det_engine,),
                )
            )
    elif is_yolo_model(det_name) and not amd and tensorrt_possible:
        det_engine = get_yolo_tensorrt_engine_path(det_weights, fp16=bool(settings.fp16_mode))
        det_exists = det_engine.is_file()
        reqs.append(
            EngineRequirement(
                key="yolo",
                label=f"YOLO ({det_weights.name})",
                paths=(det_engine,),
                exists=det_exists,
                missing_paths=() if det_exists else (det_engine,),
            )
        )

    standard_model = settings.restoration_model == "basicvsrpp"
    restoration_model_path = default_restoration_model_path("basicvsrpp")
    if standard_model and bool(settings.compile_basicvsrpp) and not amd and tensorrt_possible:
        sub_paths = get_basicvsrpp_sub_engine_paths(str(restoration_model_path), bool(settings.fp16_mode))
        all_engine_paths = tuple(Path(p) for p in sub_paths.values())
        missing_paths = tuple(p for p in all_engine_paths if not p.is_file())
        reqs.append(
            EngineRequirement(
                key="basicvsrpp",
                label="BasicVSR++ (restoration sub-engines)",
                paths=all_engine_paths,
                exists=len(missing_paths) == 0,
                missing_paths=missing_paths,
            )
        )

    if standard_model and settings.secondary_restoration == "unet-4x" and tensorrt_possible:
        unet_engine = expected_unet4x_engine_path(fp16=bool(settings.fp16_mode))
        unet_exists = unet_engine.is_file()
        reqs.append(
            EngineRequirement(
                key="unet_4x",
                label="UNet 4x (secondary restoration)",
                paths=(unet_engine,),
                exists=unet_exists,
                missing_paths=() if unet_exists else (unet_engine,),
            )
        )

    should_warn = any(not r.exists for r in reqs)

    device_summary = ""
    archive_token = ""
    other_archives: tuple[str, ...] = ()
    try:
        from jasna.engine_paths import device_identity, device_identity_token, other_archives as _others

        identity = device_identity()
        if identity is not None:
            device_summary = identity.describe()
            archive_token = identity.token
            others: list[str] = []
            for kind in ("basicvsrpp", "detection", "yolo", "unet4x"):
                for entry in _others(kind):
                    if entry["engine_count"] and entry["description"] not in others:
                        others.append(entry["description"])
            other_archives = tuple(others)
    except Exception:
        pass

    return EnginePreflightResult(
        requirements=tuple(reqs),
        should_warn_first_run_slow=bool(should_warn),
        device_summary=device_summary,
        archive_token=archive_token,
        other_gpu_archives=other_archives,
    )
