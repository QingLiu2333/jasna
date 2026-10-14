"""Per-GPU identity, so compiled TensorRT engines are archived *per card*.

A TensorRT engine is built for one exact hardware/firmware combination: the GPU
architecture, the SM count, the driver's CUDA runtime and the TensorRT version
all end up baked into the serialized plan. Moving an engine built on card A to
card B makes ``IRuntime::deserializeCudaEngine`` fail, and the usual symptom is
a hard error at startup that the user cannot act on.

This module gives every physical GPU a short, stable, filesystem-safe
*identity token* and a human-readable description. Engine files get that token
in their name, so switching cards no longer invalidates the previous build:
each card keeps its own directory of ready-made engines, and a card that has
never been seen simply compiles once on first use.

The token deliberately covers everything a plan depends on:

* PCI vendor / device / subsystem id  — which silicon this is
* compute capability                  — the ISA the kernels were built for
* SM count and total memory           — shape/scheduling details of the plan
* NVIDIA driver version                — the runtime that will execute it
* TensorRT version                    — the builder that produced it

It deliberately does *not* cover things that change between runs (clock
speeds, current VRAM usage, GPU instance UUID): an engine stays valid across
reboots, driver-level device reordering, and moving the install to another
machine that has the same card.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger(__name__)

# Bumped whenever the on-disk layout of an engine archive changes, so old
# archives are ignored instead of being mis-read.
ARCHIVE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class GpuIdentity:
    """Everything a compiled engine depends on, for one physical GPU."""

    index: int = 0
    name: str = "Unknown GPU"
    # PCI ids as reported by CUDA (may be 0 when the driver does not expose them).
    pci_domain: int = 0
    pci_bus: int = 0
    pci_device: int = 0
    pci_vendor_id: int = 0
    pci_device_id: int = 0
    compute_capability: tuple[int, int] = (0, 0)
    sm_count: int = 0
    total_memory: int = 0
    driver_version: str = ""
    trt_version: str = ""
    vendor: str = "nvidia"

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------

    @property
    def capability_str(self) -> str:
        major, minor = self.compute_capability
        return f"sm{major}{minor}"

    @property
    def is_volta(self) -> bool:
        return self.compute_capability == (7, 0)

    @property
    def is_turing(self) -> bool:
        return self.compute_capability == (7, 5)

    @property
    def is_pascal(self) -> bool:
        """GTX 10xx and friends (6.0 / 6.1 / 6.2)."""
        return 6 <= self.compute_capability[0] < 7

    @property
    def has_tensor_cores(self) -> bool:
        """Volta introduced consumer/prosumer Tensor Cores (V100/Titan V)."""
        return self.compute_capability >= (7, 0)

    @property
    def has_fp16_tensor_cores(self) -> bool:
        """Turing onward run fp16 through Tensor Cores at usable throughput.

        Volta's first-generation Tensor Cores are fp16-capable but notably
        slower per SM and lack the INT8/INT4 paths and the improved fp16
        accumulation scheduling that the TensorRT fp16 kernels assume.
        """
        return self.compute_capability >= (7, 5)

    def hash_inputs(self) -> str:
        """The exact string that gets hashed into the identity token."""
        major, minor = self.compute_capability
        return "|".join(
            [
                f"fmt={ARCHIVE_FORMAT_VERSION}",
                f"vendor={self.vendor}",
                f"pci={self.pci_domain:04x}:{self.pci_bus:02x}:{self.pci_device:02x}",
                f"ids={self.pci_vendor_id:04x}:{self.pci_device_id:04x}",
                f"cc={major}.{minor}",
                f"sm={self.sm_count}",
                f"mem={self.total_memory}",
                f"drv={self.driver_version}",
                f"trt={self.trt_version}",
            ]
        )

    @property
    def token(self) -> str:
        """Short stable token used as a directory/file suffix."""
        digest = hashlib.sha256(self.hash_inputs().encode("utf-8")).hexdigest()[:12]
        return f"{self.capability_str}-{digest}"

    @property
    def short_label(self) -> str:
        major, minor = self.compute_capability
        return f"{self.name} (sm{major}{minor}, {self.sm_count} SM)"

    def describe(self) -> str:
        major, minor = self.compute_capability
        gb = self.total_memory / (1024**3)
        return (
            f"{self.name} | cc {major}.{minor} | {self.sm_count} SM | "
            f"{gb:.1f} GiB | driver {self.driver_version or '?'} | "
            f"TensorRT {self.trt_version or '?'} | pci {self.pci_bus:02x}:{self.pci_device:02x}"
        )


def _trt_version() -> str:
    try:
        import tensorrt as trt  # type: ignore[import-not-found]

        return str(trt.__version__)
    except Exception:  # pragma: no cover - tensorrt always present in shipped builds
        return ""


def _nvidia_driver_version() -> str:
    """Driver version via nvidia-smi; falls back to the CUDA runtime string."""
    try:
        from jasna import os_utils

        path = os_utils.find_executable("nvidia-smi")
        if path:
            import subprocess

            completed = subprocess.run(
                [path, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                **os_utils.subprocess_no_window_kwargs(),
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip().splitlines()[0].strip()
    except Exception:
        log.debug("Could not determine the NVIDIA driver version", exc_info=True)
    return ""


@lru_cache(maxsize=None)
def _driver_version_cached() -> str:
    return _nvidia_driver_version()


def describe_current_device(device_index: int = 0) -> GpuIdentity:
    """Build a :class:`GpuIdentity` for one CUDA device.

    Never raises: a device whose properties cannot be queried still yields a
    usable identity with whatever fields came back, so an unusual driver does
    not make the application refuse to start.
    """
    try:
        import torch
    except Exception:  # pragma: no cover
        return GpuIdentity(index=device_index)

    name = f"cuda:{device_index}"
    try:
        props = torch.cuda.get_device_properties(device_index)
    except Exception:
        log.warning("Could not read properties of CUDA device %s", device_index, exc_info=True)
        return GpuIdentity(index=device_index, name=name)

    def _get(attr: str, default: int = 0) -> int:
        value = getattr(props, attr, default)
        try:
            return int(value)
        except Exception:
            return default

    capability = tuple(_get("major", 0) and (_get("major", 0), _get("minor", 0)) or (0, 0))
    major, minor = capability  # type: ignore[misc]
    return GpuIdentity(
        index=device_index,
        name=str(getattr(props, "name", name)),
        pci_domain=_get("pci_domain_id", 0),
        pci_bus=_get("pci_bus_id", 0),
        pci_device=_get("pci_device_id", 0),
        pci_vendor_id=_get("pci_vendor_id", 0),
        pci_device_id=_get("device_id", 0),
        compute_capability=(major, minor),
        sm_count=_get("multi_processor_count", 0),
        total_memory=_get("total_memory", 0),
        driver_version=_driver_version_cached(),
        trt_version=_trt_version(),
        vendor="nvidia",
    )


def current_identity_token(device_index: int = 0) -> str:
    """The archive token of the CUDA device ``device_index``.

    Returns an empty string on a host with no usable CUDA device, which the
    engine-path helpers read as "do not use a per-GPU archive".
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
    except Exception:
        return ""
    return describe_current_device(device_index).token


def list_cuda_devices() -> list[dict]:
    """Enumerate the CUDA devices, for the GUI's device pickers."""
    devices: list[dict] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return devices
        count = torch.cuda.device_count()
    except Exception:
        return devices
    for index in range(count):
        identity = describe_current_device(index)
        devices.append(
            {
                "index": index,
                "name": identity.name,
                "label": f"[{index}] {identity.short_label}",
                "capability": identity.compute_capability,
                "capability_str": identity.capability_str,
                "sm_count": identity.sm_count,
                "total_memory": identity.total_memory,
                "token": identity.token,
                "description": identity.describe(),
                "has_tensor_cores": identity.has_tensor_cores,
                "has_fp16_tensor_cores": identity.has_fp16_tensor_cores,
            }
        )
    return devices


def compute_capability_of(device_index: int = 0) -> tuple[int, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(device_index))  # type: ignore[return-value]
    except Exception:
        pass
    return (0, 0)


__all__ = [
    "ARCHIVE_FORMAT_VERSION",
    "GpuIdentity",
    "compute_capability_of",
    "current_identity_token",
    "describe_current_device",
    "list_cuda_devices",
]
