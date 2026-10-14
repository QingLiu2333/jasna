"""Run the restoration models on GPU generations the build was not tuned for.

Upstream Jasna gates on compute capability 7.5 (Turing) and assumes two things
about the card underneath:

* **fp16 needs Tensor Cores.** Turing and later run fp16 through Tensor Cores at
  a large throughput advantage over fp32. Volta (7.0) *has* Tensor Cores — the
  first generation — and runs fp16 fine, just at a much smaller multiple.
  Pascal (6.x, GTX 10xx) has none, and fp16 there is emulated, i.e. slower than
  fp32 rather than faster.
* **The quantized LTX paths need sm_89+.** The NVFP4 (W4A4) and FP8 block-scaled
  GEMMs go through ``torch._scaled_mm`` with FP8/FP4 operands, which wants Ada's
  sm_89, Hopper's sm_90 or Blackwell. On anything older PyTorch raises
  ``NotImplementedError`` the moment the first quantized layer runs.

This module answers "what should this card actually run?" and provides
architecture-neutral replacements for the two pieces that assume otherwise:

* :func:`recommended_fp16` — whether fp16 is worth using on this card,
* :func:`supports_block_quantized_gemm` — whether the fast FP4/FP8 GEMMs exist,
* :func:`dequantized_nvfp4_linear` — an NVFP4 W4A4 layer evaluated by unpacking
  the weights and running bf16 matmuls, so an FP4 LTX model still *loads and
  runs* on Volta/Turing/Ampere instead of raising,
* :func:`dequantized_int8_linear` — the same idea for the ConvRot W8A8 layers,
  for cards where even ``torch._int_mm`` is unavailable.

The dequantized paths are functionally correct but several times slower and use
more memory than the tensor-core kernels, so they are a compatibility floor, not
a recommendation: the user is told which one is in effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch

log = logging.getLogger(__name__)

# Compute capability thresholds used across the codebase.
VOLTA = (7, 0)
TURING = (7, 5)
AMPERE = (8, 0)
ADA = (8, 9)  # first consumer architecture with FP8 tensor cores
HOPPER = (9, 0)

# Architecture above which the block-scaled FP4/FP8 GEMMs exist in PyTorch's
# ``_scaled_mm`` (Ada/Hopper/Blackwell, or an MI300-class AMD part).
_SCALED_MM_ENV = "JASNA_ALLOW_BLOCK_QUANTIZED_GEMM"


def _cuda_capability() -> tuple[int, int]:
    try:
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]
    except Exception:
        pass
    return (0, 0)


@lru_cache(maxsize=1)
def device_capability() -> tuple[int, int]:
    return _cuda_capability()


def reset_capability_cache() -> None:
    """Forget the cached capability (tests, or after a device change)."""
    device_capability.cache_clear()
    supports_block_quantized_gemm.cache_clear()
    supports_tensor_cores.cache_clear()
    supports_int8_mm.cache_clear()
    tensorrt_version.cache_clear()


def _is_amd() -> bool:
    try:
        return bool(torch.version.hip)
    except Exception:
        return False


@lru_cache(maxsize=1)
def supports_block_quantized_gemm() -> bool:
    """Whether ``torch._scaled_mm`` accepts FP8 / block-scaled FP4 operands here.

    PyTorch's own gate is "compute capability >= 9.0 or 8.9, or ROCm MI300+".
    We ask it rather than duplicating the rule, by attempting a tiny FP8 GEMM
    once and caching the answer: a driver that advertises an architecture but
    lacks the kernel is then handled correctly too.
    """
    import os

    override = os.environ.get(_SCALED_MM_ENV)
    if override is not None:
        enabled = override.strip().lower() not in {"0", "false", "no", "off"}
        log.info("Block-quantized GEMM forced %s by %s", "on" if enabled else "off", _SCALED_MM_ENV)
        return enabled

    if _is_amd():
        # MI300-class ROCm builds carry FP8 block-scaled GEMMs; let the probe decide.
        pass

    try:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda", torch.cuda.current_device())
        a = torch.zeros(16, 32, device=device, dtype=torch.float8_e4m3fn)
        b = torch.zeros(16, 32, device=device, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, device=device)
        torch.nn.functional.scaled_mm(
            a,
            b.t(),
            [scale],
            [torch.nn.functional.ScalingType.TensorWise],
            [scale],
            [torch.nn.functional.ScalingType.TensorWise],
            output_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize(device)
        return True
    except Exception as exc:
        log.info(
            "Block-quantized FP8/FP4 GEMMs are unavailable on this device (%s); "
            "quantized weights will be dequantized before use",
            exc.__class__.__name__,
        )
        return False


@lru_cache(maxsize=1)
def supports_tensor_cores() -> bool:
    return device_capability() >= VOLTA


@lru_cache(maxsize=1)
def supports_int8_mm() -> bool:
    """``torch._int_mm`` needs sm_75+ (Turing) on CUDA."""
    if _is_amd():
        return True
    return device_capability() >= TURING


def supports_fp8_tensor_cores() -> bool:
    return device_capability() >= ADA or _is_amd()


# --- TensorRT capability ---------------------------------------------------
#
# Which cards TensorRT can build an engine for is decided by the *installed*
# TensorRT, not by the app: the 10.0-10.4 line still targeted Volta (7.0),
# TensorRT 10.5 removed it, the 10.5+ line starts at Turing (7.5), and the 8.6
# line went down to Maxwell. Pascal (6.x) has never been supported by the 10.x
# line at all - TRT has no path for it, not even a slow one.
TRT_VOLTA_DROPPED = (10, 5)
TRT_TURING_FLOOR = (7, 5)
TRT_VOLTA_FLOOR = (7, 0)
TRT_MAXWELL_FLOOR = (5, 3)


@lru_cache(maxsize=1)
def tensorrt_version() -> tuple[int, int] | None:
    """The installed TensorRT's ``(major, minor)``, or None when it is absent."""
    try:
        import tensorrt as trt

        parts = str(trt.__version__).split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        log.debug("TensorRT is not importable", exc_info=True)
        return None


def tensorrt_min_compute() -> tuple[int, int] | None:
    """Lowest compute capability the installed TensorRT can build for."""
    version = tensorrt_version()
    if version is None:
        return None
    major, minor = version
    if major >= 11:
        return TRT_TURING_FLOOR
    if major == 10:
        return TRT_TURING_FLOOR if minor >= TRT_VOLTA_DROPPED[1] else TRT_VOLTA_FLOOR
    return TRT_MAXWELL_FLOOR


def tensorrt_supports(capability: tuple[int, int] | None = None) -> bool:
    """Whether TensorRT can build an engine for this card at all.

    False means "do not try": falling back to the PyTorch models is the only way
    the pipeline runs, and attempting a compile first would waste minutes and
    leave a confusing TensorRT error in the log.
    """
    floor = tensorrt_min_compute()
    if floor is None:
        return False
    resolved = tuple(capability) if capability is not None else device_capability()
    if not resolved or resolved == (0, 0):
        return False
    return resolved >= floor


def tensorrt_supports_device(device=None) -> bool:
    """``tensorrt_supports`` for a torch device, including the vendor check."""
    if _is_amd():
        # ROCm builds have no TensorRT path; RF-DETR runs as a torch model.
        return False
    capability = device_capability()
    if device is not None:
        try:
            capability = tuple(torch.cuda.get_device_capability(device))
        except Exception:
            log.debug("Could not read the capability of %s", device, exc_info=True)
    return tensorrt_supports(capability)


@dataclass(frozen=True)
class ArchProfile:
    """What this card should run, and why."""

    capability: tuple[int, int]
    name: str
    fp16_worthwhile: bool
    fp16_reason: str
    block_quantized_gemm: bool
    int8_gemm: bool
    tensor_cores: bool

    @property
    def capability_str(self) -> str:
        major, minor = self.capability
        return f"sm{major}{minor}"

    def summary(self) -> str:
        parts = [f"{self.name} ({self.capability_str})"]
        parts.append("fp16" if self.fp16_worthwhile else "fp32/fp16-emulated")
        parts.append("int8-tc" if self.int8_gemm else "int8-emulated")
        parts.append("fp4/fp8" if self.block_quantized_gemm else "dequantized-lowprecision")
        return " | ".join(parts)


def profile_for(capability: tuple[int, int] | None = None, name: str = "") -> ArchProfile:
    """Describe what the given card (or the live one) should run.

    ``capability`` overrides the live device for planning what *another* card
    would do; the GEMM capability flags then follow the architecture thresholds
    rather than the live probe, since the probe can only speak for the card that
    is actually installed.
    """
    simulated = capability is not None
    capability = tuple(capability) if capability is not None else device_capability()
    if not name:
        try:
            name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        except Exception:
            name = "Unknown GPU"

    if capability >= ADA:
        fp16_ok, reason = True, "native fp16 tensor cores"
    elif capability >= TURING:
        fp16_ok, reason = True, "Turing fp16 tensor cores"
    elif capability >= VOLTA:
        fp16_ok, reason = True, "Volta first-generation tensor cores (small speedup)"
    elif capability >= (6, 0):
        fp16_ok, reason = False, "no tensor cores (Pascal); fp16 is emulated and slower than fp32"
    else:
        fp16_ok, reason = False, "no tensor cores and no usable fp16 path"

    if simulated:
        block_quantized = capability >= ADA or _is_amd()
        int8_gemm = _is_amd() or capability >= TURING
    else:
        block_quantized = supports_block_quantized_gemm()
        int8_gemm = supports_int8_mm()

    return ArchProfile(
        capability=capability,
        name=name,
        fp16_worthwhile=fp16_ok,
        fp16_reason=reason,
        block_quantized_gemm=block_quantized,
        int8_gemm=int8_gemm,
        tensor_cores=capability >= VOLTA,
    )


def recommended_fp16(requested: bool, capability: tuple[int, int] | None = None) -> tuple[bool, str]:
    """Whether to honour an fp16 request on this card.

    Returns ``(use_fp16, note)``; ``note`` is empty when nothing was changed.
    """
    capability = tuple(capability) if capability is not None else device_capability()
    if not requested or not capability:
        return requested, ""
    if capability >= VOLTA:
        if capability < TURING:
            return True, (
                "Volta has first-generation Tensor Cores: fp16 works but the speedup "
                "over fp32 is much smaller than on Turing and later cards"
            )
        return True, ""
    return False, (
        f"compute capability {capability[0]}.{capability[1]} has no Tensor Cores; "
        "fp16 is emulated and would be slower than fp32, so fp32 is used instead"
    )


# ---------------------------------------------------------------------------
# Dequantized replacements for the tensor-core-only quantized GEMMs
# ---------------------------------------------------------------------------

# E2M1 magnitudes indexed by the 3-bit magnitude field (sign is bit 3).
_E2M1_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP8_E4M3_MAX = 448.0


def _e2m1_lookup_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1_TABLE, dtype=torch.float32, device=device)


# Weights are unpacked once and cached; a 22B model has several thousand of
# them, so re-unpacking per call would dominate the run.
_unpacked_nvfp4_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
_UNPACK_CACHE_LIMIT = 4096


def unpack_nvfp4(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``(rows, cols/2)`` packed E2M1 -> ``(rows, cols)`` float32 magnitudes + sign.

    High nibble first, matching the storage layout the FP4 layers use.
    """
    key = (weight.data_ptr(), weight.shape[0], weight.shape[1], device.index or 0)
    cached = _unpacked_nvfp4_cache.get(key)
    if cached is not None:
        return cached

    packed = weight.view(torch.uint8) if weight.dtype != torch.uint8 else weight
    packed = packed.contiguous()
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    # High nibble holds the first (even) element of each pair.
    interleaved = torch.stack((high, low), dim=-1).reshape(packed.shape[0], -1)

    magnitude = _e2m1_lookup_table(packed.device)[interleaved & 0x07]
    sign = (interleaved >> 3) & 0x01
    values = magnitude * (1.0 - 2.0 * sign.float())

    if len(_unpacked_nvfp4_cache) >= _UNPACK_CACHE_LIMIT:
        _unpacked_nvfp4_cache.clear()
    _unpacked_nvfp4_cache[key] = values
    return values


def clear_dequant_cache() -> None:
    _unpacked_nvfp4_cache.clear()


def _blockwise_scale_expand(
    block_scales: torch.Tensor, rows: int, cols: int, block: int
) -> torch.Tensor:
    """``(rows, cols/block)`` scales -> ``(rows, cols)``, one scale per block value."""
    expanded = block_scales.float().repeat_interleave(block, dim=1)
    if expanded.shape[1] < cols:
        pad = cols - expanded.shape[1]
        expanded = torch.nn.functional.pad(expanded, (0, pad), value=1.0)
    return expanded[:, :cols]


def dequantized_nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    block: int = 16,
) -> torch.Tensor:
    """NVFP4 W4A4 evaluated without FP4 tensor cores.

    The activation is quantized to the same E2M1 grid (so the numeric result is
    the same low-precision product the tensor-core path would compute), then
    both operands are dequantized and multiplied in bf16. Slower and heavier
    than ``F.scaled_mm`` but available on every architecture.
    """
    out_features = weight.shape[0]
    in_features = weight.shape[1] * 2
    flat = x.reshape(-1, in_features).float()

    weight_values = unpack_nvfp4(weight, x.device)  # (out, in) float32
    w_block = _blockwise_scale_expand(block_scales, out_features, in_features, block)
    w_tensor = tensor_scale.float().reshape(())
    # Effective per-element weight = value * block scale * tensor scale.
    weight_deq = (weight_values * w_block * w_tensor).to(torch.bfloat16)

    # Activate through the same grid: block scales + one dynamic tensor scale.
    from jasna.ltx.transformer import fp4_codes  # local import avoids a cycle

    act_scale = flat.abs().amax()
    act_tensor_scale = (act_scale / (6.0 * _FP8_E4M3_MAX)) if act_scale > 0 else torch.ones_like(act_scale)
    blocks = flat.view(flat.shape[0], -1, block)
    act_block_scales = (
        (blocks.abs().amax(-1) / 6.0 / act_tensor_scale).clamp(max=_FP8_E4M3_MAX)
    )
    divisor = act_block_scales * act_tensor_scale
    divisor = torch.where(divisor > 0, divisor, torch.ones_like(divisor))
    codes = fp4_codes(blocks / divisor.unsqueeze(-1)).view(flat.shape[0], -1).float()
    magnitude = _e2m1_lookup_table(x.device)[codes.to(torch.long) & 0x07]
    sign = (codes.to(torch.long) >> 3) & 0x01
    act_deq = (magnitude * (1.0 - 2.0 * sign.float())) * divisor.repeat_interleave(block, dim=1)
    act_deq = act_deq[:, :in_features].to(torch.bfloat16)

    out = act_deq @ weight_deq.t()
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*x.shape[:-1], out_features).to(x.dtype)


def dequantized_int8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """ConvRot W8A8 without ``torch._int_mm`` (pre-Turing CUDA, unusual ROCm)."""
    out_features, in_features = qdata.shape
    flat = x.reshape(-1, in_features)
    out = torch.empty(flat.shape[0], out_features, dtype=x.dtype, device=x.device)
    for start in range(0, flat.shape[0], 4096):
        chunk = flat[start : start + 4096]
        codes, act_scales = _quantize_rows_int8(chunk)
        result = (
            codes.float() @ qdata.float().t()
        ) * (act_scales.unsqueeze(1) * scales.float())
        if bias is not None:
            result = result + bias.float()
        out[start : start + chunk.shape[0]] = result
    return out.reshape(*x.shape[:-1], out_features)


def _quantize_rows_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from jasna.ltx.transformer import quantize_rows, rotate

    return quantize_rows(rotate(x))


__all__ = [
    "ADA",
    "AMPERE",
    "HOPPER",
    "TURING",
    "VOLTA",
    "ArchProfile",
    "clear_dequant_cache",
    "dequantized_int8_linear",
    "dequantized_nvfp4_linear",
    "device_capability",
    "profile_for",
    "recommended_fp16",
    "reset_capability_cache",
    "supports_block_quantized_gemm",
    "supports_fp8_tensor_cores",
    "supports_int8_mm",
    "supports_tensor_cores",
    "tensorrt_min_compute",
    "tensorrt_supports",
    "tensorrt_supports_device",
    "tensorrt_version",
    "unpack_nvfp4",
]
