"""Manual selection of the device that performs video *codec* work.

Jasna always runs the restoration models on the compute device (usually the
CUDA GPU). Decoding the source and encoding the result are a separate concern:
plenty of compute GPUs have no video engines at all (NVIDIA CMP mining cards
such as the 90HX, for example) while the machine's integrated GPU does. This
module lets the user pick which device does the codec work, independently of
where the models run.

Selection precedence:

1. an explicit per-job value (``SessionConfig.codec_backend``)
2. the process-wide value set by the GUI/CLI (``set_codec_backend``)
3. the ``JASNA_CODEC_BACKEND`` environment variable
4. ``auto``

``auto`` resolves to a backend that actually works on this machine: it probes
the compute GPU's encoder for the requested codec first, then Intel Quick Sync,
then AMD AMF, then software. On a host whose compute GPU has working NVENC this
is byte-for-byte the original behaviour; on a host whose compute GPU has none
it quietly lands on the iGPU or the CPU instead of failing.

An explicitly chosen backend is honoured, but if its encoder cannot open the
requested codec (``av1_qsv`` on Alder Lake, for instance) it degrades to
software with a loud warning rather than aborting the job.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum

log = logging.getLogger(__name__)

CODEC_BACKEND_ENV = "JASNA_CODEC_BACKEND"
# Decode can be pinned independently of encode, e.g. Quick Sync encoding with
# CPU decoding.
DECODE_DEVICE_ENV = "JASNA_DECODE_DEVICE"


class CodecBackend(StrEnum):
    """Which device/vendor performs video decode and encode."""

    AUTO = "auto"
    NVIDIA = "nvidia"
    QSV = "qsv"
    AMD = "amd"
    CPU = "cpu"


CODEC_BACKENDS: tuple[str, ...] = tuple(backend.value for backend in CodecBackend)

# Display names for the GUI. Kept free of punctuation so they can be used as
# OptionMenu values directly.
CODEC_BACKEND_LABELS: dict[str, str] = {
    CodecBackend.AUTO.value: "Auto (recommended)",
    CodecBackend.NVIDIA.value: "NVIDIA NVENC / NVDEC",
    CodecBackend.QSV.value: "Intel Quick Sync (iGPU)",
    CodecBackend.AMD.value: "AMD AMF",
    CodecBackend.CPU.value: "CPU / software",
}
CODEC_BACKEND_LABEL_TO_VALUE: dict[str, str] = {
    label: value for value, label in CODEC_BACKEND_LABELS.items()
}

# Encoder used by each backend per output codec.
ENCODER_NAMES_BY_BACKEND: dict[CodecBackend, dict[str, str]] = {
    CodecBackend.NVIDIA: {"hevc": "hevc_nvenc", "h264": "h264_nvenc", "av1": "av1_nvenc"},
    CodecBackend.QSV: {"hevc": "hevc_qsv", "h264": "h264_qsv", "av1": "av1_qsv"},
    CodecBackend.AMD: {"hevc": "hevc_amf", "h264": "h264_amf", "av1": "av1_amf"},
    CodecBackend.CPU: {"hevc": "libx265", "h264": "libx264", "av1": "libsvtav1"},
}

# Second-choice software encoder per codec, for builds that lack the first one.
SOFTWARE_ENCODER_FALLBACKS: dict[str, tuple[str, ...]] = {
    "hevc": ("libx265",),
    "h264": ("libx264",),
    "av1": ("libsvtav1", "libaom-av1"),
}


def is_codec_backend(value: object) -> bool:
    return str(value).strip().lower() in CODEC_BACKENDS


def normalize_codec_backend(
    value: object, *, default: CodecBackend = CodecBackend.AUTO
) -> CodecBackend:
    """Coerce a setting/env value to a CodecBackend, warning on unknown input."""
    text = str(value).strip().lower() if value is not None else ""
    if text == "":
        return default
    try:
        return CodecBackend(text)
    except ValueError:
        log.warning(
            "Unknown codec backend %r; expected one of %s. Using %r.",
            value,
            ", ".join(CODEC_BACKENDS),
            default.value,
        )
        return default


_process_override: CodecBackend | None = None


def set_codec_backend(value: object) -> CodecBackend:
    """Set the process-wide backend used when a job does not name one.

    The GUI calls this before starting a job; the CLI calls it from
    ``--codec-backend``. Returns the normalized value.
    """
    global _process_override
    _process_override = normalize_codec_backend(value)
    log.info("Codec backend set to %s", _process_override.value)
    return _process_override


def process_codec_backend() -> CodecBackend | None:
    return _process_override


def configured_codec_backend() -> CodecBackend:
    """The environment/process-level backend, ignoring per-job values."""
    if _process_override is not None:
        return _process_override
    raw = os.environ.get(CODEC_BACKEND_ENV)
    if raw is None or raw.strip() == "":
        return CodecBackend.AUTO
    return normalize_codec_backend(raw)


def resolve_codec_backend(explicit: object | None = None) -> CodecBackend:
    """Resolve the encoded backend using the documented precedence.

    ``auto`` counts as "no explicit choice" so a per-job value only wins when it
    names a concrete backend; this is what lets the GUI/CLI set a process-wide
    backend while jobs keep their ``auto`` default.
    """
    if explicit is not None and str(explicit).strip() != "":
        normalized = normalize_codec_backend(explicit)
        if normalized is not CodecBackend.AUTO:
            return normalized
    return configured_codec_backend()


# --------------------------------------------------------------------------
# Decode device (independent of the encoding backend)
# --------------------------------------------------------------------------

_decode_device_override: CodecBackend | None = None


def set_decode_device(value: object) -> CodecBackend:
    """Pin the device used for *decoding*, or ``auto`` to follow the encoder."""
    global _decode_device_override
    _decode_device_override = normalize_codec_backend(value)
    log.info("Decode device set to %s", _decode_device_override.value)
    return _decode_device_override


def process_decode_device() -> CodecBackend | None:
    return _decode_device_override


def configured_decode_device() -> CodecBackend:
    if _decode_device_override is not None:
        return _decode_device_override
    raw = os.environ.get(DECODE_DEVICE_ENV)
    if raw is None or raw.strip() == "":
        return CodecBackend.AUTO
    return normalize_codec_backend(raw)


def normalize_compute_vendor(value: object) -> str:
    vendor = str(value).strip().lower()
    if vendor in {"nvidia", "amd"}:
        return vendor
    return "cpu"


def compute_vendor_backend(compute_vendor: object) -> CodecBackend:
    vendor = normalize_compute_vendor(compute_vendor)
    if vendor == "nvidia":
        return CodecBackend.NVIDIA
    if vendor == "amd":
        return CodecBackend.AMD
    return CodecBackend.CPU


def effective_codec_backend(backend: CodecBackend, *, compute_vendor: str) -> CodecBackend:
    """Resolve ``auto`` into a concrete backend from the compute device vendor."""
    if backend is not CodecBackend.AUTO:
        return backend
    return compute_vendor_backend(compute_vendor)


def decode_family(backend: CodecBackend | str, *, compute_vendor: str = "nvidia") -> str:
    """How the reader should decode: ``cuda``, ``qsv``, ``amf`` or ``software``."""
    resolved = effective_codec_backend(
        CodecBackend(str(backend)) if not isinstance(backend, CodecBackend) else backend,
        compute_vendor=compute_vendor,
    )
    if resolved is CodecBackend.NVIDIA:
        return "cuda"
    if resolved is CodecBackend.QSV:
        return "qsv"
    if resolved is CodecBackend.AMD:
        return "amf"
    return "software"


def encoder_name(backend: CodecBackend | str, codec: str) -> str:
    resolved = CodecBackend(str(backend)) if not isinstance(backend, CodecBackend) else backend
    try:
        return ENCODER_NAMES_BY_BACKEND[resolved][codec]
    except KeyError as exc:
        raise ValueError(
            f"No encoder for codec {codec!r} on codec backend {resolved.value!r}"
        ) from exc


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

_probe_cache: dict[tuple[str, str, str], tuple[bool, str]] = {}


def probe_encoder(
    encoder: str,
    *,
    width: int = 1920,
    height: int = 1080,
    pix_fmt: str,
    options: dict[str, str] | None = None,
    frame_rate: int = 30,
) -> tuple[bool, str]:
    """Try to open ``encoder`` and encode one blank frame.

    Being present in the FFmpeg build is not enough: ``av1_qsv`` is compiled in
    on most Windows builds but its runtime rejects the codec, and NVENC is
    compiled in even on cards with no video engines. Both only fail at
    ``avcodec_open2``, so the probe really opens the encoder and pushes a frame.
    """
    key = (encoder, f"{int(width)}x{int(height)}:{pix_fmt}", str(frame_rate))
    cached = _probe_cache.get(key)
    if cached is not None:
        return cached
    result = _probe_encoder_uncached(
        encoder,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        options=options,
        frame_rate=frame_rate,
    )
    _probe_cache[key] = result
    return result


def _probe_encoder_uncached(
    encoder: str,
    *,
    width: int,
    height: int,
    pix_fmt: str,
    options: dict[str, str] | None,
    frame_rate: int,
) -> tuple[bool, str]:
    import io
    from fractions import Fraction

    import av

    try:
        av.Codec(encoder, "w")
    except ValueError as exc:
        return False, f"{encoder} is not in the bundled FFmpeg libraries ({exc})"

    try:
        buf = io.BytesIO()
        dst = av.open(buf, "w", format="null")
    except (av.FFmpegError, ValueError) as exc:  # pragma: no cover - muxer always present
        return False, f"cannot open the probe muxer: {exc}"

    try:
        stream = dst.add_stream(encoder, rate=frame_rate, options=dict(options or {}))
        ctx = stream.codec_context
        ctx.width = int(width)
        ctx.height = int(height)
        ctx.pix_fmt = pix_fmt
        ctx.time_base = Fraction(1, max(1, int(frame_rate)))
        ctx.framerate = Fraction(max(1, int(frame_rate)), 1)
        frame = av.VideoFrame(int(width), int(height), format=pix_fmt)
        frame.pts = 0
        frame.time_base = Fraction(1, max(1, int(frame_rate)))
        stream.encode(frame)
        stream.encode(None)
    except (av.FFmpegError, ValueError, RuntimeError, OSError) as exc:
        return False, str(exc)
    finally:
        try:
            dst.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    return True, ""


def probe_backend_codec(
    backend: CodecBackend,
    codec: str,
    *,
    width: int = 1920,
    height: int = 1080,
    frame_rate: int = 30,
) -> tuple[bool, str, str]:
    """Probe one (backend, codec) pair. Returns ``(ok, encoder, reason)``."""
    from jasna.media.encoder_specs import probe_spec_for

    if backend is CodecBackend.AUTO:  # pragma: no cover - callers resolve first
        return False, "", "auto must be resolved before probing"

    def _try(name: str) -> tuple[bool, str]:
        pix_fmt, options = probe_spec_for(backend, codec, encoder=name)
        return probe_encoder(
            name,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            options=options,
            frame_rate=frame_rate,
        )

    primary = encoder_name(backend, codec)
    ok, reason = _try(primary)
    if ok:
        return True, primary, ""
    if backend is CodecBackend.CPU:
        for fallback in SOFTWARE_ENCODER_FALLBACKS.get(codec, ()):
            if fallback == primary:
                continue
            ok, fb_reason = _try(fallback)
            if ok:
                return True, fallback, ""
            reason = f"{reason}; {fallback}: {fb_reason}"
    return False, primary, reason


# --------------------------------------------------------------------------
# Backend choice
# --------------------------------------------------------------------------

# Order in which ``auto`` (and the explicit-backend fallback chain) tries other
# devices after the compute GPU's own encoder.
_FALLBACK_ORDER: tuple[CodecBackend, ...] = (
    CodecBackend.NVIDIA,
    CodecBackend.QSV,
    CodecBackend.AMD,
    CodecBackend.CPU,
)


@dataclass(frozen=True)
class EncoderChoice:
    """The backend that will actually encode, and why it differs from the request."""

    backend: CodecBackend
    encoder: str
    requested: CodecBackend
    fallback_reason: str = ""

    @property
    def is_fallback(self) -> bool:
        return bool(self.fallback_reason)


_last_choice: dict[str, EncoderChoice] = {}


def last_encoder_choice(codec: str | None = None) -> EncoderChoice | None:
    """The most recent resolved choice, used so decode follows encode."""
    if codec is None:
        if not _last_choice:
            return None
        return next(reversed(_last_choice.values()))
    return _last_choice.get(str(codec).lower())


def clear_backend_cache() -> None:
    _probe_cache.clear()
    _last_choice.clear()


def choose_encoder_backend(
    requested: CodecBackend | str,
    *,
    compute_vendor: str,
    codec: str,
    width: int,
    height: int,
    frame_rate: int = 30,
) -> EncoderChoice:
    """Pick the backend that will really encode this job.

    ``auto`` probes candidate backends in order and returns the first whose
    encoder opens for ``codec``. An explicitly requested backend is tried first
    and degrades to software when it cannot encode the codec.
    """
    resolved_request = (
        requested if isinstance(requested, CodecBackend) else normalize_codec_backend(requested)
    )

    if resolved_request is CodecBackend.AUTO:
        primary = compute_vendor_backend(compute_vendor)
        candidates = [primary] + [b for b in _FALLBACK_ORDER if b is not primary]
        failures: list[str] = []
        for candidate in candidates:
            ok, encoder, reason = probe_backend_codec(
                candidate, codec, width=width, height=height, frame_rate=frame_rate
            )
            if ok:
                choice = EncoderChoice(
                    backend=candidate,
                    encoder=encoder,
                    requested=CodecBackend.AUTO,
                    fallback_reason=(
                        ""
                        if candidate is primary
                        else f"{primary.value} cannot encode {codec} ({failures[0] if failures else 'unavailable'})"
                    ),
                )
                _log_choice(choice, codec, auto=True)
                _last_choice[codec] = choice
                return choice
            failures.append(f"{candidate.value}: {reason}")
        # Nothing worked: hand back the compute GPU's own encoder so the caller
        # raises its normal, detailed open error.
        choice = EncoderChoice(
            backend=primary,
            encoder=encoder_name(primary, codec),
            requested=CodecBackend.AUTO,
            fallback_reason="",
        )
        _last_choice[codec] = choice
        log.error("No codec backend could encode %s: %s", codec, "; ".join(failures))
        return choice

    ok, encoder, reason = probe_backend_codec(
        resolved_request, codec, width=width, height=height, frame_rate=frame_rate
    )
    if ok:
        choice = EncoderChoice(
            backend=resolved_request, encoder=encoder, requested=resolved_request
        )
        if encoder != encoder_name(resolved_request, codec):
            choice = EncoderChoice(
                backend=resolved_request,
                encoder=encoder,
                requested=resolved_request,
                fallback_reason=f"{encoder_name(resolved_request, codec)} is unavailable",
            )
        _last_choice[codec] = choice
        return choice

    if resolved_request is CodecBackend.CPU:
        choice = EncoderChoice(
            backend=CodecBackend.CPU,
            encoder=encoder_name(CodecBackend.CPU, codec),
            requested=resolved_request,
        )
        _last_choice[codec] = choice
        log.error("No software encoder could encode %s: %s", codec, reason)
        return choice

    ok, software_encoder, software_reason = probe_backend_codec(
        CodecBackend.CPU, codec, width=width, height=height, frame_rate=frame_rate
    )
    choice = EncoderChoice(
        backend=CodecBackend.CPU,
        encoder=software_encoder if ok else encoder_name(CodecBackend.CPU, codec),
        requested=resolved_request,
        fallback_reason=(
            f"the {resolved_request.value} encoder {encoder_name(resolved_request, codec)} "
            f"cannot encode {codec} ({reason})"
            if ok
            else f"the {resolved_request.value} encoder {encoder_name(resolved_request, codec)} "
            f"failed ({reason}) and no software encoder worked either ({software_reason})"
        ),
    )
    _log_choice(choice, codec, auto=False)
    _last_choice[codec] = choice
    return choice


def _log_choice(choice: EncoderChoice, codec: str, *, auto: bool) -> None:
    if not choice.is_fallback:
        return
    log.warning(
        "Using %s (%s) for %s %s: %s",
        choice.backend.value.upper(),
        choice.encoder,
        codec,
        "auto-selection" if auto else "fallback",
        choice.fallback_reason,
    )


def resolve_decode_backend(
    explicit: object | None,
    *,
    compute_vendor: str,
    codec: str | None = None,
) -> CodecBackend:
    """Backend the reader should use for decoding.

    Precedence: an explicit per-job decode device, then the process/GUI decode
    device, then -- when that is ``auto`` -- the device the encoder chose, and
    finally the compute GPU. Following the encoder keeps both halves of the
    pipeline on one device, which is the original behaviour; pinning the decode
    device lets a slow or video-engine-less GPU be bypassed on purpose.
    """
    for candidate in (explicit, configured_decode_device()):
        if candidate is None or str(candidate).strip() == "":
            continue
        normalized = normalize_codec_backend(candidate)
        if normalized is not CodecBackend.AUTO:
            return normalized
    # The encoder caches its choice per *output* codec while the reader looks up
    # the *source* codec, so fall back to the most recent choice of any codec.
    cached = last_encoder_choice(codec) or last_encoder_choice()
    if cached is not None and cached.backend is not CodecBackend.CPU:
        return cached.backend
    return compute_vendor_backend(compute_vendor)
