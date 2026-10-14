"""FFmpeg encoder specs for every codec backend.

Jasna historically had exactly two codec backends -- NVENC (NVIDIA) and AMF
(AMD) -- selected implicitly from the compute device. Hosts whose compute GPU
has no video engines need more: Intel Quick Sync on the integrated GPU, or
plain software encoding. Splitting the spec tables out of ``video_encoder``
keeps the selection logic (``jasna.media.codec_backend``) free of a circular
import, and keeps every per-backend FFmpeg option in one place.

``frame_format`` is the *host* pixel format PyAV builds frames in. NVENC is the
only backend that consumes CUDA device frames; every other backend reads
system-memory planes and uploads internally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from jasna.accelerator import AcceleratorVendor
from jasna.media.encoder_settings import encoder_cq_spec

# PyAV hands the encoder exactly the pixel format we name, and unlike the ffmpeg
# CLI nothing inserts a conversion filter in between: naming a format the encoder
# does not list fails avcodec_open2 with EINVAL. libx265 and libsvtav1 advertise
# yuv420p10le but NOT p010le, and yuv420p but NOT nv12, so the software backend
# needs planar chroma. NVENC/QSV/AMF all take the packed P010/NV12 layouts the
# GPU conversions produce.
_TEN_BIT_FRAME_FORMATS = frozenset({"p010le", "yuv420p10le", "yuv420p12le"})


@dataclass(frozen=True)
class EncoderSpec:
    encoder_name: str
    frame_format: str  # FFmpeg pixel format name handed to PyAV
    default_options: Mapping[str, str]
    # True when the format is planar YUV (separate U and V planes) instead of
    # the packed Y + interleaved-UV layout the GPU converters produce.
    planar_chroma: bool = False

    @property
    def ten_bit(self) -> bool:
        return self.frame_format in _TEN_BIT_FRAME_FORMATS


# --------------------------------------------------------------------------
# NVIDIA NVENC
# --------------------------------------------------------------------------

DEFAULT_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "main10",
    "rc": "vbr",
    "cq": str(encoder_cq_spec("hevc", AcceleratorVendor.NVIDIA).default),
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}

# lookahead_level breaks avcodec_open2 on h264_nvenc with this lookahead/AQ
# combination (ENOSYS on RTX 5090), so H.264 deliberately omits it.
DEFAULT_H264_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "high",
    "rc": "vbr",
    # CQ 25 kept representative capped and uncapped H.264 encodes above VMAF 95.
    "cq": str(encoder_cq_spec("h264", AcceleratorVendor.NVIDIA).default),
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}

# AV1 target quality uses a 0..63 scale rather than H.264/HEVC's 0..51.
# CQ 35 matches HEVC CQ 28 on the same seven-above-HEVC scale. AV1 QP limits use a separate
# 0..255 scale, so the HEVC qmin/qmax/init_qp values must not be copied here.
# No profile: P010 input makes av1_nvenc emit AV1 Main 10-bit on its own.
# av1_nvenc only consumes the hyphenated spatial-aq spelling.
DEFAULT_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "rc": "vbr",
    "cq": str(encoder_cq_spec("av1", AcceleratorVendor.NVIDIA).default),
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial-aq": "1",
    "aq-strength": "8",
    "bf": "4",
    "b_ref_mode": "middle",
}

# --------------------------------------------------------------------------
# AMD AMF
# --------------------------------------------------------------------------

DEFAULT_AMF_H264_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "qvbr",
    "qvbr_quality_level": str(
        encoder_cq_spec("h264", AcceleratorVendor.AMD).default
    ),
    "g": "250",
    "preanalysis": "1",
    "vbaq": "1",
    "profile": "high",
}

DEFAULT_AMF_HEVC_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "cqp",
    "qp_i": str(encoder_cq_spec("hevc", AcceleratorVendor.AMD).default),
    "qp_p": str(encoder_cq_spec("hevc", AcceleratorVendor.AMD).default),
    "g": "250",
    "preanalysis": "0",
    "vbaq": "0",
    "profile": "main10",
    "bitdepth": "10",
}


def amf_av1_qindex(cq: int) -> int:
    # AMF AV1 frame QPs are AV1 qindex 0..255; CQ uses the H.264/HEVC QP scale 0..51.
    return cq * 255 // 51


_AMF_AV1_DEFAULT_QINDEX = str(amf_av1_qindex(encoder_cq_spec("av1", AcceleratorVendor.AMD).default))

DEFAULT_AMF_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "cqp",
    "qp_i": _AMF_AV1_DEFAULT_QINDEX,
    "qp_p": _AMF_AV1_DEFAULT_QINDEX,
    "qp_b": _AMF_AV1_DEFAULT_QINDEX,
    "g": "250",
    "preanalysis": "0",
    "aq_mode": "none",
    "profile": "main",
    "bitdepth": "10",
}

# --------------------------------------------------------------------------
# Intel Quick Sync (oneVPL). ``global_quality`` is ICQ, the QSV equivalent of
# NVENC's CQ: a constant-quality target with no bitrate floor. Alder Lake and
# newer iGPUs encode HEVC and H.264; AV1 encode needs Arc or Meteor Lake+.
#
# B-frames are off on purpose. With ``bf`` > 0 the QSV encoder returns packets
# whose DTS equals their display-order PTS, so the muxed stream has a
# non-monotonic DTS (the ffmpeg CLI does exactly the same). Matroska rejects it
# outright and MP4 stores it, after which players decode frames against the
# wrong references -- visible as bursts of garbage frames. ``bf=0`` gives a
# strictly increasing DTS and is the only QSV setting measured to produce a
# clean file.
# --------------------------------------------------------------------------

DEFAULT_QSV_HEVC_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "medium",
    "global_quality": str(encoder_cq_spec("hevc", "qsv").default),
    "profile": "main10",
    "g": "250",
    "bf": "0",
    # No look_ahead: hevc_qsv leaves this option unconsumed through PyAV, which
    # PyAV treats as a hard error ("did not accept encoder option(s)").
}

DEFAULT_QSV_H264_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "medium",
    "global_quality": str(encoder_cq_spec("h264", "qsv").default),
    "profile": "high",
    "g": "250",
    "bf": "0",
}

DEFAULT_QSV_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "medium",
    "global_quality": str(encoder_cq_spec("av1", "qsv").default),
    "profile": "main",
    "g": "250",
    "bf": "0",
}

# --------------------------------------------------------------------------
# Software (libx264 / libx265 / libsvtav1). No video device involved; these run
# on the CPU that also feeds the GPU pipeline.
# --------------------------------------------------------------------------

DEFAULT_X265_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "medium",
    "crf": str(encoder_cq_spec("hevc", "cpu").default),
    "profile": "main10",
    "g": "250",
}

DEFAULT_X264_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "medium",
    "crf": str(encoder_cq_spec("h264", "cpu").default),
    "profile": "high",
    "g": "250",
}

DEFAULT_SVTAV1_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "8",
    "crf": str(encoder_cq_spec("av1", "cpu").default),
    "g": "250",
}

DEFAULT_AOMAV1_ENCODER_OPTIONS: dict[str, str] = {
    "cpu-used": "6",
    "crf": str(encoder_cq_spec("av1", "cpu").default),
    "row-mt": "1",
    "g": "250",
}

# --------------------------------------------------------------------------
# Smart-render fragments splice re-encoded spans between copied source spans,
# so each encoder family needs its own "force a keyframe here" flag name.
# --------------------------------------------------------------------------

NVENC_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced-idr": "1"})
AMF_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced_idr": "1"})
QSV_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced_idr": "1"})
SOFTWARE_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced-idr": "1"})

SMART_FRAGMENT_OPTIONS_BY_BACKEND: dict[str, Mapping[str, str]] = {
    "nvidia": NVENC_SMART_FRAGMENT_OPTIONS,
    "amd": AMF_SMART_FRAGMENT_OPTIONS,
    "qsv": QSV_SMART_FRAGMENT_OPTIONS,
    "cpu": SOFTWARE_SMART_FRAGMENT_OPTIONS,
}

# --------------------------------------------------------------------------
# Spec tables
# --------------------------------------------------------------------------

ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        encoder_name="hevc_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_ENCODER_OPTIONS),
    ),
    "h264": EncoderSpec(
        encoder_name="h264_nvenc",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_H264_ENCODER_OPTIONS),
    ),
    "av1": EncoderSpec(
        encoder_name="av1_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AV1_ENCODER_OPTIONS),
    ),
}

AMF_ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        encoder_name="hevc_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_HEVC_ENCODER_OPTIONS),
    ),
    "h264": EncoderSpec(
        encoder_name="h264_amf",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_AMF_H264_ENCODER_OPTIONS),
    ),
    "av1": EncoderSpec(
        encoder_name="av1_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_AV1_ENCODER_OPTIONS),
    ),
}

QSV_ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        encoder_name="hevc_qsv",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_QSV_HEVC_ENCODER_OPTIONS),
    ),
    "h264": EncoderSpec(
        encoder_name="h264_qsv",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_QSV_H264_ENCODER_OPTIONS),
    ),
    "av1": EncoderSpec(
        encoder_name="av1_qsv",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_QSV_AV1_ENCODER_OPTIONS),
    ),
}

SOFTWARE_ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        encoder_name="libx265",
        frame_format="yuv420p10le",
        default_options=MappingProxyType(DEFAULT_X265_ENCODER_OPTIONS),
        planar_chroma=True,
    ),
    "h264": EncoderSpec(
        encoder_name="libx264",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_X264_ENCODER_OPTIONS),
        planar_chroma=False,
    ),
    "av1": EncoderSpec(
        encoder_name="libsvtav1",
        frame_format="yuv420p10le",
        default_options=MappingProxyType(DEFAULT_SVTAV1_ENCODER_OPTIONS),
        planar_chroma=True,
    ),
}

# Alternate software encoders, keyed by FFmpeg encoder name, used as fallbacks
# when the preferred one is missing from the build.
SOFTWARE_SPECS_BY_ENCODER: dict[str, EncoderSpec] = {
    "libx265": SOFTWARE_ENCODER_SPECS["hevc"],
    "libx264": SOFTWARE_ENCODER_SPECS["h264"],
    "libsvtav1": SOFTWARE_ENCODER_SPECS["av1"],
    "libaom-av1": EncoderSpec(
        encoder_name="libaom-av1",
        frame_format="yuv420p10le",
        default_options=MappingProxyType(DEFAULT_AOMAV1_ENCODER_OPTIONS),
        planar_chroma=True,
    ),
}

SPECS_BY_BACKEND: dict[str, dict[str, EncoderSpec]] = {
    "nvidia": ENCODER_SPECS,
    "amd": AMF_ENCODER_SPECS,
    "qsv": QSV_ENCODER_SPECS,
    "cpu": SOFTWARE_ENCODER_SPECS,
}

# ``auto`` never reaches these tables: callers resolve it first.
SUPPORTED_ENCODER_BACKENDS: frozenset[str] = frozenset(SPECS_BY_BACKEND)


def specs_for_backend(backend: str | object) -> dict[str, EncoderSpec]:
    key = str(backend).strip().lower()
    try:
        return SPECS_BY_BACKEND[key]
    except KeyError as exc:
        raise ValueError(
            f"Gpu video encoding is not available on codec backend {key!r}; "
            f"expected one of {sorted(SUPPORTED_ENCODER_BACKENDS)}"
        ) from exc


def smart_fragment_options(backend: str | object) -> Mapping[str, str]:
    return SMART_FRAGMENT_OPTIONS_BY_BACKEND.get(
        str(backend).strip().lower(), SOFTWARE_SMART_FRAGMENT_OPTIONS
    )


def eight_bit_fragment_spec(
    spec: EncoderSpec, codec: str, backend: str | object = "nvidia"
) -> EncoderSpec:
    """Smart-render fragments keep the source bit depth instead of the codec default.

    The 8-bit format depends on the backend: the GPU encoders take packed NV12,
    while libx265/libsvtav1 take planar yuv420p.
    """
    options = dict(spec.default_options)
    if codec == "hevc":
        options["profile"] = "main"
    # AMF pins output depth via "bitdepth"; dropping it lets FFmpeg derive
    # 8-bit from the nv12 input instead of conflicting with it.
    options.pop("bitdepth", None)
    if str(backend).strip().lower() == "cpu" and codec in {"hevc", "av1"}:
        frame_format, planar = "yuv420p", True
    else:
        frame_format, planar = "nv12", False
    return replace(
        spec,
        frame_format=frame_format,
        planar_chroma=planar,
        default_options=MappingProxyType(options),
    )


def probe_spec_for(
    backend: str | object,
    codec: str,
    *,
    ten_bit: bool = True,
    encoder: str | None = None,
) -> tuple[str, dict[str, str]]:
    """The pixel format and options the probe should open an encoder with.

    Mirrors what ``VideoEncoder`` will actually use, so a successful probe is a
    real guarantee that the job's first frame will not fail to open the encoder.
    """
    if encoder is not None:
        spec = SOFTWARE_SPECS_BY_ENCODER.get(encoder)
        if spec is None:
            spec = specs_for_backend(backend)[codec]
    else:
        spec = specs_for_backend(backend)[codec]
    return spec.frame_format, dict(spec.default_options)
