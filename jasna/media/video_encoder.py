from __future__ import annotations

import heapq
import logging
import math
import queue
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    device_name,
    new_event,
    new_stream,
    set_device,
    stream_context,
    vendor_for_device,
)
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.cas import GpuCasSharpener
from jasna.media.container_utils import (
    MOV_SUFFIXES,
    is_mov_chapter_stream,
    subtitle_transcode_codec,
)
from jasna.media.encoder_settings import encoder_cq_spec, validate_encoder_settings
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.probe import VideoMetadata
from jasna.media.rgb_to_yuv import RgbToYuvConverter

logger = logging.getLogger(__name__)

# Encoder specs and FFmpeg option tables live in jasna.media.encoder_specs so the
# backend-selection module (jasna.media.codec_backend) can reach them without a
# circular import. Everything is re-exported here because this module is the
# historical import site for these names.
from jasna.media.codec_backend import (  # noqa: E402
    CodecBackend,
    EncoderChoice,
    choose_encoder_backend,
    resolve_codec_backend,
)
from jasna.media.encoder_specs import (  # noqa: E402,F401
    AMF_ENCODER_SPECS,
    AMF_SMART_FRAGMENT_OPTIONS,
    DEFAULT_AMF_AV1_ENCODER_OPTIONS,
    DEFAULT_AMF_H264_ENCODER_OPTIONS,
    DEFAULT_AMF_HEVC_ENCODER_OPTIONS,
    DEFAULT_AOMAV1_ENCODER_OPTIONS,
    DEFAULT_AV1_ENCODER_OPTIONS,
    DEFAULT_ENCODER_OPTIONS,
    DEFAULT_H264_ENCODER_OPTIONS,
    DEFAULT_QSV_AV1_ENCODER_OPTIONS,
    DEFAULT_QSV_H264_ENCODER_OPTIONS,
    DEFAULT_QSV_HEVC_ENCODER_OPTIONS,
    DEFAULT_SVTAV1_ENCODER_OPTIONS,
    DEFAULT_X264_ENCODER_OPTIONS,
    DEFAULT_X265_ENCODER_OPTIONS,
    ENCODER_SPECS,
    NVENC_SMART_FRAGMENT_OPTIONS,
    QSV_ENCODER_SPECS,
    QSV_SMART_FRAGMENT_OPTIONS,
    SOFTWARE_ENCODER_SPECS,
    SOFTWARE_SMART_FRAGMENT_OPTIONS,
    EncoderSpec,
    amf_av1_qindex,
    eight_bit_fragment_spec,
    smart_fragment_options,
    specs_for_backend,
)

# Backends whose encoder consumes system-memory planes. NVENC is the only one
# that wants CUDA device frames; QSV uploads internally and the software
# encoders read host memory directly.
_HOST_FRAME_BACKENDS = frozenset({CodecBackend.QSV, CodecBackend.AMD, CodecBackend.CPU})


def _is_host_frame_backend(backend: CodecBackend) -> bool:
    return backend in _HOST_FRAME_BACKENDS

# ITU-T H.273 matrix, primaries, and transfer-characteristic code points.
_COLOR_TAGS = {
    AvColorspace.ITU709: (1, 1, 1),
    AvColorspace.ITU601: (6, 6, 6),
    AvColorspace.BT2020: (9, 9, 14),  # bt2020nc, bt2020 primaries, bt2020-10 transfer
}
_COLOR_PRIMARIES = {
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "bt2020": 9,
}
_COLOR_TRANSFERS = {
    "bt709": 1,
    "smpte170m": 6,
    "bt2020-10": 14,
    "smpte2084": 16,
    "arib-std-b67": 18,
}
_COLOR_VARIANTS = {
    (AvColorspace.ITU709, AvColorRange.MPEG): "bt709_limited",
    (AvColorspace.ITU709, AvColorRange.JPEG): "bt709_full",
    (AvColorspace.ITU601, AvColorRange.MPEG): "bt601_limited",
    (AvColorspace.ITU601, AvColorRange.JPEG): "bt601_full",
    (AvColorspace.BT2020, AvColorRange.MPEG): "bt2020_limited",
    (AvColorspace.BT2020, AvColorRange.JPEG): "bt2020_full",
}

_NVENC_PITCH_ALIGNMENT = 16
ENCODE_BUFFER_SIZE = 8


def _plane_payload(
    src: torch.Tensor, width: int, sample_bytes: int, line_size: int, buffer_size: int
) -> bytes:
    """Lay ``src`` out for one AVFrame plane, honouring its line size and padding.

    ``VideoPlane.update`` wants a buffer of the plane's full size, so rows are
    placed at ``line_size`` offsets and any trailing padding is zeroed.
    """
    import numpy as np

    row_bytes = width * sample_bytes
    rows = int(src.shape[0])
    data = np.zeros(buffer_size, dtype=np.uint8)
    view = data[: rows * line_size].reshape(rows, line_size)
    view[:, :row_bytes] = (
        src.detach().to("cpu").contiguous().numpy().view(np.uint8).reshape(rows, row_bytes)
    )
    return data.tobytes()


def _host_planes(
    host: torch.Tensor, height: int, *, planar: bool, ten_bit: bool
) -> list[torch.Tensor]:
    """Split the packed Y + interleaved-UV buffer into the encoder's planes.

    The GPU conversions always produce NV12/P010 layout: a Y plane followed by a
    chroma plane whose U and V samples alternate. PyAV hands the encoder exactly
    the layout it is told, so packed formats (NV12/P010) keep that chroma plane
    while planar formats (yuv420p/yuv420p10le) need U and V separately.
    """
    y = host[:height]
    uv = host[height:]
    if not planar:
        return [y, uv]
    # U and V alternate inside each chroma row: even indices are U, odd are V.
    return [y, uv[:, 0::2].contiguous(), uv[:, 1::2].contiguous()]


# `cq` alone targets a fixed quality and ignores how the source was stored, so a
# cheaply encoded source is re-encoded far above its own quality point and grows
# several times over (issues #235, #243). A ceiling tied to the source bitrate
# bounds that without measurably costing quality: across 27 clips (1080p to 8K,
# VR and flat) capped encodes landed on the uncapped quality-vs-bitrate curve to
# within 0.07 VMAF, and the ceiling stays inert on sources that were already
# generously encoded. HEVC sources get restoration headroom; NVIDIA H.264 output
# gets a larger ceiling because the old 1x limit flattened CQ values (issue #282).
SOURCE_BITRATE_CAP_FACTORS: dict[str, float] = {"hevc": 1.25}
DEFAULT_SOURCE_BITRATE_CAP_FACTOR = 1.0
NVENC_H264_SOURCE_BITRATE_CAP_FACTOR = 2.0
# Any VBV buffer of roughly a second or more never becomes the binding
# constraint; only sub-second buffers throttle, which is the #243 unit trap.
SOURCE_BITRATE_CAP_BUFFER_RATIO = 2


def source_bitrate_cap_options(
    metadata: VideoMetadata,
    *,
    output_codec: str,
    vendor: AcceleratorVendor | str,
) -> dict[str, str]:
    if metadata.video_bitrate <= 0:
        logger.warning(
            "No source bitrate for %s; encoding without a source-tied bitrate ceiling",
            metadata.video_file,
        )
        return {}
    # Compare by value: this is called with both AcceleratorVendor and the
    # codec-backend enum, which are distinct types sharing the "nvidia" value.
    if str(vendor) == str(AcceleratorVendor.NVIDIA) and output_codec == "h264":
        factor = NVENC_H264_SOURCE_BITRATE_CAP_FACTOR
    else:
        factor = SOURCE_BITRATE_CAP_FACTORS.get(
            metadata.codec_name.lower(), DEFAULT_SOURCE_BITRATE_CAP_FACTOR
        )
    maxrate = int(metadata.video_bitrate * factor)
    return {
        "maxrate": str(maxrate),
        "bufsize": str(maxrate * SOURCE_BITRATE_CAP_BUFFER_RATIO),
    }


def _option_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _drop_unsupported_nvenc_overrides(
    codec: str, overrides: dict[str, str], defaults: Mapping[str, str]
) -> None:
    # NVENC rejects these combinations at avcodec_open2, so dropping them with
    # a warning beats failing the whole job.
    if codec == "h264" and "lookahead_level" in overrides:
        overrides.pop("lookahead_level")
        logger.warning("dropping lookahead_level: h264_nvenc fails to open with it")
    if overrides.get("weighted_pred", "0") != "0":
        if codec == "av1":
            overrides.pop("weighted_pred")
            logger.warning("dropping weighted_pred: av1_nvenc does not support it")
        elif overrides.get("bf", defaults.get("bf", "0")) != "0":
            overrides.pop("weighted_pred")
            logger.warning("dropping weighted_pred: NVENC supports it only with bf=0")


def _normalize_amf_cq(
    codec: str,
    overrides: dict[str, str],
    defaults: dict[str, str],
    *,
    ten_bit: bool,
) -> None:
    rc = overrides.get("rc", defaults["rc"])
    cqp_modes = {"cqp", "0"}
    qvbr_modes = {"qvbr", "hqvbr", "4", "5"}
    if codec in {"hevc", "av1"} and rc not in cqp_modes:
        defaults.pop("qp_i", None)
        defaults.pop("qp_p", None)
        defaults.pop("qp_b", None)
    if codec in {"hevc", "av1"} and ten_bit and rc in qvbr_modes:
        raise ValueError(
            f"AMD {codec.upper()} Main10 does not support QVBR or HQVBR; use the default CQP mode"
        )

    aliases = [key for key in ("cq", "qvbr_quality_level") if key in overrides]
    if len(aliases) > 1:
        raise ValueError(
            "Conflicting encoder settings: cq and qvbr_quality_level are aliases "
            "on AMD; use only one"
        )
    if not aliases:
        return

    value = overrides.pop(aliases[0])
    if codec in {"hevc", "av1"}:
        if rc in cqp_modes and codec == "av1":
            qindex = str(amf_av1_qindex(int(value)))
            overrides.update(qp_i=qindex, qp_p=qindex, qp_b=qindex)
        elif rc in cqp_modes:
            overrides["qp_i"] = value
            overrides["qp_p"] = value
        elif not ten_bit and rc in qvbr_modes:
            overrides["qvbr_quality_level"] = value
        else:
            raise ValueError(f"AMD {codec.upper()} CQ requires rc=cqp")
    else:
        overrides["qvbr_quality_level"] = value


def _align_yuv_pitch(packed: torch.Tensor) -> torch.Tensor:
    item_size = packed.element_size()
    if packed.stride(0) * item_size % _NVENC_PITCH_ALIGNMENT == 0:
        return packed

    width = packed.shape[1]
    row_bytes = width * item_size
    aligned_row_bytes = (
        row_bytes + _NVENC_PITCH_ALIGNMENT - 1
    ) // _NVENC_PITCH_ALIGNMENT * _NVENC_PITCH_ALIGNMENT
    pitch_elements = aligned_row_bytes // item_size
    storage = packed.new_empty((packed.shape[0], pitch_elements))
    storage[:, :width].copy_(packed)
    storage[:, width:].zero_()
    return storage[:, :width]


def _mov_container_options(suffix: str, *, fmp4: bool) -> dict[str, str]:
    if suffix.lower() not in MOV_SUFFIXES:
        return {}
    # A fragmented MP4 writes a sample-free moov up front and one moof+mdat per
    # keyframe, so the growing file stays playable; +faststart instead relocates
    # a single moov at close, leaving the file unreadable until then. The two
    # are mutually exclusive.
    flags = "+frag_keyframe+empty_moov" if fmp4 else "+faststart"
    return {"movflags": flags}


def _normalized_audio_layout(layout: av.AudioLayout) -> av.AudioLayout:
    channels = layout.nb_channels
    if layout.name == f"{channels} channels" and channels <= 8:
        return av.AudioLayout(f"{channels}c")
    return layout


def _apply_qsv_cq(
    overrides: dict[str, str], defaults: dict[str, str]
) -> None:
    """Map the portable ``cq`` control onto QSV's ``global_quality`` (ICQ)."""
    value = overrides.pop("cq", None)
    if value is None:
        return
    if "global_quality" in overrides and overrides["global_quality"] != value:
        raise ValueError(
            "Conflicting encoder settings: cq and global_quality both set the QSV "
            "quality target; use only one"
        )
    overrides["global_quality"] = value
    defaults.setdefault("global_quality", value)


def _apply_software_cq(
    codec: str, overrides: dict[str, str], defaults: dict[str, str]
) -> None:
    """Map the portable ``cq`` control onto the software encoders' ``crf``."""
    value = overrides.pop("cq", None)
    if value is None:
        return
    if "crf" in overrides and overrides["crf"] != value:
        raise ValueError(
            "Conflicting encoder settings: cq and crf both set the quality target; "
            "use only one"
        )
    overrides["crf"] = value
    defaults.setdefault("crf", value)


def resolve_encoder_options(
    backend: CodecBackend | str,
    codec: str,
    metadata: VideoMetadata,
    encoder_settings: dict[str, object],
    *,
    smart_fragment: bool,
    encoder_name_override: str | None = None,
) -> tuple[EncoderSpec, dict[str, str]]:
    """Pick the encoder spec and the final FFmpeg options for one output.

    ``backend`` is a concrete codec backend ("nvidia", "qsv", "amd", "cpu");
    ``auto`` must already be resolved by the caller.
    """
    resolved_backend = (
        backend if isinstance(backend, CodecBackend) else CodecBackend(str(backend))
    )
    specs = specs_for_backend(resolved_backend)
    if codec not in specs:
        raise ValueError(f"Unsupported codec: {codec}")
    spec = specs[codec]
    if encoder_name_override and encoder_name_override != spec.encoder_name:
        from jasna.media.encoder_specs import SOFTWARE_SPECS_BY_ENCODER

        alternate = SOFTWARE_SPECS_BY_ENCODER.get(encoder_name_override)
        if alternate is None:
            raise ValueError(
                f"Unknown software encoder {encoder_name_override!r} for codec {codec}"
            )
        spec = alternate
    # Smart-render fragments are spliced between copied source spans, so they
    # keep the source bit depth instead of the codec's 10-bit default.
    if smart_fragment and codec in {"hevc", "av1"} and not metadata.is_10bit:
        spec = eight_bit_fragment_spec(spec, codec, resolved_backend)
    if encoder_settings:
        validate_encoder_settings(encoder_settings, codec=codec, vendor=resolved_backend)

    encoder_options = dict(spec.default_options)
    overrides: dict[str, str] = {}
    if encoder_settings:
        overrides = {k: _option_value(v) for k, v in encoder_settings.items()}
        # FFmpeg accepts both spellings for HEVC/H.264, but their defaults
        # use the underscore key. Normalize the alias so a user override
        # replaces that default instead of passing two conflicting options.
        if "spatial-aq" in overrides and "spatial_aq" in encoder_options:
            overrides["spatial_aq"] = overrides.pop("spatial-aq")
        if resolved_backend is CodecBackend.AMD:
            _normalize_amf_cq(codec, overrides, encoder_options, ten_bit=spec.ten_bit)
        elif resolved_backend is CodecBackend.QSV:
            _apply_qsv_cq(overrides, encoder_options)
        elif resolved_backend is CodecBackend.CPU:
            _apply_software_cq(codec, overrides, encoder_options)
        else:
            _drop_unsupported_nvenc_overrides(codec, overrides, encoder_options)
    uses_amf_cqp = (
        resolved_backend is CodecBackend.AMD
        and codec in {"hevc", "av1"}
        and overrides.get("rc", encoder_options["rc"]) in {"cqp", "0"}
    )
    if "maxrate" not in overrides and not uses_amf_cqp:
        encoder_options.update(source_bitrate_cap_options(metadata, output_codec=codec, vendor=resolved_backend))
    encoder_options.update(overrides)
    if smart_fragment:
        encoder_options.update(smart_fragment_options(resolved_backend))
    return spec, encoder_options


class VideoEncoder:
    def __init__(
        self,
        file: str,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        codec: str,
        encoder_settings: dict[str, object],
        lut_path: str | Path | None = None,
        sharpen_strength: float = 0.0,
        output_fps: Fraction | None = None,
        pts_origin: int = 0,
        smart_fragment: bool = False,
        fmp4: bool = False,
        codec_backend: object | None = None,
    ):
        self.device = torch.device(device)
        self.vendor = vendor_for_device(self.device)
        self.codec = codec
        self.requested_codec_backend = resolve_codec_backend(codec_backend)
        self.smart_fragment = bool(smart_fragment)
        self.fmp4 = bool(fmp4)

        # Which device actually does the codec work. ``auto`` probes this
        # machine; an explicit choice degrades to software when its encoder
        # cannot handle the requested codec.
        choice = choose_encoder_backend(
            self.requested_codec_backend,
            compute_vendor=str(self.vendor),
            codec=codec,
            width=int(metadata.video_width),
            height=int(metadata.video_height),
            frame_rate=int(round(float(metadata.video_fps_exact or 30)) or 30),
        )
        self.backend_choice: EncoderChoice = choice
        self.codec_backend = choice.backend
        if (
            self.codec_backend is CodecBackend.NVIDIA
            and self.vendor is not AcceleratorVendor.NVIDIA
        ):
            raise RuntimeError(
                "The NVIDIA codec backend needs a CUDA compute device; "
                f"the compute device is {self.vendor.value}"
            )
        if choice.is_fallback:
            logger.warning(
                "Encoding %s with %s instead of %s: %s",
                codec,
                choice.encoder,
                choice.requested.value,
                choice.fallback_reason,
            )

        spec, self.encoder_options = resolve_encoder_options(
            self.codec_backend,
            codec,
            metadata,
            encoder_settings,
            smart_fragment=smart_fragment,
            encoder_name_override=choice.encoder,
        )
        color_variant = _COLOR_VARIANTS.get((metadata.color_space, metadata.color_range))
        if color_variant is None:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")
        pixel_format = "p010" if spec.ten_bit else "nv12"
        self.metadata = metadata
        self.output_path = Path(file)
        self.spec = spec
        self.encoder_name = spec.encoder_name
        self.pts_origin = int(pts_origin)
        self.output_fps = Fraction(
            metadata.video_fps_exact if output_fps is None else output_fps
        )

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        self._cas: GpuCasSharpener | None = None
        if sharpen_strength > 0.0:
            self._cas = GpuCasSharpener(
                sharpen_strength, ten_bit=spec.ten_bit, device=self.device
            )

        self._converter = RgbToYuvConverter(f"{pixel_format}_{color_variant}", device=self.device)

        self._lut_flags: deque[bool] = deque()
        # Only AMD reuses one packed frame (set in __enter__); NVENC still holds
        # its input frame after encode() returns, so NVIDIA allocates per frame.
        self._packed: torch.Tensor | None = None
        self._cas_luma: torch.Tensor | None = None
        self._source_chapters = ()

    def __enter__(self):
        try:
            av.Codec(self.encoder_name, "w")
        except ValueError as exc:  # av.codec.codec.UnknownCodecError
            raise RuntimeError(
                f"Encoder {self.encoder_name} (codec {self.codec}) is not available in the "
                f"bundled FFmpeg libraries: {exc}"
            ) from exc
        self._src = av.open(self.metadata.video_file)
        in_v = self._src.streams.video[0]

        container_options = _mov_container_options(
            self.output_path.suffix, fmp4=self.fmp4
        )
        self.dst = av.open(str(self.output_path), "w", container_options=container_options)

        stream_kwargs = {
            "rate": self.output_fps,
            "options": dict(self.encoder_options),
        }
        if self.codec_backend is CodecBackend.AMD:
            stream_kwargs["hwaccel"] = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                is_hw_owned=False,
            )
            pix_fmt = self.spec.frame_format
        elif self.codec_backend is CodecBackend.NVIDIA:
            pix_fmt = "cuda"
        else:
            # QSV and the software encoders both take system-memory planes and
            # do their own upload, so no hwaccel context is needed here.
            pix_fmt = self.spec.frame_format
        out_v = self.dst.add_stream(self.encoder_name, **stream_kwargs)
        if self.codec == "hevc" and self.output_path.suffix.lower() in MOV_SUFFIXES:
            out_v.codec_tag = "hvc1"
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.metadata.time_base
        ctx = out_v.codec_context
        ctx.time_base = self.metadata.time_base
        ctx.framerate = self.output_fps
        ctx.pix_fmt = pix_fmt
        if self.smart_fragment:
            from av.codec.context import Flags

            ctx.flags |= Flags.closed_gop
        if self.metadata.sample_aspect_ratio != 1:
            ctx.sample_aspect_ratio = self.metadata.sample_aspect_ratio
        matrix, primaries, transfer = _COLOR_TAGS[self.metadata.color_space]
        primaries = _COLOR_PRIMARIES.get(self.metadata.color_primaries.lower(), primaries)
        transfer = _COLOR_TRANSFERS.get(self.metadata.color_transfer.lower(), transfer)
        ctx.color_range = int(self.metadata.color_range)
        ctx.colorspace = matrix
        ctx.color_primaries = primaries
        ctx.color_trc = transfer
        self.out_stream = out_v

        self._copy_source_metadata(in_v, out_v)
        self._setup_source_streams(in_v)

        # Wrap torch's already-current primary context.  FFmpeg's primary_ctx
        # mode tries to change its scheduling flags and fails once torch has
        # initialized it; current_ctx leaves the context and its flags alone.
        # Keeping conversion and NVENC in one context also avoids a ~500 MiB
        # secondary CUDA context and cross-context scheduling overhead.
        # NVENC consumes device memory, so conversion runs on its own stream and
        # overlaps the rest of the pipeline. AMF consumes host memory and the
        # conversion is eager Torch math, so on AMD everything stays on the
        # current stream: a private stream there let ROCm recycle in-flight
        # conversion buffers into the restorer's allocations (issue #252).
        height = self.metadata.video_height
        width = self.metadata.video_width
        self._cuda_ctx = None
        self._host_yuv = None
        if self.codec_backend is CodecBackend.NVIDIA:
            from av.video.frame import CudaContext

            self.stream = new_stream(self.device)
            self._cuda_ctx = CudaContext(
                device_id=self.device.index or 0,
                primary_ctx=False,
                current_ctx=True,
                cuda_stream=self.stream.cuda_stream,
            )
        else:
            self.stream = current_stream(self.device)
            self._packed = torch.empty(
                (height + height // 2, width),
                dtype=self._converter.sample_dtype,
                device=self.device,
            )
            if self._cas is not None:
                self._cas_luma = torch.empty_like(self._packed[:height])
            self._host_yuv = torch.empty(
                (height + height // 2, width),
                dtype=torch.uint16 if self.spec.ten_bit else torch.uint8,
                pin_memory=True,
            )
        self.pts_heap: list[int] = []
        self.frame_buffer: deque = deque()
        self._lut_flags.clear()
        self.pts_set: set[int] = set()
        self._last_emitted_pts: int | None = None
        self._video_started = False
        self._options_validated = False
        self._worker_error: Exception | None = None

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=ENCODE_BUFFER_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="VideoEncoderWorker", daemon=True)
        self._encode_thread.start()
        return self

    def _copy_source_metadata(self, in_v, out_v) -> None:
        self.dst.metadata.update(self._src.metadata)
        out_v.metadata.update(in_v.metadata)
        out_v.disposition = in_v.disposition
        if not self.smart_fragment:
            self._source_chapters = self._src.chapters()
            self.dst.set_chapters(self._source_chapters)

    def _setup_source_streams(self, in_v) -> None:
        self._source_pipes: dict[int, tuple[str, object, object]] = {}
        self._source_backlog: deque = deque()
        self._source_iter = None
        self._last_source_dts: dict[int, tuple[int, Fraction]] = {}
        self._warned_source_dts: set[int] = set()
        self._last_video_dts: int | None = None
        self._warned_non_monotonic_dts = False
        if self.smart_fragment:
            return

        packet_streams = []
        output_formats = set(self.dst.format.name.split(","))
        source_formats = set(self._src.format.name.split(","))
        source_chapters = self._source_chapters
        for in_stream in self._src.streams:
            if in_stream.index == in_v.index:
                continue
            if is_mov_chapter_stream(
                in_stream,
                source_formats=source_formats,
                chapters=source_chapters,
            ):
                continue
            if in_stream.type == "attachment" and "matroska" not in output_formats:
                logger.warning(
                    "Skipping attachment stream %s: %s output does not support attachments",
                    in_stream.index,
                    self.output_path.suffix,
                )
                continue
            if in_stream.codec_context is None and in_stream.type != "attachment":
                logger.warning(
                    "Skipping %s stream %s: it has no copyable codec",
                    in_stream.type,
                    in_stream.index,
                )
                continue

            source_audio_layout = (
                in_stream.codec_context.layout
                if in_stream.type == "audio"
                else None
            )
            audio_layout = (
                _normalized_audio_layout(source_audio_layout)
                if source_audio_layout is not None
                else None
            )
            if in_stream.type == "audio" and needs_audio_reencode(
                in_stream.codec_context.name,
                self.output_path.suffix,
                self.dst.supported_codecs,
            ):
                logger.info(
                    "re-encoding audio %s -> aac for %s",
                    in_stream.codec_context.name,
                    self.output_path.suffix,
                )
                out_stream = self.dst.add_stream(
                    "aac", rate=in_stream.codec_context.sample_rate
                )
                out_stream.codec_context.layout = audio_layout
                out_stream.bit_rate = 256_000
                processor = av.AudioResampler(
                    format="fltp",
                    layout=audio_layout,
                    rate=in_stream.codec_context.sample_rate,
                )
                kind = "transcode"
            else:
                try:
                    out_stream = self.dst.add_stream_from_template(
                        in_stream, opaque=True
                    )
                except ValueError as exc:
                    transcode_codec = subtitle_transcode_codec(
                        in_stream.codec_context.name,
                        output_formats=output_formats,
                        supported_codecs=self.dst.supported_codecs,
                    )
                    if in_stream.type != "subtitle" or transcode_codec is None:
                        logger.warning(
                            "Skipping %s stream %s: %s",
                            in_stream.type,
                            in_stream.index,
                            exc,
                        )
                        continue
                    logger.info(
                        "re-encoding subtitle %s -> %s for %s",
                        in_stream.codec_context.name,
                        transcode_codec,
                        self.output_path.suffix,
                    )
                    out_stream = self.dst.add_stream(transcode_codec)
                    subtitle_time_base = (
                        getattr(in_stream, "time_base", None)
                        or Fraction(1, 1_000)
                    )
                    out_stream.time_base = subtitle_time_base
                    out_stream.codec_context.time_base = subtitle_time_base
                    out_stream.codec_context.subtitle_header = b""
                    processor = None
                    kind = "subtitle_transcode"
                else:
                    if (
                        audio_layout is not None
                        and audio_layout is not source_audio_layout
                    ):
                        out_stream.codec_context.layout = audio_layout
                    processor = self._aac_copy_filter(
                        in_stream, out_stream, output_formats
                    )
                    kind = "copy"

            out_stream.metadata.update(in_stream.metadata)
            out_stream.disposition = in_stream.disposition
            if in_stream.type == "attachment":
                continue
            self._source_pipes[in_stream.index] = (kind, out_stream, processor)
            packet_streams.append(in_stream)

        if packet_streams:
            self._source_iter = self._src.demux(packet_streams)

    @staticmethod
    def _aac_copy_filter(in_stream, out_stream, output_formats):
        if (
            in_stream.type == "audio"
            and in_stream.codec_context.name == "aac"
            and output_formats & {"mp4", "mov"}
            and out_stream.codec_context.extradata
        ):
            return av.bitstream.BitStreamFilterContext("aac_adtstoasc", in_stream)
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                while self.frame_buffer:
                    self._process_buffer(flush_all=True)
            self._encode_queue.join()
            self._encode_queue.put(self._stop_sentinel)
            self._encode_thread.join()

            if exc_type is None and self._worker_error is None and self.out_stream.codec_context.is_open:
                for packet in self.out_stream.encode(None):
                    self._mux_video(packet)
                self._drain_source_streams()
        finally:
            self.dst.close()
            self._src.close()
        if exc_type is None and self._worker_error is not None:
            raise self._worker_error

    def _encode_worker(self):
        set_device(self.device)

        while True:
            item = self._encode_queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                if self._worker_error is None:
                    frame, pts, apply_lut, ready_event = item
                    self.stream.wait_event(ready_event)
                    frame.record_stream(self.stream)
                    self._encode_frame(frame, pts, apply_lut=apply_lut)
            except Exception as exc:
                self._worker_error = exc
                logger.exception("[encoder-worker] crashed")
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
        apply_lut: bool,
    ) -> tuple[torch.Tensor, int, bool, object]:
        producer_stream = current_stream(self.device)
        ready_event = new_event(self.device)
        producer_stream.record_event(ready_event)
        return frame, pts, apply_lut, ready_event

    def _validate_encoder_options(self):
        leftover = dict(self.out_stream.codec_context.options)
        if leftover:
            raise ValueError(f"{self.encoder_name} did not accept encoder option(s): {sorted(leftover)}")
        self._options_validated = True

    def _mux_video(self, packet: av.Packet):
        threshold = (
            float(packet.dts * packet.time_base)
            if packet.dts is not None and packet.time_base is not None
            else None
        )
        # A container needs a non-decreasing DTS. Some hardware encoders hand
        # back packets whose DTS equals their display-order PTS, which produces a
        # stream that players decode against the wrong references (visible as
        # bursts of garbage frames) and that Matroska rejects outright. Warn once
        # so it is diagnosable instead of looking like a random rendering bug.
        dts = packet.dts
        if dts is not None:
            if self._last_video_dts is not None and dts < self._last_video_dts:
                if not self._warned_non_monotonic_dts:
                    self._warned_non_monotonic_dts = True
                    logger.warning(
                        "Encoder %s emitted a non-monotonic DTS (%s after %s); the "
                        "output will have corrupted frames. Disable B-frames for "
                        "this encoder or pick another codec backend.",
                        self.encoder_name,
                        dts,
                        self._last_video_dts,
                    )
            self._last_video_dts = dts
        try:
            self.dst.mux(packet)
        except av.FFmpegError as exc:
            raise RuntimeError(
                f"Failed to mux {self.codec} video into '{self.output_path.suffix}' output: {exc}"
            ) from exc
        self._video_started = True
        if not self._options_validated:
            self._validate_encoder_options()
        if threshold is not None:
            self._pump_source_streams(threshold)

    def _produce_source_packets(self, in_packet) -> list:
        kind, out_stream, processor = self._source_pipes[in_packet.stream.index]
        if kind == "copy":
            return self._copy_source_packets(in_packet, out_stream, processor)
        if kind == "subtitle_transcode":
            if in_packet.size == 0:
                return []
            subtitle = in_packet.stream.codec_context.decode2(in_packet)
            if subtitle is None:
                return []
            packet = out_stream.codec_context.encode_subtitle(subtitle)
            output_time_base = (
                out_stream.time_base
                or out_stream.codec_context.time_base
                or in_packet.time_base
                or Fraction(1, 1_000)
            )

            def rescale(value):
                if value is None or in_packet.time_base is None:
                    return value
                return round(value * in_packet.time_base / output_time_base)

            packet.stream = out_stream
            packet.pts = rescale(in_packet.pts)
            packet.dts = rescale(
                in_packet.dts if in_packet.dts is not None else in_packet.pts
            )
            packet.duration = rescale(in_packet.duration)
            packet.time_base = output_time_base
            return [packet]
        out_packets = []
        sample_time_base = Fraction(1, out_stream.codec_context.sample_rate)
        for aframe in in_packet.decode():
            if aframe.pts is not None:
                aframe.pts = round(aframe.pts * aframe.time_base / sample_time_base)
            aframe.time_base = sample_time_base
            for rframe in processor.resample(aframe):
                out_packets.extend(out_stream.encode(rframe))
        return out_packets

    @staticmethod
    def _copy_source_packets(in_packet, out_stream, bitstream_filter):
        if in_packet.size == 0:
            return []
        if bitstream_filter is None:
            packets = [in_packet]
        else:
            packets = bitstream_filter.filter(in_packet)
        for packet in packets:
            packet.stream = out_stream
        return packets

    def _pump_source_streams(self, upto_seconds: float | None):
        if self._source_iter is None:
            return
        while True:
            if self._source_backlog:
                packet = self._source_backlog[0]
                ts = packet.dts if packet.dts is not None else packet.pts
                if (
                    upto_seconds is not None
                    and ts is not None
                    and packet.time_base is not None
                    and float(ts * packet.time_base) > upto_seconds
                ):
                    return
                self._source_backlog.popleft()
                self._mux_source_packet(packet)
                continue
            in_packet = next(self._source_iter, None)
            if in_packet is None:
                self._source_iter = None
                return
            self._source_backlog.extend(self._produce_source_packets(in_packet))

    def _drain_source_streams(self):
        self._pump_source_streams(None)
        for kind, out_stream, processor in self._source_pipes.values():
            if kind != "transcode":
                continue
            packets = []
            for rframe in processor.resample(None):
                packets.extend(out_stream.encode(rframe))
            packets.extend(out_stream.encode(None))
            for packet in packets:
                self._mux_source_packet(packet)

    def _mux_source_packet(self, packet):
        if packet.dts is not None:
            last = self._last_source_dts.get(packet.stream.index)
            if last is not None:
                time_base = Fraction(packet.time_base or packet.stream.time_base)
                minimum_time = (last[0] + 1) * last[1]
                if packet.dts * time_base < minimum_time:
                    first_nudge = packet.stream.index not in self._warned_source_dts
                    self._warned_source_dts.add(packet.stream.index)
                    logger.log(
                        logging.WARNING if first_nudge else logging.DEBUG,
                        "Source DTS %s at %s overlaps last muxed DTS %s at %s in output stream %s; nudging forward",
                        packet.dts,
                        time_base,
                        last[0],
                        last[1],
                        packet.stream.index,
                    )
                    packet.dts = math.ceil(minimum_time / time_base)
                    if packet.pts is not None and packet.pts < packet.dts:
                        packet.pts = packet.dts
        self.dst.mux(packet)
        if packet.dts is not None:
            self._last_source_dts[packet.stream.index] = (
                packet.dts,
                Fraction(packet.time_base or packet.stream.time_base),
            )

    def _clamp_pts_monotonic(self, pts: int) -> int:
        last = self._last_emitted_pts
        if last is not None and pts <= last:
            pts = last + 1
        self._last_emitted_pts = pts
        return pts

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (ENCODE_BUFFER_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            pts_to_assign = self._clamp_pts_monotonic(pts_to_assign)
            self._encode_queue.put(
                self._build_encode_item(frame_to_encode, pts_to_assign, self._lut_flags.popleft())
            )

    def _encoder_open_error(self, exc: Exception) -> RuntimeError:
        message = (
            f"Failed to open {self.codec} encoder ({self.encoder_name}) for "
            f"'{self.output_path.suffix}' output on {device_name(self.device)}: {exc}"
        )
        if self.codec == "av1":
            backend = {
                CodecBackend.AMD: "AMF",
                CodecBackend.QSV: "QSV",
                CodecBackend.CPU: "software",
            }.get(self.codec_backend, "NVENC")
            if self.codec_backend is CodecBackend.CPU:
                message += (
                    ". The software AV1 encoder is unavailable in the bundled "
                    "FFmpeg libraries."
                )
            else:
                message += (
                    f". AV1 {backend} encoding requires a GPU/driver generation "
                    "that provides it; pick another codec or the CPU codec backend."
                )
        elif self.codec_backend is CodecBackend.NVIDIA:
            message += (
                ". This GPU reports no usable NVENC encoder; if it has no video "
                "engines (an NVIDIA CMP mining card, for example), select the "
                "Intel Quick Sync or CPU codec backend instead."
            )
        elif self.codec_backend is CodecBackend.QSV:
            message += (
                ". Intel Quick Sync could not be initialised; make sure the "
                "integrated GPU is enabled and its driver installed, or select "
                "another codec backend."
            )
        return RuntimeError(message)

    def _packed_frame(self, height: int, width: int) -> torch.Tensor:
        # NVENC takes the device frame by pointer and still owns it after
        # encode() returns, so NVIDIA hands it a fresh one every time. AMF has
        # no zero-copy path: the AMD branch synchronizes then blocking-copies
        # this buffer into pinned host memory, so one buffer can serve every frame.
        if self._packed is not None:
            return self._packed
        return torch.empty(
            (height + height // 2, width),
            dtype=self._converter.sample_dtype,
            device=self.device,
        )

    def _to_yuv(self, frame: torch.Tensor, height: int) -> torch.Tensor:
        # Sharpening happens here rather than after, because it must see a
        # contiguous plane: pitch alignment can hand back a strided view into a
        # wider buffer, and the AMD path copies straight to host memory.
        packed = self._packed_frame(height, frame.shape[2])
        if self._cas is None:
            self._converter.convert_into(frame, packed[:height], packed[height:])
            return packed
        # CAS is a 3x3 stencil, so it cannot run in place: the conversion writes
        # luma to a scratch plane and sharpening reads from there into the frame
        # the encoder receives, instead of copying a whole plane back.
        luma = self._cas_luma
        if luma is None:
            luma = torch.empty_like(packed[:height])
        self._converter.convert_into(frame, luma, packed[height:])
        self._cas.sharpen_into(luma, packed[:height])
        return packed

    def _encode_frame(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        height = self.metadata.video_height
        with stream_context(self.stream):
            if apply_lut and self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            packed = self._to_yuv(frame, height)
            if self.codec_backend is CodecBackend.NVIDIA:
                packed = _align_yuv_pitch(packed)
                if self.spec.frame_format == "p010le":
                    planes = [
                        packed[:height].view(torch.uint16),
                        packed[height:].view(torch.uint16),
                    ]
                else:
                    planes = [packed[:height], packed[height:]]
                hw_frame = av.VideoFrame.from_dlpack(
                    planes,
                    format=self.spec.frame_format,
                    stream=self.stream.cuda_stream,
                    cuda_context=self._cuda_ctx,
                )

        if _is_host_frame_backend(self.codec_backend):
            # QSV, AMF and the software encoders all read host planes. Reading
            # them while a non-blocking copy was still in flight corrupted
            # frames (issue #252): finish the conversion, then blocking-copy
            # into pinned memory.
            self.stream.synchronize()
            self._host_yuv.copy_(
                packed.view(torch.uint16) if self.spec.ten_bit else packed,
                non_blocking=False,
            )
            hw_frame = self._ffmpeg_owned_frame(height)
        hw_frame.pts = pts
        hw_frame.time_base = self.metadata.time_base
        try:
            packets = self.out_stream.encode(hw_frame)
        except av.FFmpegError as exc:
            if not self._video_started:
                raise self._encoder_open_error(exc) from exc
            raise
        for packet in packets:
            self._mux_video(packet)

    def _ffmpeg_owned_frame(self, height: int) -> av.VideoFrame:
        """Copy the packed host buffer into a frame FFmpeg owns.

        The GPU conversion always writes the packed NV12/P010 layout, and that
        layout is what the encoders and PyAV agree on. Copying into an
        FFmpeg-allocated frame matters for two reasons:

        * Encoders keep a reference to the frame they are handed (frame
          threading in libx265/libsvtav1, and the QSV/AMF internal queues), so
          giving them a view of a buffer that gets overwritten for the next
          frame lets them read torn data — which showed up as intermittent
          bursts of corrupted output frames.
        * libx265 and libsvtav1 only accept planar YUV, and the copy is where
          the packed chroma is de-interleaved.
        """
        width = int(self.metadata.video_width)
        sample_bytes = 2 if self.spec.ten_bit else 1
        chroma = self._host_yuv[height:]
        if self.spec.planar_chroma:
            # Go through the packed frame (a layout both sides agree on) and let
            # swscale produce the planar one; hand-rolled de-interleaving into
            # the plane buffers measured as corrupt.
            packed_format = "p010le" if self.spec.ten_bit else "nv12"
            packed_frame = av.VideoFrame.from_dlpack(
                _host_planes(self._host_yuv, height, planar=False, ten_bit=self.spec.ten_bit),
                format=packed_format,
            )
            return packed_frame.reformat(
                format=self.spec.frame_format, width=width, height=height
            )

        frame = av.VideoFrame(width, height, format=self.spec.frame_format)
        planes = frame.planes
        planes[0].update(
            _plane_payload(
                self._host_yuv[:height], width, sample_bytes, planes[0].line_size, planes[0].buffer_size
            )
        )
        planes[1].update(
            _plane_payload(chroma, width, sample_bytes, planes[1].line_size, planes[1].buffer_size)
        )
        return frame

    def encode(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        if self._worker_error is not None:
            raise self._worker_error
        pts = int(pts) - self.pts_origin
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self._lut_flags.append(bool(apply_lut))
        self.pts_set.add(pts)
        self._process_buffer()
