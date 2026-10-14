import logging
import os
import sys
from fractions import Fraction
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange, VideoReformatter

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    new_stream,
    stream_context,
    vendor_for_device,
)
from jasna.media.codec_backend import decode_family, resolve_decode_backend
from jasna.media.container_utils import demux_video
from jasna.media.probe import VideoMetadata, resolve_video_start_pts
from jasna.media.cuda_kernel import create_stream, destroy_stream
from jasna.media.yuv_to_rgb import YuvToRgbConverter

log = logging.getLogger(__name__)

CORRUPT_PACKET_TOLERANCE = 10

# Decode backend selection through `JASNA_DECODE_BACKEND`:
# - "auto":    NVIDIA tries VALI first and falls back to PyAV hwaccel, then PyAV
#              software, when VALI cannot open or decode the first frame. AMD
#              keeps its AMF -> software escalation.
# - "vali":    VALI only; any failure raises (NVIDIA only).
# - "pyav-hw": skip VALI, use the PyAV hwaccel path with its software fallback.
# - "pyav-sw": force FFmpeg software decoding with GPU upload on every vendor.
DECODE_BACKEND_ENV = "JASNA_DECODE_BACKEND"
_DECODE_BACKENDS = ("auto", "vali", "pyav-hw", "pyav-sw")

# PyAV's avcodec_find_decoder returns libdav1d for AV1, which carries no NVDEC
# hwaccel config, so av.open silently decodes AV1 in software. Force the native
# FFmpeg av1 decoder, which does carry the CUDA hwaccel config. Keyed by codec
# name whose default PyAV decoder lacks NVDEC (only AV1 today).
_NVDEC_DECODER_OVERRIDES = {"av1": "av1"}
_NVDEC_MIN_CODED_SIZE = {"av1": (128, 128)}

# Set JASNA_QSV_HW_DECODE=0 to keep the QSV backend's decode in software.
QSV_HW_DECODE_ENV = "JASNA_QSV_HW_DECODE"

# Diagnostics switch: device-wide synchronise before writing into a freshly
# allocated decode block. See _flush_pending_writes.
DECODE_ALLOC_SYNC = os.environ.get("JASNA_DECODE_ALLOC_FLUSH", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _flush_pending_writes(device: torch.device) -> None:
    """Wait for every stream before writing into a freshly allocated block.

    The decode output tensor is a fresh ``torch.empty`` from the caching
    allocator, so the allocator is free to hand back a block that another
    pipeline component (the TensorRT restorer, the crop/blend buffers) has just
    released. If that component still has kernels in flight writing into the
    block, and it never called ``record_stream`` for them, those writes land in
    our frame *after* the allocator gave it to us.

    The result is exactly the intermittent corruption seen on hosts where
    decoding has to go through this path: whole frames replaced with noise, or a
    restored patch sitting on garbage, in bursts that only appear while
    restoration is running. Flushing once per group costs a few percent of
    throughput and removes the whole class of failure, because every write
    queued before our allocation has then already happened.

    Set ``JASNA_DECODE_ALLOC_FLUSH=0`` to skip it (diagnostics only).
    """
    if DECODE_ALLOC_SYNC:
        torch.cuda.synchronize(device)

# Pixel-format names FFmpeg uses for decoded surfaces that still live in device
# memory. Anything here has to be copied to system memory before the CUDA upload
# kernels can read it.
_HW_FRAME_FORMATS = frozenset(
    {
        "cuda",
        "d3d11",
        "d3d11va",
        "d3d12",
        "qsv",
        "vaapi",
        "dxva2",
        "vulkan",
        "amf",
        "opencl",
        "mediacodec",
        "videotoolbox",
    }
)


def _download_hw_frame(frame, target_format: str):
    """Copy a hardware-decoded surface into system memory.

    PyAV performs ``av_hwframe_transfer_data`` inside ``to_ndarray``, so that is
    the only way to get at a hardware surface's pixels. ``target_format`` must be
    a format PyAV can convert to; only NV12 is supported for 8-bit output (there
    is no P010 numpy path), which is why 10-bit sources stay in software.
    """
    array = frame.to_ndarray(format=target_format)
    return av.VideoFrame.from_ndarray(array, format=target_format)


def _downloading(frames, target_format: str):
    """Yield ``frames`` with any hardware surface copied to system memory."""
    for frame in frames:
        if frame.format.name in _HW_FRAME_FORMATS:
            frame = _download_hw_frame(frame, target_format)
        yield frame


# Remembered per process: once D3D11VA/QSV decoding has been observed falling
# back to software, later readers skip the attempt instead of re-paying it (and
# re-printing FFmpeg's failure notice) for every video.
_qsv_hw_decode_failed = False


def _note_qsv_hw_decode_failed() -> None:
    global _qsv_hw_decode_failed
    _qsv_hw_decode_failed = True


def qsv_hw_decode_unavailable() -> bool:
    return _qsv_hw_decode_failed


class VideoDecodeError(RuntimeError):
    pass


def _decode_backend() -> str:
    backend = os.environ.get(DECODE_BACKEND_ENV, "auto")
    if backend not in _DECODE_BACKENDS:
        raise ValueError(
            f"Unknown decode backend {backend!r} from {DECODE_BACKEND_ENV}, "
            f"expected {_DECODE_BACKENDS}"
        )
    return backend


def _cuda_hwaccel(device: torch.device) -> HWAccel:
    hwaccel = HWAccel(
        "cuda",
        device=str(device.index or 0),
        allow_software_fallback=True,
        is_hw_owned=True,
    )
    # Reuse torch's current primary context without changing its scheduling flags.
    hwaccel.options["primary_ctx"] = "0"
    hwaccel.options["current_ctx"] = "1"
    return hwaccel


def _create_blocking_cuda_stream(device: torch.device) -> tuple[int, torch.cuda.ExternalStream]:
    # cuStreamCreate needs a current CUDA context; threads other than the one
    # torch initialized on have none until torch binds them to the device.
    torch.cuda.set_device(device)
    handle = create_stream()
    return handle, torch.cuda.ExternalStream(handle, device=device)


class _ValiFrameSource:
    """NVDEC decoding through python_vali, converted by the shared CUDA kernel.

    Construction is the whole VALI viability check: it opens the decoder and
    decodes the first frame, so any container/codec VALI cannot handle raises
    here, before the caller commits to this backend. Corrupt packets after that
    are skipped inside the VALI fork (up to its internal consecutive-packet
    tolerance) and surface only as recovery messages; exceeding the tolerance
    raises VideoDecodeError like the PyAV path.
    """

    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        frame_stride: int,
    ):
        import python_vali as vali

        if not hasattr(vali.PyDecoder, "DecodeSingleSurfaceAsyncDetailed"):
            raise VideoDecodeError(
                "python_vali build lacks DecodeSingleSurfaceAsyncDetailed; "
                "the corruption-tolerant fork is required"
            )
        self._vali = vali
        self.file = file
        self.batch_size = batch_size
        self.device = device
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.decoder = None
        self.surface = None
        self._raw_stream, self.stream = _create_blocking_cuda_stream(device)
        try:
            self.decoder = vali.PyDecoder(
                file,
                {},
                gpu_id=device.index or 0,
                stream=self.stream.cuda_stream,
            )
            if not self.decoder.IsAccelerated:
                raise VideoDecodeError(f"VALI decoder for {file} is not hardware accelerated")
            fmt = self.decoder.Format
            if fmt == vali.PixelFormat.NV12:
                self.is_10bit = False
            elif fmt == vali.PixelFormat.P10:
                self.is_10bit = True
            else:
                raise VideoDecodeError(
                    f"VALI surface format {fmt.name} of {file} has no CUDA conversion"
                )
            self.width = self.decoder.Width
            self.height = self.decoder.Height
            self.surface = vali.Surface.Make(
                format=fmt, width=self.width, height=self.height, gpu_id=device.index or 0
            )
            plane = self.surface.Planes[0]
            self._pitch = plane.Pitch
            self._y_ptr = plane.GpuMem
            self._uv_ptr = plane.GpuMem + plane.Pitch * self.height
            self._pkt_data = vali.PacketData()
            self._first_pts = self._decode_next(None)
        except BaseException:
            self.close()
            raise

    def _decode_next(self, seek_ctx) -> int | None:
        success, details = self.decoder.DecodeSingleSurfaceAsyncDetailed(
            self.surface, self._pkt_data, seek_ctx
        )
        if success:
            if details.message:
                log.warning("Recovered video corruption in %s: %s", self.file, details.message)
            return self._pkt_data.pts
        if details.info == self._vali.TaskExecInfo.END_OF_STREAM:
            return None
        raise VideoDecodeError(
            f"Failed to decode {self.file} ({details.info.name}): "
            f"{details.message or details.info.name}"
        )

    def frames(self, seek_ts: float | None) -> Iterator[tuple[torch.Tensor, list[int]]]:
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self.metadata.color_range == AvColorRange.JPEG,
            self.is_10bit,
            self.device,
        )
        # VALI seeks in absolute stream seconds while the reader contract is
        # seconds past the stream start, so shift by the start offset. VALI's
        # own StartTime is unreliable (wrong scale), use the probed metadata.
        seek_ctx = None
        if seek_ts is not None:
            start_seconds = float(
                self.metadata.start_pts * self.metadata.time_base
            )
            seek_ctx = self._vali.SeekContext(seek_ts=seek_ts + start_seconds)
        pending_pts = self._first_pts if seek_ctx is None else None
        exhausted = seek_ctx is None and self._first_pts is None
        self._first_pts = None
        frame_index = 0
        while not exhausted:
            batch = torch.empty(
                (self.batch_size, 3, self.height, self.width),
                device=self.device,
                dtype=torch.uint8,
            )
            pts: list[int] = []
            while len(pts) < self.batch_size:
                if pending_pts is not None:
                    frame_pts = pending_pts
                    pending_pts = None
                else:
                    frame_pts = self._decode_next(seek_ctx)
                    seek_ctx = None
                    if frame_pts is None:
                        exhausted = True
                        break
                selected = frame_index % self.frame_stride == 0
                frame_index += 1
                if not selected:
                    continue
                converter.convert_surface_into(
                    self._y_ptr,
                    self._uv_ptr,
                    self._pitch,
                    batch[len(pts)],
                    self.stream.cuda_stream,
                )
                pts.append(frame_pts)
            if pts:
                self.stream.synchronize()
                yield batch[: len(pts)], pts

    def close(self) -> None:
        self.decoder = None
        self.surface = None
        if self._raw_stream is None:
            return
        raw_stream, self._raw_stream = self._raw_stream, None
        destroy_stream(raw_stream)


class VideoReader:
    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        frame_stride: int = 1,
        decode_device: object | None = None,
    ):
        frame_stride = int(frame_stride)
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        self.device = device
        self.file = file
        self.batch_size = batch_size
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.vendor = vendor_for_device(device)
        # Which device decodes. Defaults to the process/GUI decode device; when
        # that is "auto" the choice follows the encoder so both halves of the
        # pipeline sit on one device, and otherwise the compute GPU.
        self.codec_backend = resolve_decode_backend(
            decode_device,
            compute_vendor=str(self.vendor),
            codec=str(metadata.codec_name).lower() or None,
        )
        self._decoder_ctx = None
        self._vali_source: _ValiFrameSource | None = None
        self._software_only = False

    def _decode_family(self, backend: str) -> str:
        if backend == "pyav-sw":
            return "software"
        return decode_family(self.codec_backend, compute_vendor=str(self.vendor))

    def _qsv_uses_hw_decode(self) -> bool:
        """Whether the QSV backend should also decode on the integrated GPU.

        PyAV can only transfer 8-bit hardware frames back to system memory
        (``to_ndarray`` has no P010 path), so 10-bit sources always stay on the
        FFmpeg software decoder. Set ``JASNA_QSV_HW_DECODE=0`` to force software
        decoding for every source.
        """
        if bool(self.metadata.is_10bit):
            return False
        if qsv_hw_decode_unavailable():
            return False
        value = os.environ.get(QSV_HW_DECODE_ENV, "1")
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _qsv_hwaccel() -> HWAccel:
        # On Windows PyAV's own "qsv" hardware device never engages (the
        # D3D11-backed QSV device fails to create and everything silently stays
        # in software); plain D3D11VA does engage and drives the same Intel video
        # engine. Elsewhere the qsv device works directly.
        device_type = "d3d11va" if sys.platform == "win32" else "qsv"
        # is_hw_owned=True is what actually keeps frames on the device; the
        # transfer back to system memory happens in _frames_software.
        return HWAccel(device_type, allow_software_fallback=True, is_hw_owned=True)

    def __enter__(self):
        current_stream(self.device)
        backend = _decode_backend()
        family = self._decode_family(backend)
        if family == "cuda" and backend in ("auto", "vali"):
            if self.vendor is AcceleratorVendor.NVIDIA:
                try:
                    self._vali_source = _ValiFrameSource(
                        self.file,
                        self.batch_size,
                        self.device,
                        self.metadata,
                        self.frame_stride,
                    )
                except (ImportError, RuntimeError, ValueError, VideoDecodeError) as exc:
                    if backend == "vali":
                        raise VideoDecodeError(f"VALI cannot decode {self.file}: {exc}") from exc
                    log.warning(
                        "VALI cannot decode %s (codec %s): %s; falling back to PyAV",
                        self.file,
                        self.metadata.codec_name,
                        exc,
                    )
                else:
                    self.width = self._vali_source.width
                    self.height = self._vali_source.height
                    log.info("Using VALI NVDEC decoder for %s", self.file)
                    return self
            elif backend == "vali":
                raise VideoDecodeError("The VALI decode backend requires an NVIDIA device")
        software_only = backend == "pyav-sw" or family == "software"
        qsv_hw = family == "qsv" and self._qsv_uses_hw_decode()
        if family == "qsv" and not qsv_hw:
            log.info(
                "Decoding %s in software; the Intel Quick Sync backend encodes on the "
                "iGPU (%s)",
                self.file,
                "10-bit source" if self.metadata.is_10bit else "hardware decode disabled",
            )
        self._software_only = software_only
        try:
            if software_only:
                self.container = av.open(self.file)
            elif family == "cuda":
                self.container = av.open(self.file, hwaccel=_cuda_hwaccel(self.device))
            elif qsv_hw:
                self.container = av.open(self.file, hwaccel=self._qsv_hwaccel())
            else:
                self.container = av.open(self.file)
            self.video_stream = self.container.streams.video[0]
        except av.FFmpegError as e:
            if family == "qsv":
                log.warning(
                    "Intel Quick Sync cannot open %s (codec %s): %s; using FFmpeg "
                    "software decoding and uploading frames to CUDA",
                    self.file,
                    self.metadata.codec_name,
                    e,
                )
                self.container = av.open(self.file)
                self.video_stream = self.container.streams.video[0]
                family = "software"
                software_only = True
                self._software_only = True
            else:
                raise VideoDecodeError(f"Failed to open {self.file}: {e}") from e
        self.decode_family = family

        ctx = self.video_stream.codec_context
        if software_only:
            ctx.thread_type = "AUTO"
        elif family == "amf":
            self._setup_amf_decoder(ctx)
        elif family == "qsv":
            if qsv_hw and not ctx.is_hwaccel:
                # AV1's default PyAV decoder (libdav1d) carries no hardware
                # config, so the hwaccel would be dropped silently.
                self._setup_hw_decode_override(ctx, hwaccel=self._qsv_hwaccel(), label="QSV")
            elif not qsv_hw:
                ctx.thread_type = "AUTO"
        elif not ctx.is_hwaccel:
            if family == "cuda":
                self._setup_hw_decode_override(
                    ctx, hwaccel=_cuda_hwaccel(self.device), label="NVDEC"
                )
            else:
                # Definite software decode: let FFmpeg pick frame/slice threading.
                # CUDA contexts must keep their default threading configuration.
                ctx.thread_type = "AUTO"
        self.width = ctx.width
        self.height = ctx.height
        self._full_range = (
            ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )
        self._raw_stream: int | None = None
        return self

    @property
    def start_pts(self) -> int:
        if self._vali_source is not None:
            return self.metadata.start_pts
        return resolve_video_start_pts(
            self.video_stream.start_time,
            self.metadata.start_pts,
        )

    def _setup_amf_decoder(self, source_ctx) -> None:
        decoder_name = {
            "h264": "h264_amf",
            "hevc": "hevc_amf",
            "av1": "av1_amf",
        }.get(str(source_ctx.name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        try:
            hwaccel = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                is_hw_owned=False,
            )
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            # PyAV 18 rejects assigning time_base on a decoder ("Cannot access
            # 'time_base' as a decoder"); decoders take timing from packets.
            # PyAV returns None for unset SAR/framerate rationals (0/0, 0/1),
            # and its setter then crashes on None.numerator. Assigning None
            # would crash, so fall back to square pixels (1:1) for streams
            # without a declared sample aspect ratio.
            if source_ctx.framerate is not None:
                decoder.framerate = source_ctx.framerate
            if source_ctx.sample_aspect_ratio is not None:
                decoder.sample_aspect_ratio = source_ctx.sample_aspect_ratio
            else:
                decoder.sample_aspect_ratio = Fraction(1, 1)
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            log.info("Using AMF hardware decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "AMF cannot decode %s (codec %s): %s; using FFmpeg software "
                "decoding and uploading frames to ROCm",
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def _setup_hw_decode_override(self, source_ctx, *, hwaccel: HWAccel, label: str) -> None:
        """Force a native FFmpeg decoder that carries the hardware config.

        PyAV's default decoder for a codec can lack the hardware hwaccel config
        (libdav1d for AV1), so ``av.open`` silently decodes in software. Only
        the codecs in ``_NVDEC_DECODER_OVERRIDES`` need this.
        """
        decoder_name = _NVDEC_DECODER_OVERRIDES.get(str(self.metadata.codec_name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        min_width, min_height = _NVDEC_MIN_CODED_SIZE[decoder_name]
        if source_ctx.width < min_width or source_ctx.height < min_height:
            source_ctx.thread_type = "AUTO"
            log.info(
                "Skipping %s decoder %s for %s: %dx%d is below its %dx%d minimum",
                label,
                decoder_name,
                self.file,
                source_ctx.width,
                source_ctx.height,
                min_width,
                min_height,
            )
            return
        try:
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            if source_ctx.pix_fmt is not None:
                decoder.pix_fmt = source_ctx.pix_fmt
            decoder.profile = source_ctx.profile
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            log.info("Using %s decoder %s for %s", label, decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "%s decoder %s unavailable for %s (codec %s): %s; using FFmpeg "
                "software decoding and uploading frames to CUDA",
                label,
                decoder_name,
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        if self._vali_source is not None:
            source, self._vali_source = self._vali_source, None
            source.close()
            return
        self.container.close()
        self._decoder_ctx = None
        if self._raw_stream is None:
            return
        try:
            destroy_stream(self._raw_stream)
        except RuntimeError:
            if exc_type is None:
                raise
            log.warning("Could not destroy the decode stream while handling another error", exc_info=True)

    def _decode_packet(self, packet, consecutive_errors: int) -> tuple[list, int]:
        try:
            frames = (
                self._decoder_ctx.decode(packet)
                if self._decoder_ctx is not None
                else packet.decode()
            )
        except av.error.InvalidDataError as e:
            consecutive_errors += 1
            if consecutive_errors > CORRUPT_PACKET_TOLERANCE:
                raise VideoDecodeError(
                    f"Failed to decode {self.file}: too many consecutive corrupt packets "
                    f"({consecutive_errors}): {e}"
                ) from e
            log.warning("Recovered video corruption in %s: %s", self.file, e)
            return [], consecutive_errors
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
        if frames:
            consecutive_errors = 0
        return frames, consecutive_errors

    def _decoded_frames(self, seek_ts: float | None):
        target_pts = None
        if seek_ts is not None:
            start = resolve_video_start_pts(
                self.video_stream.start_time,
                self.metadata.start_pts,
            )
            target_pts = start + round(seek_ts / self.video_stream.time_base)
            self.container.seek(target_pts, stream=self.video_stream, backward=True)
            if self._decoder_ctx is not None:
                self._decoder_ctx.flush_buffers()

        consecutive_errors = 0
        for packet in demux_video(self.container, self.video_stream):
            frames, consecutive_errors = self._decode_packet(packet, consecutive_errors)
            for frame in frames:
                if target_pts is not None and frame.pts is not None and frame.pts < target_pts:
                    continue
                target_pts = None
                yield frame

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group

    def _selected_frames(self, decoded):
        if self.frame_stride == 1:
            yield from decoded
            return
        for frame_index, frame in enumerate(decoded):
            if frame_index % self.frame_stride == 0:
                yield frame

    def frames(
        self,
        seek_ts: float | None = None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # With seek_ts, strided selection re-anchors at the first decoded frame
        # after the seek instead of the start of the file: sample phase is only
        # stable relative to the seek target.
        if self._vali_source is not None:
            yield from self._vali_source.frames(seek_ts)
            return
        # The first decoded frame's format is the final backend decision: a codec
        # can advertise a CUDA config and still fall back to software when
        # hardware initialization rejects a profile or pixel format. Dispatch
        # once here so neither per-frame loop carries a backend branch.
        decoded = self._selected_frames(self._decoded_frames(seek_ts))
        group = self._read_group(decoded)
        if not group:
            return
        family = getattr(self, "decode_family", None) or self._decode_family(
            _decode_backend()
        )
        if family == "cuda" and group[0].format.name == "cuda":
            backend = self._frames_hardware(decoded, group)
        else:
            if family == "cuda" and not self._software_only:
                log.warning(
                    "CUDA/NVDEC cannot decode %s (codec %s, %s); using FFmpeg "
                    "software decoding and uploading frames to CUDA",
                    self.file,
                    self.metadata.codec_name,
                    group[0].format.name,
                )
            elif family == "qsv" and not self._software_only:
                if group[0].format.name in _HW_FRAME_FORMATS:
                    log.info(
                        "Intel Quick Sync decoded %s (codec %s) to system memory; "
                        "uploading frames to CUDA",
                        self.file,
                        self.metadata.codec_name,
                    )
                else:
                    _note_qsv_hw_decode_failed()
                    log.info(
                        "Quick Sync hardware decoding is unavailable for %s (codec %s); "
                        "decoded in software and uploaded to CUDA. Encoding still uses "
                        "Quick Sync.",
                        self.file,
                        self.metadata.codec_name,
                    )
            backend = self._frames_software(decoded, group)

        # The backend generator now owns the first group. Drop this outer
        # reference before yielding: retaining four 4K P010 NVDEC surfaces here
        # for the reader's lifetime costs about 96 MiB of avoidable VRAM.
        del group
        yield from backend

    def _frames_hardware(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # FFmpeg 8 maps NVDEC output on CUDA stream 0. Conversion runs in a
        # blocking stream in that same context, so legacy-default-stream ordering
        # makes the decoded writes visible before this kernel without a race.
        # Decode one group ahead while conversion runs; keep both groups' frame
        # references alive until conversion is synchronized so their mapped
        # surfaces cannot be recycled underneath queued work.
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            self.metadata.is_10bit,
            self.device,
        )
        if self._raw_stream is None:
            self._raw_stream, self.stream = _create_blocking_cuda_stream(self.device)
        while group:
            batch = torch.empty(
                (len(group), 3, self.height, self.width), device=self.device, dtype=torch.uint8
            )
            _flush_pending_writes(self.device)
            pts = [frame.pts for frame in group]
            with torch.cuda.stream(self.stream):
                converter.convert_frames_into(group, batch, self.stream.cuda_stream)

            next_group = self._read_group(decoded)
            self.stream.synchronize()
            group = next_group
            yield batch, pts

    def _frames_software(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # Normalize CPU frames to the two layouts the CUDA conversion kernel
        # accepts (NV12 for <=8-bit sources, P010 above), keeping the resolved
        # matrix/range identical on both reformat sides so swscale changes only
        # layout/subsampling/depth. The one authoritative YUV->RGB conversion
        # stays in the CUDA kernel.
        hw_target = "p010le" if self.metadata.is_10bit else "nv12"
        if group and group[0].format.name in _HW_FRAME_FORMATS:
            group = [_download_hw_frame(frame, hw_target) for frame in group]
            decoded = _downloading(decoded, hw_target)
        depth = max(
            (component.bits for component in group[0].format.components if component.bits),
            default=10 if self.metadata.is_10bit else 8,
        )
        ten_bit = depth > 8
        if depth > 10:
            log.warning(
                "Reducing %d-bit source %s to 10-bit P010 before CUDA upload", depth, self.file
            )
        target_format = "p010le" if ten_bit else "nv12"
        dtype = torch.uint16 if ten_bit else torch.uint8
        bytes_per_sample = 2 if ten_bit else 1

        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            ten_bit,
            self.device,
        )
        reformatter = VideoReformatter()
        color_range = AvColorRange.JPEG if self._full_range else AvColorRange.MPEG
        H, W = self.height, self.width

        # Pinned host batch is shared. Device staging is gated on
        # AcceleratorVendor.AMD (not "if not NVIDIA") so Intel/CPU keep the
        # historical single-staging private-stream fallback unless AMD-specific.
        # NVIDIA: one staging frame on a private stream (H2D+convert ordered so
        # the next overwrite starts only after the prior kernel consumed it).
        # AMD (issue #252): batch device YUV on current_stream. Isolated D1 was
        # clean; residual glitches are pipeline-contended (Phase 0). Per-frame
        # staging reuse under multi-thread load can race; batch staging removes
        # overwrite races. Phase 4 on gfx1201 cleared full-pipeline static.
        # Extra VRAM ≈ (batch_size - 1) × (1.5 × H × W × bytes) — a few MiB at
        # batch 4 @ 1080p8.
        pinned = torch.empty((self.batch_size, H + H // 2, W), dtype=dtype, pin_memory=True)
        if self.vendor is AcceleratorVendor.AMD:
            device_yuv = torch.empty(
                (self.batch_size, H + H // 2, W), dtype=dtype, device=self.device
            )
            staging = None
            stream = current_stream(self.device)
        else:
            device_yuv = None
            staging = torch.empty((H + H // 2, W), dtype=dtype, device=self.device)
            stream = new_stream(self.device)

        while group:
            batch = torch.empty((len(group), 3, H, W), device=self.device, dtype=torch.uint8)
            _flush_pending_writes(self.device)
            pts = [frame.pts for frame in group]
            for i, frame in enumerate(group):
                try:
                    normalized = reformatter.reformat(
                        frame,
                        width=W,
                        height=H,
                        format=target_format,
                        src_colorspace=self.metadata.color_space,
                        dst_colorspace=self.metadata.color_space,
                        src_color_range=color_range,
                        dst_color_range=color_range,
                    )
                except av.FFmpegError as e:
                    raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
                y_plane, uv_plane = normalized.planes
                y = torch.frombuffer(y_plane, dtype=dtype).reshape(
                    H, y_plane.line_size // bytes_per_sample
                )[:, :W]
                uv = torch.frombuffer(uv_plane, dtype=dtype).reshape(
                    H // 2, uv_plane.line_size // bytes_per_sample
                )[:, :W]
                pinned[i, :H].copy_(y)
                pinned[i, H:].copy_(uv)

            with stream_context(stream):
                # Shared copy/convert loop: AMD uses a per-frame plane from the
                # batch device buffer; NVIDIA reuses the single staging frame
                # (H2D + convert ordered on the private stream so the next
                # overwrite starts only after the prior kernel consumed it).
                for i in range(len(group)):
                    plane = staging if staging is not None else device_yuv[i]
                    plane.copy_(pinned[i], non_blocking=True)
                    converter.convert_into(
                        plane[:H],
                        plane[H:].view(H // 2, W // 2, 2),
                        batch[i],
                    )

            next_group = self._read_group(decoded)
            # Sync before yield so other pipeline threads never see in-flight planes.
            stream.synchronize()
            group = next_group
            yield batch, pts
