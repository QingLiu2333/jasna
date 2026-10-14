"""Suppress noisy warnings and log messages from torch / torch_tensorrt.

Call ``install()`` once before importing torch.  The function is idempotent.
"""
from __future__ import annotations

from importlib.util import find_spec
import logging
import warnings

import av

_installed = False


def _add_message_drop_filter(logger_name: str, *needles: str) -> None:
    """Drop any record on ``logger_name`` whose message contains one of ``needles``.
    Filtering on the source logger (not a handler) muting works regardless of which
    handler/formatter is upstream and survives torch._logging resetting logger levels."""

    class _Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return not any(n in message for n in needles)

    logging.getLogger(logger_name).addFilter(_Drop())


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    warnings.filterwarnings(
        "ignore",
        message=r"^To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*",
        category=UserWarning,
        module=r"^torch_tensorrt\.dynamo\.utils$",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^Unable to import quantization op\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^Unable to import quantize op\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^TensorRT-LLM is not installed\..*",
    )
    warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"^Unable to execute the generated python source code from the graph\..*",
        category=UserWarning,
        module=r"^torch\.export\.exported_program$",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule!.*",
        category=UserWarning,
        module=r"^torch_tensorrt\.dynamo\._exporter$",
    )

    av.logging.set_level(logging.ERROR)
    logging.getLogger("torch.export.pt2_archive._package").setLevel(logging.ERROR)
    for _name in (
        "torch_tensorrt",
        "torch_tensorrt._utils",
        "torch_tensorrt [TensorRT Conversion Context]",
    ):
        logging.getLogger(_name).setLevel(logging.ERROR)

    _add_message_drop_filter(
        "torch.distributed.elastic.multiprocessing.redirects", "Redirects are currently not supported"
    )
    _add_message_drop_filter(
        "torch._export.serde.serialize", "did not appear in the graph that was deserialized"
    )

    # torch_tensorrt logs the CUDA/TRT-LLM-plugin note at ERROR level (so the level bump
    # above can't mute it), and torch's flop_counter logs "triton not found" through
    # torch's own glog handler. Both are harmless for jasna — drop them by message.
    _add_message_drop_filter(
        "torch_tensorrt._utils", "is not currently supported for TRT-LLM plugins"
    )
    _add_message_drop_filter("torch.utils.flop_counter", "triton not found")

    # FFmpeg's own logs reach Python through PyAV's "libav.<component>" loggers.
    # Filters only run on the logger the record was created on, so the specific
    # child loggers seen in practice have to be named.
    #
    # * The QSV backend deliberately asks for D3D11VA decoding first and falls
    #   back to software when the device cannot be created (a CUDA context being
    #   live is enough); VideoReader reports which path it took itself.
    # * hevc_qsv/h264_qsv print the CQP note whenever `global_quality` arrives
    #   through the codec options dictionary rather than the AVCodecContext
    #   field. It was measured to work on this path (output size tracks the
    #   value), so the note is misleading rather than actionable.
    for _libav_logger in ("libav.hevc", "libav.h264", "libav.av1", "libav.hevc_qsv", "libav.h264_qsv"):
        _add_message_drop_filter(_libav_logger, "hwaccel initialisation returned error")
        _add_message_drop_filter(_libav_logger, "Using the constant quantization parameter (CQP) by default")
        # NVDEC/D3D11VA rejecting the source because this GPU has no video
        # engine; VideoReader logs the switch to software decoding itself.
        _add_message_drop_filter(_libav_logger, "Hardware is lacking required capabilities")

    # tensorrt.plugin logs an "experimental" warning through TensorRT's C++
    # logger at import time (triggered by torch_tensorrt). It bypasses the
    # logging/warnings machinery, so the only way to mute it is to filter the
    # message in trt.Logger.log before the module is imported.
    if find_spec("tensorrt") is None:
        return

    import tensorrt as trt

    _original_trt_log = trt.Logger.log

    def _trt_log_without_plugin_experimental(self, severity, msg):
        if "tensorrt.plugin module is experimental" in str(msg):
            return
        return _original_trt_log(self, severity, msg)

    trt.Logger.log = _trt_log_without_plugin_experimental
