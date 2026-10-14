from __future__ import annotations

import contextlib
import logging
import os
import re
import warnings

import torch

logger = logging.getLogger(__name__)

_torchtrt_muted = False

# torch_tensorrt routes every TensorRT message (including
# ``IRuntime::deserializeCudaEngine`` failures) to this Python logger. It is
# muted for normal operation, so a failed load has to read the messages back
# from a temporary handler - otherwise the only thing left to report is
# torch's generic "please check the warnings above".
_TRT_LOG_NAME = "torch_tensorrt [TensorRT Conversion Context]"


class EngineArchiveError(RuntimeError):
    """A compiled engine archive exists but cannot be loaded on this GPU.

    TensorRT bakes the compute capability, SM count, driver and TensorRT
    version of the *building* machine into a serialized plan, and a plan built
    for one card cannot run on another (``deserializeCudaEngine`` refuses it).
    Engines copied from another machine - a shipped pre-built archive, or the
    flat folder an older install left behind - therefore look present but are
    unusable.

    Callers treat this as "the engine set for this card is missing": the archive
    is moved aside and rebuilt for the card that is actually installed.
    """

    def __init__(self, path: str, reason: str, *, detail: str = "") -> None:
        self.path = str(path)
        self.reason = reason
        self.detail = detail
        message = f"{path}: {reason}"
        if detail:
            message = f"{message}\n{detail}"
        super().__init__(message)


class _MessageCollector(logging.Handler):
    """Collects TensorRT messages emitted while an engine is being loaded."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - never break logging
            pass


@contextlib.contextmanager
def _capture_trt_messages():
    """Yield a list that collects TensorRT warnings/errors raised inside."""
    trt_logger = logging.getLogger(_TRT_LOG_NAME)
    collector = _MessageCollector()
    previous_level = trt_logger.level
    trt_logger.addHandler(collector)
    if previous_level == logging.NOTSET or previous_level > logging.WARNING:
        trt_logger.setLevel(logging.WARNING)
    try:
        yield collector.messages
    finally:
        trt_logger.removeHandler(collector)
        trt_logger.setLevel(previous_level)


def _mute_torch_tensorrt() -> None:
    global _torchtrt_muted
    if _torchtrt_muted:
        return
    _torchtrt_muted = True
    import tensorrt as trt
    from jasna._frozen import patch_frozen_torch
    patch_frozen_torch()
    import torch_tensorrt
    # The internals these rely on moved between torch_tensorrt releases; every
    # step must be optional so one code path serves all the bundled builds.
    tt_logger = getattr(torch_tensorrt.logging, "_LOGGER", None)
    if tt_logger is not None:
        tt_logger.setLevel(logging.ERROR)
        tt_logger.handlers.clear()
        tt_logger.addHandler(logging.NullHandler())
        tt_logger.propagate = False
    set_level = getattr(getattr(torch.ops, "tensorrt", None), "set_logging_level", None)
    if set_level is not None:
        set_level(int(trt.ILogger.Severity.ERROR))


def get_workspace_size_bytes() -> int:
    free, _total = torch.cuda.mem_get_info()
    return int(free * 0.95)


def _flush_c_streams() -> None:
    """Flush the Python-level streams so torch's own messages are ordered.

    TensorRT's C++ runtime writes straight to the C streams, which the frozen GUI
    points at NUL, so its text is out of reach - hence the message collector, which
    reads TensorRT back through torch_tensorrt's Python logger instead. This only
    makes sure anything torch printed itself (``print(message)`` in the compile
    path) is flushed before the report is written.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # pragma: no cover - a dead stream must not mask the real error
            pass


def _trt_detail(messages: list[str]) -> str:
    """Condense collected TensorRT messages into a short, readable hint."""
    interesting = [
        m for m in messages
        if "deserialize" in m.lower() or "error code" in m.lower() or "error" in m.lower()
    ]
    chosen = interesting or messages
    seen: list[str] = []
    for message in chosen:
        cleaned = message.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned[:300])
    return " | ".join(seen[:3])


_UNUSABLE_REASON = (
    "the stored TensorRT plan cannot be deserialized on this GPU (an engine is only "
    "valid on the card and TensorRT build that produced it), or the file is damaged"
)


def load_torchtrt_export(*, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    _mute_torch_tensorrt()

    path = os.fspath(checkpoint_path)
    logger.debug("Loading TensorRT export from %s", path)

    try:
        if os.path.getsize(path) == 0:
            raise EngineArchiveError(path, "the engine file is empty")
    except OSError as exc:
        raise EngineArchiveError(path, f"the engine file cannot be read ({exc})") from exc

    fake_reg_logger = logging.getLogger("torch._library.fake_class_registry")
    prev_level = fake_reg_logger.level
    fake_reg_logger.setLevel(logging.ERROR)
    try:
        with _capture_trt_messages() as trt_messages:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*PytorchStreamReader.*")
                    if hasattr(torch.export, "load"):
                        try:
                            with open(path, "rb") as f:
                                trt_module = torch.export.load(f).module()
                        except Exception:
                            # Older builds wrote a plain torch.save() pickled module,
                            # which only torch.load can read. Only worth reporting if
                            # that fallback fails too.
                            logger.debug(
                                "torch.export.load failed for %s, falling back to torch.load",
                                path,
                                exc_info=True,
                            )
                            trt_module = torch.load(path, map_location=device, weights_only=False)
                    else:
                        # torch < 2.6 has no torch.export.save/load: archives from the
                        # Volta-era build are plain torch.save() pickled modules.
                        trt_module = torch.load(path, map_location=device, weights_only=False)
            except Exception as exc:
                _flush_c_streams()
                detail = _trt_detail(trt_messages)
                logger.error(
                    "Engine archive unusable: %s\n  reason: %s%s",
                    path,
                    _UNUSABLE_REASON,
                    f"\n  TensorRT said: {detail}" if detail else "",
                    exc_info=True,
                )
                raise EngineArchiveError(path, _UNUSABLE_REASON, detail=detail) from exc

            try:
                return trt_module.to(device)
            except Exception as exc:
                _flush_c_streams()
                detail = _trt_detail(trt_messages)
                reason = "the compiled TensorRT plan failed to initialize on this GPU"
                logger.error(
                    "Engine archive unusable: %s\n  reason: %s%s",
                    path,
                    reason,
                    f"\n  TensorRT said: {detail}" if detail else "",
                    exc_info=True,
                )
                raise EngineArchiveError(path, reason, detail=detail) from exc
    finally:
        fake_reg_logger.setLevel(prev_level)


def uses_legacy_dynamo() -> bool:
    """Whether this stack is the torch 2.5 / torch_tensorrt 2.5 (Volta) one.

    torch 2.5's tracer derives stricter shape guards than later versions — for
    the BasicVSR++ preprocess graph it demands a batch of at least 5 where the
    newer builds happily accept the declared 3 (see
    ``_PreprocessWrapper``). Callers use this to align engine profiles and
    runtime batch floors with what the tracer will actually permit.
    """
    parts = torch.__version__.split("+")[0].split(".")
    try:
        return (int(parts[0]), int(parts[1])) < (2, 6)
    except (IndexError, ValueError):
        return False


# torch._dynamo names the fix it wants in "Suggested fixes: _0 = Dim('_0', min=5, max=60)".
_CONSTRAINT_SUGGESTION = re.compile(
    r"Dim\('(\w+)',\s*min=(\d+),\s*max=(\d+)\)"
)


def _rebuild_inputs_with_suggested_range(torch_tensorrt, inputs: list, error_text: str):
    """Rebuild the first dynamic ``torch_tensorrt.Input`` per dynamo's suggestion.

    Returns ``(new_inputs, note)`` or ``None`` when the error does not carry a
    usable suggestion or no dynamic input can absorb it.
    """
    if "Constraints violated" not in error_text:
        return None
    match = _CONSTRAINT_SUGGESTION.search(error_text)
    if match is None:
        return None
    lo, hi = int(match.group(2)), int(match.group(3))
    for index, inp in enumerate(inputs):
        if not isinstance(inp, torch_tensorrt.Input):
            continue
        spec = inp.shape
        mins, opts, maxs = (
            list(spec["min_shape"]),
            list(spec["opt_shape"]),
            list(spec["max_shape"]),
        )
        for dim, (mn, mx) in enumerate(zip(mins, maxs)):
            if mn == mx or not (mn < lo or mx > hi):
                continue
            if not (lo <= mx <= hi and lo <= opts[dim] <= hi):
                continue  # the suggested range excludes this input's optimum
            mins[dim] = lo
            kwargs = dict(min_shape=mins, opt_shape=opts, max_shape=maxs)
            dtype = getattr(inp, "dtype", None)
            if dtype is not None:
                kwargs["dtype"] = dtype
            rebuilt = list(inputs)
            rebuilt[index] = torch_tensorrt.Input(**kwargs)
            return rebuilt, f"input {index} dim {dim} bounded to {lo}..{mx}"
    return None


def _compile_compat(torch_tensorrt, module: torch.nn.Module, kwargs: dict):
    """``torch_tensorrt.compile`` across every bundled torch_tensorrt release.

    The dynamo keyword set has grown over releases (``optimization_level``,
    ``truncate_double``, ...), and older builds reject unknown keywords with a
    TypeError naming the offending argument. Retry, dropping the named
    arguments, so one code path serves the CUDA 13 build and the Volta-era
    torch 2.5 / torch_tensorrt 2.5 build alike.
    """
    import re

    dropped: list[str] = []
    for _ in range(len(kwargs) + 1):
        try:
            return torch_tensorrt.compile(module, **kwargs)
        except TypeError as exc:
            match = re.search(r"keyword argument ['\"](\w+)['\"]", str(exc))
            if match is None or match.group(1) not in kwargs:
                raise
            dropped.append(match.group(1))
            kwargs.pop(match.group(1))
            logger.debug("torch_tensorrt.compile: dropped unsupported kwarg %s", dropped[-1])
    raise RuntimeError(
        "torch_tensorrt.compile rejected every keyword argument: " + ", ".join(dropped)
    )


def _torch_tensorrt_version(torch_tensorrt) -> tuple[int, int]:
    """The installed torch_tensorrt release as a comparable tuple."""
    raw = str(getattr(torch_tensorrt, "__version__", "0"))
    try:
        parts = raw.split("+")[0].split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


def compile_and_save_torchtrt_dynamo(
    *,
    module: torch.nn.Module,
    inputs: list,
    output_path: str,
    dtype: torch.dtype,
    workspace_size_bytes: int,
    message: str,
    device: torch.device | None = None,
    optimization_level: int = 3,
) -> str:
    """Compile a module to TensorRT and save the result.

    ``inputs`` may be plain tensors (static shapes) or
    ``torch_tensorrt.Input`` specs (dynamic shapes).  Pass ``device``
    explicitly when using ``torch_tensorrt.Input`` objects.
    """
    import torch_tensorrt  # type: ignore[import-not-found]
    _mute_torch_tensorrt()

    has_dynamic = any(isinstance(inp, torch_tensorrt.Input) for inp in inputs)
    if device is None:
        device = inputs[0].device
    print(message)
    logger.info("%s", message)
    with torch.cuda.device(device):
        base_kwargs = dict(
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            workspace_size=int(workspace_size_bytes),
            enabled_precisions={dtype},
            use_fp32_acc=False,
            use_explicit_typing=False,
            sparse_weights=False,
            optimization_level=int(optimization_level),
            hardware_compatible=False,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
            truncate_double=True,
        )
        # Older torch tracers can derive stricter shape guards than the declared
        # input range (the Volta stack demands batch >= 5 for the preprocess
        # graph). Dynamo says what it wants; honour the suggestion and retry.
        trt_gm = None
        for attempt in range(3):
            try:
                trt_gm = _compile_compat(torch_tensorrt, module, dict(base_kwargs))
                break
            except Exception as exc:
                suggestion = (
                    _rebuild_inputs_with_suggested_range(torch_tensorrt, inputs, str(exc))
                    if attempt < 2 else None
                )
                if suggestion is None:
                    raise
                inputs, note = suggestion
                base_kwargs["inputs"] = inputs
                logger.info(
                    "Retrying TensorRT compile with the shape range dynamo suggests (%s)",
                    note,
                )
        fake_reg_logger = logging.getLogger("torch._library.fake_class_registry")
        prev_level = fake_reg_logger.level
        fake_reg_logger.setLevel(logging.ERROR)
        try:
            if _torch_tensorrt_version(torch_tensorrt) < (2, 6):
                # torch_tensorrt 2.5's save() re-exports the compiled module through
                # its own exporter, which trips over a torch 2.5 ExportedProgram
                # pass on real models (KeyError on multi-submodule graphs such as
                # the optical-flow warpers). Pickle the compiled module instead:
                # the engine — including its dynamic-shape range — lives inside the
                # graph, and load_torchtrt_export() reads this back through
                # torch.load.
                torch.save(trt_gm, output_path)
            elif has_dynamic:
                _save_with_dynamic_shapes(trt_gm, output_path, inputs, device, dtype)
            else:
                torch_tensorrt.save(trt_gm, output_path, inputs=inputs)
        finally:
            fake_reg_logger.setLevel(prev_level)
    del trt_gm
    return output_path


def _save_with_dynamic_shapes(
    trt_gm: torch.nn.Module,
    output_path: str,
    inputs: list,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Save a TRT-compiled GraphModule that uses dynamic shapes.

    Builds sample tensors and a ``dynamic_shapes`` spec from
    ``torch_tensorrt.Input`` objects so that ``torch.export.export``
    records the correct symbolic dimension constraints.
    """
    import torch_tensorrt  # type: ignore[import-not-found]
    from torch.export import Dim

    sample_args: list[torch.Tensor] = []
    dyn_shapes: list[dict[int, Dim] | None] = []

    for inp in inputs:
        if isinstance(inp, torch_tensorrt.Input):
            shape_dict = inp.shape
            opt = shape_dict["opt_shape"]
            sample_args.append(torch.randn(*opt, dtype=dtype, device=device))

            min_s, max_s = shape_dict["min_shape"], shape_dict["max_shape"]
            dim_map: dict[int, Dim] = {}
            for d in range(len(opt)):
                if min_s[d] != max_s[d]:
                    dim_map[d] = Dim(f"d{d}", min=int(min_s[d]), max=int(max_s[d]))
            dyn_shapes.append(dim_map if dim_map else None)
        else:
            sample_args.append(inp)
            dyn_shapes.append(None)

    try:
        ep = torch.export.export(
            trt_gm,
            tuple(sample_args),
            dynamic_shapes=tuple(dyn_shapes),
            strict=False,
        )
        torch.export.save(ep, output_path)
    except RuntimeError:
        logger.debug(
            "torch.export.export failed (multi-subgraph dynamic shapes); "
            "falling back to torch.save for %s", output_path,
        )
        torch.save(trt_gm, output_path)
