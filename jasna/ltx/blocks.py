"""Transformer block weights, resident on the GPU or streamed from pinned host memory.

Each block is packed into one flat byte buffer so it moves in a single copy. As many
leading blocks as the VRAM budget allows stay resident; the rest live in pinned host
memory and are copied on a side stream into one of two GPU slots, one block ahead of
the block being computed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_ALIGN = 256


@dataclass(frozen=True)
class _Entry:
    offset: int
    dtype: torch.dtype
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count * torch.empty((), dtype=self.dtype).element_size()


def _layout(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, _Entry], int]:
    layout: dict[str, _Entry] = {}
    offset = 0
    for name, tensor in tensors.items():
        entry = _Entry(offset, tensor.dtype, tuple(tensor.shape))
        layout[name] = entry
        offset += -(-entry.nbytes // _ALIGN) * _ALIGN
    return layout, offset


def _pack(tensors: dict[str, torch.Tensor], layout: dict[str, _Entry], buffer: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        entry = layout[name]
        buffer[entry.offset : entry.offset + entry.nbytes].copy_(tensor.contiguous().view(-1).view(torch.uint8))


def _views(buffer: torch.Tensor, layout: dict[str, _Entry]) -> dict[str, torch.Tensor]:
    return {
        name: buffer[entry.offset : entry.offset + entry.nbytes].view(entry.dtype).view(entry.shape)
        for name, entry in layout.items()
    }


_host_pinning_reported = False


def _alloc_host_buffer(nbytes: int) -> torch.Tensor:
    """A host buffer for one streamed block, degrading gracefully.

    Pinned (page-locked) memory makes the side-stream prefetch asynchronous and
    fast, but it is also the one allocation here that the CUDA driver can
    refuse: PyTorch's caching host allocator never returns cached pages to the
    OS on its own, and earlier phases (the video decoder's batch buffers, for
    one) leave plenty cached, so a multi-hundred-MiB request can hit a driver
    or page-lock limit. When that happens, retry once after evicting the host
    cache and, if the driver still refuses, run the streamed blocks from
    pageable memory — the copies then synchronise internally, which is slower
    but perfectly correct, and the job completes instead of failing.
    """
    global _host_pinning_reported
    try:
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    except Exception as first:
        logger.warning(
            "Pinned allocation of %.0f MiB failed (%s); emptying the host cache and retrying",
            nbytes / 2**20,
            first,
        )
    try:
        torch._C._host_emptyCache()
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    except Exception as second:
        if not _host_pinning_reported:
            _host_pinning_reported = True
            logger.warning(
                "Pinned memory still unavailable (%s); LTX streamed blocks will use pageable "
                "host memory. Streams still work - the block copies just synchronise, so "
                "throughput drops a little.",
                second,
            )
        return torch.empty(nbytes, dtype=torch.uint8)


class BlockStore:
    """Hands out each block's weights on ``device`` in a fixed cyclic block order."""

    def __init__(self, blocks: list[dict[str, torch.Tensor]], device: torch.device, *, resident: int) -> None:
        self.device = device
        self.count = len(blocks)
        self.resident = min(resident, self.count)
        self._layouts: list[dict[str, _Entry]] = []
        self._resident: list[dict[str, torch.Tensor]] = []
        self._host: list[torch.Tensor] = []
        # Surfaced before the multi-hundred-MiB allocations below: CUDA errors are
        # per-thread and only reported by the next checking call, so a stale
        # asynchronous failure from an earlier pass shows up as a baffling
        # "resource already mapped" attributed to the pinned allocation. If the
        # context really is poisoned, fail here where the origin is visible.
        torch.cuda.synchronize(self.device)
        slot_bytes = 0
        for index, tensors in enumerate(blocks):
            layout, nbytes = _layout(tensors)
            self._layouts.append(layout)
            if index < self.resident:
                buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
                _pack(tensors, layout, buffer)
                self._resident.append(_views(buffer, layout))
            else:
                buffer = _alloc_host_buffer(nbytes)
                _pack(tensors, layout, buffer)
                self._host.append(buffer)
                slot_bytes = max(slot_bytes, nbytes)
        streamed = self.count - self.resident
        self._slots = [torch.empty(slot_bytes, dtype=torch.uint8, device=device) for _ in range(min(2, streamed))]
        self._slot_block = [-1] * len(self._slots)
        self._copy_done = [torch.cuda.Event() for _ in self._slots]
        self._compute_done = [torch.cuda.Event() for _ in self._slots]
        self._copy_stream = torch.cuda.Stream(device) if self._slots else None
        logger.info("LTX blocks: %d resident, %d streamed", self.resident, streamed)

    def _slot_of(self, index: int) -> int | None:
        return self._slot_block.index(index) if index in self._slot_block else None

    def _prefetch(self, index: int, keep: int | None) -> None:
        if index < self.resident or self._slot_of(index) is not None:
            return
        busy = self._slot_of(keep) if keep is not None else None
        slot = next(s for s in range(len(self._slots)) if s != busy)
        source = self._host[index - self.resident]
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._compute_done[slot])
            self._slots[slot][: source.numel()].copy_(source, non_blocking=True)
            self._copy_done[slot].record(self._copy_stream)
        self._slot_block[slot] = index

    def acquire(self, index: int) -> dict[str, torch.Tensor]:
        """Weights of block ``index``; also starts copying the next streamed block."""
        if index < self.resident:
            weights = self._resident[index]
        else:
            self._prefetch(index, None)
            slot = self._slot_of(index)
            torch.cuda.current_stream(self.device).wait_event(self._copy_done[slot])
            weights = _views(self._slots[slot], self._layouts[index])
        following = (index + 1) % self.count
        if following < self.resident and self.resident < self.count:
            following = self.resident
        if following != index:
            self._prefetch(following, index)
        return weights

    def release(self, index: int) -> None:
        """Mark block ``index``'s slot reusable once the work queued so far is done.
        Call before acquiring the next block, so its prefetch waits for this work."""
        slot = self._slot_of(index) if index >= self.resident else None
        if slot is not None:
            self._compute_done[slot].record(torch.cuda.current_stream(self.device))

    def close(self) -> None:
        self._resident.clear()
        self._host.clear()
        self._slots.clear()
