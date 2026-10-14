"""Verify the LTX BlockStore host-buffer hardening.

Covers three paths of ``_alloc_host_buffer``:

1. Healthy: pinned buffers are used and streamed weights round-trip correctly.
2. Driver refuses pinned memory outright: BlockStore falls back to pageable
   host memory and streamed weights still round-trip correctly.
3. Transient failure: the first pinned allocation fails, retrying after
   ``torch._C._host_emptyCache()`` succeeds — the retry must be exercised.

Exit code 0 = all paths correct.
"""
from __future__ import annotations

import logging
import sys

sys.path.insert(0, r"D:\jasna2\work")

import torch

from jasna.ltx import blocks as B


def make_blocks(count: int, *, seed: int) -> list[dict[str, torch.Tensor]]:
    """Six tensors per block, the weight-ish mix the real store packs."""
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for index in range(count):
        blocks.append(
            {
                "attn.q": torch.randn(64, 32, generator=generator) * 0.1,
                "attn.k": torch.randn(64, 32, generator=generator) * 0.1,
                "mlp.up": torch.randn(128, 32, generator=generator) * 0.1,
                "norm": torch.rand(32, generator=generator),
                "quant.w": torch.randint(0, 255, (4096,), dtype=torch.uint8, generator=generator),
                "quant.scale": torch.rand(4, generator=generator) + 0.5,
            }
        )
    return blocks


def cycle_and_check(store: B.BlockStore, source: list[dict[str, torch.Tensor]], label: str) -> None:
    """Acquire every block twice (two full cycles) and compare against the source."""
    for cycle in (0, 1):
        for index in range(store.count):
            weights = store.acquire(index)
            for name, tensor in source[index].items():
                got = weights[name]
                assert got.dtype == tensor.dtype, (label, index, name, got.dtype)
                assert got.shape == tensor.shape, (label, index, name, got.shape, tensor.shape)
                assert torch.equal(got.cpu(), tensor), (label, cycle, index, name)
            store.release(index)
    print(f"  {label}: weights round-trip across two full cycles OK")


def main() -> int:
    assert torch.cuda.is_available(), "needs a CUDA device"
    device = torch.device("cuda:0")
    failures = 0

    source = make_blocks(6, seed=7)

    print("== 1. healthy path (pinned) ==")
    store = B.BlockStore(make_blocks(6, seed=7), device, resident=2)
    try:
        assert all(buffer.is_pinned() for buffer in store._host), "host buffers should be pinned"
        cycle_and_check(store, source, "pinned")
    finally:
        store.close()

    print("== 2. driver refuses pinned memory -> pageable fallback ==")
    real_empty = torch.empty
    pinned_calls = {"n": 0}

    def no_pinned_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            pinned_calls["n"] += 1
            raise RuntimeError("CUDA error: resource already mapped (simulated)")
        return real_empty(*args, **kwargs)

    torch.empty = no_pinned_empty  # type: ignore[assignment]
    try:
        store = B.BlockStore(make_blocks(6, seed=7), device, resident=2)
    finally:
        torch.empty = real_empty  # type: ignore[assignment]
    try:
        assert store._host and not any(buffer.is_pinned() for buffer in store._host), (
            "fallback host buffers must be pageable"
        )
        cycle_and_check(store, source, "pageable fallback")
    finally:
        store.close()

    print("== 3. transient failure -> empty_host_cache retry ==")
    real_empty = torch.empty
    real_empty_cache = torch._C._host_emptyCache
    calls = {"pin_fails": 1, "cache_clears": 0}

    def flaky_pinned_empty(*args, **kwargs):
        if kwargs.get("pin_memory") and calls["pin_fails"] > 0:
            calls["pin_fails"] -= 1
            raise RuntimeError("CUDA error: out of memory (simulated, once)")
        return real_empty(*args, **kwargs)

    def counting_empty_cache():
        calls["cache_clears"] += 1
        return real_empty_cache()

    torch.empty = flaky_pinned_empty  # type: ignore[assignment]
    torch._C._host_emptyCache = counting_empty_cache  # type: ignore[assignment]
    try:
        store = B.BlockStore(make_blocks(6, seed=7), device, resident=2)
    finally:
        torch.empty = real_empty  # type: ignore[assignment]
        torch._C._host_emptyCache = real_empty_cache  # type: ignore[assignment]
    try:
        assert calls["cache_clears"] == 1, f"retry should empty the host cache once, saw {calls}"
        assert all(buffer.is_pinned() for buffer in store._host), "retry should land pinned buffers"
        cycle_and_check(store, source, "retry")
    finally:
        store.close()

    print("== 4. CUDA error surfacing stays honest (sync before the big allocs) ==")
    code = open(B.__file__, encoding="utf-8").read()
    assert "torch.cuda.synchronize(self.device)" in code, "the pre-allocation sync must stay"
    print("  ok: BlockStore syncs before allocating, so a poisoned context fails at its origin")

    print()
    print("BLOCKSTORE HARDENING:", "PASS" if failures == 0 else f"{failures} FAILURES")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
