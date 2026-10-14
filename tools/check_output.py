"""Quick output-quality check: is the encoded file clean, or full of garbage?

Compares the per-frame local gradient energy of the encoded file against the
source. A correct encode sits near 1.0; a frame dominated by corruption (block
noise, tearing, misordered reference frames) spikes well above 5.

Usage:
    python check_output.py <source> <encoded> [--frames N] [--verbose]

Exit code 0 when every frame looks clean, 1 otherwise. Useful as a regression
gate after touching the decode, blend, engine-path or encoder code.
"""

from __future__ import annotations

import argparse
import sys

import av
import numpy as np

SPIKE = 5.0          # per-frame ratio that means "this frame is garbage"
MEAN_ALARM = 1.5     # mean ratio that means "something is off overall"


def frame_scores(path: str, limit: int) -> np.ndarray:
    """Local gradient energy of the first ``limit`` frames (grayscale)."""
    container = av.open(path)
    try:
        out = []
        for i, frame in enumerate(container.decode(video=0)):
            if i >= limit:
                break
            gray = frame.to_ndarray(format="gray").astype(np.float32)
            gx = np.abs(np.diff(gray, axis=1)).mean()
            gy = np.abs(np.diff(gray, axis=0)).mean()
            out.append(float(gx + gy))
    finally:
        container.close()
    return np.asarray(out, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("encoded")
    ap.add_argument("--frames", type=int, default=300, help="frames to compare (default 300)")
    ap.add_argument("--verbose", action="store_true", help="print every frame ratio")
    args = ap.parse_args()

    src = frame_scores(args.source, args.frames)
    enc = frame_scores(args.encoded, args.frames)

    n = min(len(src), len(enc))
    if n == 0:
        print("ERROR: no frames decoded", file=sys.stderr)
        return 1
    src, enc = src[:n], enc[:n]
    ratio = enc / np.maximum(src, 1e-6)

    print(f"frames compared : {n}")
    print(f"source  energy  : mean={src.mean():.3f} max={src.max():.3f}")
    print(f"encoded energy  : mean={enc.mean():.3f} max={enc.max():.3f}")
    print(f"ratio (enc/src) : mean={ratio.mean():.3f} max={ratio.max():.3f}")

    if args.verbose:
        for i, r in enumerate(ratio):
            flag = "  <-- SPIKE" if r > SPIKE else ""
            print(f"  frame {i:4d}: {r:6.3f}{flag}")

    spikes = np.flatnonzero(ratio > SPIKE)
    bad = len(spikes) > 0 or ratio.mean() > MEAN_ALARM

    if bad:
        if len(spikes):
            preview = ", ".join(str(int(i)) for i in spikes[:20])
            more = "" if len(spikes) <= 20 else f" (+{len(spikes) - 20} more)"
            print(f"FAIL: {len(spikes)} frame(s) above {SPIKE}: {preview}{more}", file=sys.stderr)
        else:
            print(f"FAIL: mean ratio {ratio.mean():.3f} above {MEAN_ALARM}", file=sys.stderr)
        return 1

    print("OK: output looks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
