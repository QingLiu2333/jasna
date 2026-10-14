"""Verify the engine-recovery fix: a foreign/unloadable engine archive must be
reported with the real TensorRT reason and moved aside instead of failing the job.

Self-provisioning - it takes a healthy engine from the staged per-GPU archive and
derives the damaged one from it, so it does not depend on leftover scratch dirs.

    D:\\jasna\\venv312\\Scripts\\python.exe D:\\jasna2\\tools\\verify_engine_recovery.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"D:\jasna2\work")

import torch  # noqa: F401
import torch_tensorrt  # noqa: F401

from jasna import engine_paths  # noqa: E402
from jasna.trt.torch_tensorrt_export import EngineArchiveError, load_torchtrt_export  # noqa: E402

# Engines only ever ship inside a per-GPU archive named after the card that built
# them (engines/basicvsrpp/<token>/); stage_dist.py prunes every flat .engine.
# The staged tree moves between builds (_pkg / dist), so probe the known roots.
ARCHIVE_ROOTS = (
    Path(r"D:\jasna2\_pkg\rt\cu130\jasna\model_weights\engines\basicvsrpp"),
    Path(r"D:\jasna2\_pkg\rt\cu124\jasna\model_weights\engines\basicvsrpp"),
    Path(r"D:\jasna2\dist\jasna\model_weights\engines\basicvsrpp"),
    Path(r"D:\jasna2\build\model_weights\engines\basicvsrpp"),
)
ENGINE_NAME = "loop_body_backward_1.trt_fp16.win.engine"


def find_healthy_engine() -> Path:
    for root in ARCHIVE_ROOTS:
        candidates = sorted(root.glob(f"*/{ENGINE_NAME}"))
        if candidates:
            return candidates[0]
    raise SystemExit(f"no {ENGINE_NAME} under any of {[str(r) for r in ARCHIVE_ROOTS]}")


def damage(source: Path, target: Path) -> None:
    """Copy *source* and scramble bytes inside its payload."""
    payload = bytearray(source.read_bytes())
    start = len(payload) // 2
    for offset in range(start, min(start + 64, len(payload))):
        payload[offset] ^= 0xFF
    target.write_bytes(payload)


def main() -> int:
    device = torch.device("cuda")
    good = find_healthy_engine()
    print(f"healthy engine: {good}")

    scratch = Path(tempfile.mkdtemp(prefix="verify-engine-"))
    damaged = scratch / ENGINE_NAME
    damage(good, damaged)

    try:
        print("\n== 1. healthy archive loads, no error ==")
        module = load_torchtrt_export(checkpoint_path=str(good), device=device)
        print("   OK:", type(module).__name__)

        print("\n== 2. damaged archive raises EngineArchiveError with a reason ==")
        try:
            load_torchtrt_export(checkpoint_path=str(damaged), device=device)
        except EngineArchiveError as exc:
            print(f"   OK: reason={exc.reason}")
            print(f"       detail={exc.detail[:160] or '(none captured)'}")
        else:
            print("   FAIL: no exception")
            return 1

        print("\n== 3. quarantine moves the archive of the owning GPU ==")
        root = Path(tempfile.mkdtemp(prefix="engines-"))
        try:
            token = "sm99-deadbeef"
            (root / "engines" / "basicvsrpp" / token).mkdir(parents=True)
            victim = root / "engines" / "basicvsrpp" / token / "loop_body.moved.engine"
            victim.write_bytes(b"x")

            original_root = engine_paths.engines_root
            engine_paths.engines_root = lambda: root / "engines"
            try:
                listed_before = {entry["token"] for entry in engine_paths.other_archives("basicvsrpp")}
                assert token in listed_before, f"{token} should be visible before quarantine"

                moved = engine_paths.quarantine_engine_archive_for_path(victim, reason="test")
                print(f"   moved -> {moved}")
                assert moved is not None and moved.is_dir()
                assert not (root / "engines" / "basicvsrpp" / token).exists()
                assert (moved / "unusable_reason.txt").is_file()
                assert (moved / "loop_body.moved.engine").is_file()
                print("   OK: directory renamed, reason recorded, files preserved")

                unlisted = {entry["token"] for entry in engine_paths.other_archives("basicvsrpp")}
                assert token not in unlisted and moved.name not in unlisted
                print("   OK: other_archives() no longer offers it")
            finally:
                engine_paths.engines_root = original_root

            print("\n== 4. a non-archive path is left alone ==")
            loose = root / "loose.engine"
            loose.write_bytes(b"x")
            assert engine_paths.quarantine_engine_archive_for_path(loose) is None
            assert loose.is_file()
            print("   OK")
        finally:
            shutil.rmtree(root, ignore_errors=True)

        print("\n== 5. legacy adoption is opt-in ==")
        assert engine_paths.legacy_engines_adoption_enabled() is False
        os.environ["JASNA_ADOPT_LEGACY_ENGINES"] = "1"
        assert engine_paths.legacy_engines_adoption_enabled() is True
        os.environ.pop("JASNA_ADOPT_LEGACY_ENGINES")
        print("   OK: default off, JASNA_ADOPT_LEGACY_ENGINES=1 enables it")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
