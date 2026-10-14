"""Reproduce the "PytorchStreamReader failed locating file data.pkl" failure on
shipped BasicVSR++ sub-engine archives.

Run with the shared build venv (torch 2.12.1 + tensorrt 10.16 + torch_tensorrt):
    D:\\jasna\\venv312\\Scripts\\python.exe D:\\jasna2\\tools\\repro_engine_load.py
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback

import torch  # noqa: F401  (must be imported before torch_tensorrt)
import torch_tensorrt  # noqa: F401

FLAT = r"D:\jasna2\dist\jasna\model_weights\lada_mosaic_restoration_model_generic_v1.2_sub_engines"
PER_GPU = r"D:\jasna2\dist\jasna\model_weights\engines\basicvsrpp\sm86-1ce0a4e075ac"
NAME = "loop_body_backward_1.trt_fp16.win.engine"

WORK = r"D:\jasna2\_engtest"


def probe(label: str, path: str, *, as_fileobj: bool) -> None:
    print(f"\n--- {label}")
    print(f"    path={path}")
    try:
        if as_fileobj:
            with open(path, "rb") as f:
                result = torch.export.load(f)
        else:
            result = torch.export.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"    [FAIL] {type(exc).__name__}: {exc}")
        return
    print(f"    [OK] {type(result).__name__}")


def probe_torch_load(label: str, path: str) -> None:
    print(f"\n--- {label} (torch.load)")
    try:
        torch.load(path, map_location="cuda", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"    [FAIL] {type(exc).__name__}: {exc}")
        return
    print("    [OK]")


def main() -> int:
    os.makedirs(WORK, exist_ok=True)

    same_name = os.path.join(WORK, NAME)
    renamed = os.path.join(WORK, "renamed_engine.engine")
    renamed_same_stem = os.path.join(WORK, "sub", NAME)

    for src, dst in (
        (os.path.join(FLAT, NAME), same_name),
        (os.path.join(FLAT, NAME), renamed),
        (os.path.join(FLAT, NAME), renamed_same_stem),
    ):
        if not os.path.isfile(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    print("torch", torch.__version__, "| trt", torch_tensorrt.__version__)

    for label, path, as_obj in (
        ("gen_B flat, exact name, path", same_name, False),
        ("gen_B flat, exact name, file object", same_name, True),
        ("gen_B flat, RENAMED, path", renamed, False),
        ("gen_B flat, RENAMED, file object", renamed, True),
        ("gen_B flat, same name, other dir", renamed_same_stem, False),
    ):
        probe(label, path, as_fileobj=as_obj)

    probe_torch_load("gen_B flat", same_name)

    per_gpu = os.path.join(PER_GPU, NAME)
    if os.path.isfile(per_gpu):
        probe("gen_A per-GPU, exact name, path", per_gpu, as_fileobj=False)
        probe("gen_A per-GPU, exact name, file object", per_gpu, as_fileobj=True)

    print("\n=== PyTorchFileWriter/Reader internal view ===")
    for p in (same_name, renamed):
        try:
            records = torch._C.PyTorchFileReader(p).get_all_records()
            print(f"  {os.path.basename(p)}: {len(records)} records, first={records[0]!r}")
            try:
                torch._C.PyTorchFileReader(p).get_record("archive_format")
                print("    get_record('archive_format') -> OK")
            except Exception as exc:  # noqa: BLE001
                print(f"    get_record('archive_format') -> {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
