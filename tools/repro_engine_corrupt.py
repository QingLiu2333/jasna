"""Confirm that a sub-engine archive whose *TRT payload* cannot be deserialized
produces exactly the error chain reported by the RTX 4070 / 4060 Ti users.

    D:\\jasna\\venv312\\Scripts\\python.exe D:\\jasna2\\tools\\repro_engine_load.py
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile

import torch  # noqa: F401
import torch_tensorrt  # noqa: F401

FLAT = r"D:\jasna2\dist\jasna\model_weights\lada_mosaic_restoration_model_generic_v1.2_sub_engines"
NAME = "loop_body_backward_1.trt_fp16.win.engine"
WORK = r"D:\jasna2\_engtest2"


def rebuild_with_damaged_payload(src: str, dst: str) -> None:
    """Copy the archive, flipping bytes inside the serialized TRT engine blob."""
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        target = [n for n in names if n.endswith("custom_obj_0")][0]
        data = {n: zin.read(n) for n in names}
    blob = bytearray(data[target])
    # Damage the middle of the engine plan, leaving the zip container intact.
    mid = len(blob) // 2
    for i in range(mid, mid + 64):
        blob[i] ^= 0xFF
    data[target] = bytes(blob)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for name in names:
            zout.writestr(name, data[name])


def attempt(label: str, fn) -> None:
    print(f"\n--- {label}")
    try:
        fn()
        print("    [OK]")
    except Exception as exc:  # noqa: BLE001
        print(f"    [FAIL] {type(exc).__name__}: {exc}")


def main() -> int:
    os.makedirs(WORK, exist_ok=True)
    dst = os.path.join(WORK, NAME)
    rebuild_with_damaged_payload(os.path.join(FLAT, NAME), dst)

    def via_export_path() -> None:
        torch.export.load(dst)

    def via_export_fileobj() -> None:
        with open(dst, "rb") as f:
            torch.export.load(f)

    def via_torch_load() -> None:
        torch.load(dst, map_location="cuda", weights_only=False)

    attempt("torch.export.load(path) - damaged TRT payload", via_export_path)
    attempt("torch.export.load(fileobj) - damaged TRT payload", via_export_fileobj)
    attempt("torch.load(path) - damaged TRT payload", via_torch_load)
    print("\n(compared against the user log: 'We ran into an error when deserializing "
          "the saved file' + 'PytorchStreamReader failed locating file data.pkl')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
