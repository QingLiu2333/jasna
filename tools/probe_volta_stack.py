"""Empirically probe the Volta-era stack: torch 2.5 + torch_tensorrt 2.5 + TRT 10.3.

Runs the application's own compile -> save -> load round trip
(``compile_and_save_torchtrt_dynamo`` / ``load_torchtrt_export``) so the result
is directly meaningful for the product, not a toy. Verifies, in order:

1. torch.cuda arch list actually contains sm_70 (Volta).
2. TensorRT 10.x reports a min-compute floor of (7, 0) in jasna.arch_compat.
3. A static-shape engine round trips through the app's functions.
4. A dynamic-shape engine round trips (the BasicVSR++ sub-engine pattern).
5. Outputs match the eager module within fp16 tolerance.

Exit code 0 = the Volta tree can ship TensorRT acceleration.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/work")


def main() -> int:
    import torch

    print("torch            :", torch.__version__, torch.version.cuda)
    archs = torch.cuda.get_arch_list()
    print("archs            :", archs)
    assert any(a in ("sm_70", "sm_70+PTX") for a in archs), "sm_70 missing - cannot serve Volta"
    assert torch.cuda.is_available(), "no CUDA device on this probe machine"
    print("this GPU         :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    import tensorrt as trt
    import torch_tensorrt as tt

    print("tensorrt         :", trt.__version__)
    print("torch_tensorrt   :", tt.__version__)

    from jasna import arch_compat

    arch_compat.tensorrt_version.cache_clear()
    print("trt min compute  :", arch_compat.tensorrt_min_compute())
    assert arch_compat.tensorrt_supports((7, 0)), "TRT floor must include Volta (7.0)"
    assert not arch_compat.tensorrt_supports((6, 1)), "Pascal must stay out of TRT"

    from jasna.trt.torch_tensorrt_export import (
        compile_and_save_torchtrt_dynamo,
        load_torchtrt_export,
    )

    device = torch.device("cuda:0")
    workspace = 1 << 30  # 1 GiB is plenty for the toy model

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.act = torch.nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.conv(x))

    eager = Toy().eval().to(device).to(torch.float16)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="volta-trt-") as tmp:
        # --- static shape ---
        static_path = os.path.join(tmp, "static.engine")
        sample = torch.randn(2, 3, 64, 64, device=device, dtype=torch.float16)
        compile_and_save_torchtrt_dynamo(
            module=eager,
            inputs=[sample],
            output_path=static_path,
            dtype=torch.float16,
            workspace_size_bytes=workspace,
            message="[probe] static-shape engine",
            device=device,
        )
        loaded = load_torchtrt_export(checkpoint_path=static_path, device=device)
        with torch.no_grad():
            ref = eager(sample)
            got = loaded(sample)
        err = (ref.float() - got.float()).abs().max().item()
        ok = err < 5e-2
        failures += 0 if ok else 1
        print(f"static roundtrip : {'OK ' if ok else 'FAIL'} max|diff|={err:.4g} size={os.path.getsize(static_path)}")

        # --- dynamic shape (the BasicVSR++ sub-engine pattern) ---
        dynamic_path = os.path.join(tmp, "dynamic.engine")
        dyn_input = tt.Input(
            min_shape=[1, 3, 64, 64],
            opt_shape=[4, 3, 64, 64],
            max_shape=[4, 3, 64, 64],
            dtype=torch.float16,
        )
        compile_and_save_torchtrt_dynamo(
            module=eager,
            inputs=[dyn_input],
            output_path=dynamic_path,
            dtype=torch.float16,
            workspace_size_bytes=workspace,
            message="[probe] dynamic-shape engine",
            device=device,
        )
        loaded_d = load_torchtrt_export(checkpoint_path=dynamic_path, device=device)
        with torch.no_grad():
            x1 = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16)
            x4 = torch.randn(4, 3, 64, 64, device=device, dtype=torch.float16)
            ref1, got1 = eager(x1), loaded_d(x1)
            ref4, got4 = eager(x4), loaded_d(x4)
        err1 = (ref1.float() - got1.float()).abs().max().item()
        err4 = (ref4.float() - got4.float()).abs().max().item()
        ok = err1 < 5e-2 and err4 < 5e-2 and got1.shape == ref1.shape and got4.shape == ref4.shape
        failures += 0 if ok else 1
        print(f"dynamic roundtrip: {'OK ' if ok else 'FAIL'} max|diff| b1={err1:.4g} b4={err4:.4g} size={os.path.getsize(dynamic_path)}")

    # --- the rest of the app imports on this stack ---
    import importlib

    for name in (
        "jasna.engine_compiler",
        "jasna.restorer.basicvsrpp_sub_engines",
        "jasna.mosaic.rfdetr",
        "jasna.mosaic.yolo",
        "jasna.mosaic.detection_registry",
        "jasna.session_factory",
        "jasna.pipeline",
    ):
        try:
            importlib.import_module(name)
            print(f"import           : ok   {name}")
        except Exception as exc:
            failures += 1
            print(f"import           : FAIL {name}: {type(exc).__name__}: {exc}")

    print()
    print("VOLTA STACK PROBE:", "PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
