"""Does a modern torch_tensorrt still work when TensorRT is pinned back to 10.4?

torch_tensorrt declares a narrow TensorRT range (2.11 wants 10.15-10.16), and
TensorRT removed Volta support in 10.5 — so a Volta-capable build has to run a
torch_tensorrt that was compiled against a *newer* TensorRT than the one shipped.
libnvinfer keeps its ABI within a major version, so this can work; this script
finds out whether it actually does, by running the app's own engine round trip:

    compile_and_save_torchtrt_dynamo()  ->  load_torchtrt_export()

against the TRT/torch combination in the active interpreter. Run it with the
legacy venv:

    D:\\jasna2\\venv126\\Scripts\\python.exe tools\\probe_trt_downgrade.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"D:\jasna2\work")

import torch


def main() -> int:
    import tensorrt as trt
    import torch_tensorrt

    print(f"torch            {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"torch_tensorrt   {torch_tensorrt.__version__}")
    print(f"tensorrt         {trt.__version__}")
    print(f"archs            {torch.cuda.get_arch_list()}")
    print(f"device           {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
    print()

    class Tiny(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x * 2.0 + 1.0)

    device = torch.device("cuda")
    model = Tiny().to(device).eval()
    sample = torch.randn(1, 3, 16, 16, device=device)

    from jasna.trt.torch_tensorrt_export import (
        EngineArchiveError,
        compile_and_save_torchtrt_dynamo,
        load_torchtrt_export,
    )

    with tempfile.TemporaryDirectory(prefix="trt-probe-") as tmp:
        out = str(Path(tmp) / "tiny.engine")

        print("compile ->", end=" ", flush=True)
        try:
            compile_and_save_torchtrt_dynamo(
                module=model,
                inputs=[sample],
                output_path=out,
                dtype=torch.float16,
                workspace_size_bytes=1 << 30,
                message="probing",
                device=device,
                optimization_level=1,
            )
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            return 1
        print(f"ok ({Path(out).stat().st_size / 1024:.0f} KiB)")

        print("load  ->", end=" ", flush=True)
        try:
            loaded = load_torchtrt_export(checkpoint_path=out, device=device)
        except EngineArchiveError as exc:
            print(f"FAILED (EngineArchiveError): {exc}")
            return 1
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            return 1
        print("ok")

        print("run   ->", end=" ", flush=True)
        try:
            with torch.no_grad():
                got = loaded(sample)
            expected = model(sample)
            close = torch.allclose(got.float(), expected.float(), atol=0.05, rtol=0.05)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            return 1
        print(f"ok (matches eager: {close})")
        if not close:
            return 1

    print()
    print("VERDICT: this torch / torch_tensorrt / TensorRT combination works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
