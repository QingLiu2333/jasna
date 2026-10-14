"""Verify the media kernels' fatbins serve every card tier natively.

The three runtime trees cover Pascal..Blackwell, so each fatbin must carry
code for all of them. Per kernel this checks:

1. **Image set** - legacy SASS for sm_60/61/62/70/72 (NVRTC 12.6, the
   toolchain generation that still supports them), the modern sm_75..sm_121
   SASS, and both PTX images (compute_60 + compute_75).
2. **Modern load** - ``cuModuleLoadData`` on the full fatbin succeeds on this
   machine and every entry point resolves (exercises the SASS path).
3. **PTX fallback** - a PTX-only repack (no SASS at all) still loads and
   resolves through the driver's JIT: the safety net for any future
   architecture not in the SASS list.
4. **Legacy symbols** - the sm_61 cubin (GTX 10xx - the reported card) and the
   sm_70 cubin (Volta) export exactly the same entry points as the modern
   build, so a Pascal/Volta card gets the same kernels, natively compiled.

Exit code 0 = every kernel passes.
"""
from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import tempfile
from ctypes import c_void_p
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "work" / "jasna" / "media"
CUDA_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin")
CUOBJDUMP = CUDA_BIN / "cuobjdump.exe"
FATBINARY = CUDA_BIN / "fatbinary.exe"

LEGACY_SASS = [60, 61, 62, 70, 72]
MODERN_SASS = [75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121]
EXPECTED_PTX = [60, 75]


def run(tool: Path, args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        [str(tool), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise SystemExit(f"{tool.name} {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def entry_symbols(cubin: Path) -> set[str]:
    return set(re.findall(r"STO_ENTRY\s+(\w+)", run(CUOBJDUMP, ["-symbols", str(cubin)])))


def main() -> int:
    sys.path.insert(0, str(ROOT / "work"))
    import torch  # noqa: F401  (creates the CUDA context below)
    from jasna.media import cuda_kernel as ck

    torch.zeros(1, device="cuda")
    lib = ck.cuda_driver()
    ck.check_cuda(lib.cuInit(0), "cuInit")

    failures = 0
    fatbins = sorted(MEDIA.glob("*.fatbin"))
    assert fatbins, "no fatbins found"

    with tempfile.TemporaryDirectory(prefix="verify-kernels-") as temp:
        temp_path = Path(temp)
        for fatbin in fatbins:
            name = fatbin.stem
            problems: list[str] = []

            # ---- 1. image set -------------------------------------------------
            elf = run(CUOBJDUMP, ["--list-elf", str(fatbin)])
            ptx = run(CUOBJDUMP, ["--list-ptx", str(fatbin)])
            sass = sorted(int(m) for m in re.findall(r"sm_(\d+)\.cubin", elf))
            ptx_arches = sorted(int(m) for m in re.findall(r"sm_(\d+)\.ptx", ptx))
            if sass != sorted(LEGACY_SASS + MODERN_SASS):
                problems.append(f"SASS set {sass} != {sorted(LEGACY_SASS + MODERN_SASS)}")
            if ptx_arches != EXPECTED_PTX:
                problems.append(f"PTX set {ptx_arches} != {EXPECTED_PTX}")

            # ---- 4. the legacy cubins export the modern entry points ----------
            elf_dir = temp_path / f"elf-{name}"
            elf_dir.mkdir()
            run(CUOBJDUMP, ["-xelf", "all", str(fatbin)], cwd=elf_dir)
            cubins = {
                int(m.group(1)): file
                for file in elf_dir.glob("*.cubin")
                if (m := re.search(r"\.sm_(\d+)\.cubin$", file.name))
            }
            reference = entry_symbols(cubins[86]) if 86 in cubins else entry_symbols(cubins[MODERN_SASS[0]])
            symbols: list[str] = sorted(reference)
            for sm in (61, 70):
                if sm not in cubins:
                    problems.append(f"missing sm_{sm}.cubin")
                    continue
                got = entry_symbols(cubins[sm])
                if got != reference:
                    problems.append(
                        f"sm_{sm} symbols differ (missing {sorted(reference - got)}, extra {sorted(got - reference)})"
                    )

            # ---- 2. modern load (SASS path on this machine) -------------------
            module = c_void_p()
            data = fatbin.read_bytes()
            rc = lib.cuModuleLoadData(ctypes.byref(module), ctypes.create_string_buffer(data))
            if rc != 0:
                problems.append(f"cuModuleLoadData(full fatbin) rc={rc}")
            else:
                for symbol in symbols:
                    fn = c_void_p()
                    rc2 = lib.cuModuleGetFunction(ctypes.byref(fn), module, symbol.encode())
                    if rc2 != 0:
                        problems.append(f"full fatbin missing symbol {symbol} (rc={rc2})")
                        break

            # ---- 3. PTX fallback (JIT path) -----------------------------------
            ptx_file = MEDIA / f"{name}.sm60.ptx"
            if not ptx_file.is_file():
                problems.append(f"{ptx_file.name} not on disk")
            else:
                ptx_only = temp_path / f"{name}.ptxonly.fatbin"
                subprocess.run(
                    [
                        str(FATBINARY), "--64", "--create", str(ptx_only),
                        f"--image3=file={ptx_file},kind=ptx,sm=60",
                    ],
                    check=True,
                    capture_output=True,
                )
                module = c_void_p()
                rc = lib.cuModuleLoadData(ctypes.byref(module), ctypes.create_string_buffer(ptx_only.read_bytes()))
                if rc != 0:
                    problems.append(f"cuModuleLoadData(PTX-only) rc={rc} - future cards would fail")
                else:
                    for symbol in symbols:
                        fn = c_void_p()
                        rc2 = lib.cuModuleGetFunction(ctypes.byref(fn), module, symbol.encode())
                        if rc2 != 0:
                            problems.append(f"PTX-only module missing symbol {symbol} (rc={rc2})")
                            break

            mark = "ok  " if not problems else "FAIL"
            detail = f"{len(sass)} SASS, PTX {ptx_arches}, {len(symbols)} entry symbols"
            print(f"{mark} {name:<24} {detail}")
            for problem in problems:
                failures += 1
                print(f"      - {problem}")

    print()
    print("LEGACY KERNEL COVERAGE:", "PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
