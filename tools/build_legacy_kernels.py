"""Rebuild the media kernels' fatbins so every card tier gets native code.

Why this exists
---------------
The three runtime trees each serve one card tier (cu126 -> Pascal and older,
cu124 -> Volta, cu130 -> Turing and newer), but the ``.fatbin`` kernels they
ship were all built by the CUDA 13 toolchain, which **cannot target anything
below compute capability 7.5** (``nvcc`` and even ``ptxas`` reject sm_61
outright). The result on a real GTX 1060: decode died with
``cuModuleLoadData failed: CUDA_ERROR_NO_BINARY_FOR_GPU`` while every other
part of the pipeline ran fine.

The fix follows the same rule as the runtime trees themselves - use the
toolchain version that supports the target. The **NVRTC that ships inside the
CUDA 12.6 tree** was never restricted the way offline compilation was, and
still emits *native SASS* for the whole legacy range. This tool uses it to add:

    sm_60  (P100, compute 6.0)
    sm_61  (GTX 10xx family - the reported card)
    sm_62  (Jetson TX2 / MX 6.2)
    sm_70  (Volta: Titan V, V100)
    sm_72  (Xavier)

so each fatbin becomes (per kernel):

    SASS sm_60,61,62,70,72   (new; native SASS from NVRTC 12.6)
  + SASS sm_75..sm_121       (kept verbatim; CUDA 13 build)
  + PTX compute_75           (kept; forward JIT for future architectures)
  + PTX compute_60           (new; ultimate fallback)

The driver picks the exact SASS for the card it runs on, as always. No runtime
code changed: ``cuModuleLoadData`` still reads a single fatbin.

Requires
--------
* ``cuobjdump``/``fatbinary`` from the CUDA 13 toolkit (packaging only - they
  themselves never see an unsupported target).
* The NVRTC pair (``nvrtc64_120_0.dll`` + ``nvrtc-builtins64_126.dll``) that
  ships inside the cu126 tree's torch, or the matching files in ``venv126``.
* Two include paths for the NVRTC compile: ``tools/nvrtc_include`` (a minimal
  ``stdint.h``, since NVRTC ships none) and the CUDA toolkit's ``include``
  directory (for ``cuda_fp16.h`` and its ``vector_types.h``).

Usage
-----
    python tools/build_legacy_kernels.py            # rebuild every kernel
    python tools/build_legacy_kernels.py yuv_to_rgb # just one
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from ctypes import byref, c_char_p, c_int, c_size_t, c_void_p
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "work" / "jasna" / "media"

DEFAULT_NVRTC_DIR = ROOT / "_pkg" / "rt" / "cu126" / "jasna" / "_internal" / "torch" / "lib"
FALLBACK_NVRTC_DIR = ROOT / "venv126" / "Lib" / "site-packages" / "torch" / "lib"
DEFAULT_CUDA_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin")

# Every consumer target below the CUDA 13 floor (7.5) that the NVRTC in the
# CUDA 12.6 tree still emits native SASS for.
LEGACY_SASS_ARCHES = (60, 61, 62, 70, 72)
# Virtual-arch PTX fallback: compute_60 JITs on anything sm_60 and newer.
LEGACY_PTX_ARCH = "60"

_ARCH_PATTERN = re.compile(r"\.sm_(\d+)\.cubin$")


class Nvrtc:
    """Minimal ctypes wrapper around the NVRTC that ships in the CUDA 12.6 tree."""

    def __init__(self, lib_dir: Path) -> None:
        self.lib_dir = lib_dir
        # NVRTC loads nvrtc-builtins64_*.dll by bare name from the search path.
        os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib_dir))
        self.lib = ctypes.WinDLL(str(lib_dir / "nvrtc64_120_0.dll"))

    def _check(self, code: int, what: str) -> None:
        if code == 0:
            return
        message = c_char_p()
        self.lib.nvrtcGetErrorString(c_int(code), byref(message))
        raise RuntimeError(f"{what} failed: {code} {message.value!r}")

    def _compile(self, source: str, name: str, arch: str, includes: list[Path]) -> c_void_p:
        program = c_void_p()
        self._check(
            self.lib.nvrtcCreateProgram(byref(program), source.encode(), name.encode(), 0, None, None),
            "nvrtcCreateProgram",
        )
        options = [f"--gpu-architecture={arch}", "-std=c++17"]
        options += [f"-I{include}" for include in includes]
        encoded = [option.encode() for option in options]
        code = self.lib.nvrtcCompileProgram(program, len(encoded), (c_char_p * len(encoded))(*encoded))
        if code != 0:
            size = c_size_t()
            self.lib.nvrtcGetProgramLogSize(program, byref(size))
            log = ctypes.create_string_buffer(size.value)
            self.lib.nvrtcGetProgramLog(program, log)
            self.lib.nvrtcDestroyProgram(byref(program))
            raise RuntimeError(f"NVRTC could not compile {name} for {arch}:\n{log.value.decode(errors='replace')}")
        return program

    def compile_ptx(self, source: str, name: str, arch: str, includes: list[Path]) -> str:
        program = self._compile(source, name, f"compute_{arch}", includes)
        size = c_size_t()
        self._check(self.lib.nvrtcGetPTXSize(program, byref(size)), "nvrtcGetPTXSize")
        buffer = ctypes.create_string_buffer(size.value)
        self._check(self.lib.nvrtcGetPTX(program, buffer), "nvrtcGetPTX")
        self.lib.nvrtcDestroyProgram(byref(program))
        return buffer.value.decode()

    def compile_cubin(self, source: str, name: str, sm: int, includes: list[Path]) -> bytes:
        """Native SASS for one exact architecture, straight out of NVRTC."""
        program = self._compile(source, name, f"sm_{sm}", includes)
        size = c_size_t()
        self._check(self.lib.nvrtcGetCUBINSize(program, byref(size)), "nvrtcGetCUBINSize")
        buffer = ctypes.create_string_buffer(size.value)
        self._check(self.lib.nvrtcGetCUBIN(program, buffer), "nvrtcGetCUBIN")
        self.lib.nvrtcDestroyProgram(byref(program))
        return buffer.raw[: size.value]


def run(tool: Path, args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        [str(tool), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{tool.name} {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def extract_images(
    cuobjdump: Path, fatbin: Path, work_dir: Path
) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    """``(sass, ptx)`` image lists of the current fatbin, preserved as-is."""
    elf_dir = work_dir / "elf"
    elf_dir.mkdir()
    run(cuobjdump, ["-xelf", "all", str(fatbin)], cwd=elf_dir)
    sass: list[tuple[int, Path]] = []
    for cubin in sorted(elf_dir.glob("*.cubin")):
        match = _ARCH_PATTERN.search(cubin.name)
        if match:
            sass.append((int(match.group(1)), cubin))
    if not sass:
        raise RuntimeError(f"no SASS could be extracted from {fatbin}")

    ptx: list[tuple[int, Path]] = []
    if re.search(r"\.sm_(\d+)\.ptx", run(cuobjdump, ["--list-ptx", str(fatbin)])):
        ptx_dir = work_dir / "ptx"
        ptx_dir.mkdir()
        run(cuobjdump, ["-xptx", "all", str(fatbin)], cwd=ptx_dir)
        for file in sorted(ptx_dir.glob("*.ptx")):
            match = re.search(r"\.sm_(\d+)\.ptx$", file.name)
            if match:
                ptx.append((int(match.group(1)), file))
    return sass, ptx


def list_images(cuobjdump: Path, fatbin: Path) -> tuple[list[int], list[int]]:
    elf = run(cuobjdump, ["--list-elf", str(fatbin)])
    ptx = run(cuobjdump, ["--list-ptx", str(fatbin)])
    elfs = [int(m) for m in re.findall(r"sm_(\d+)\.cubin", elf)]
    ptxs = [int(m) for m in re.findall(r"sm_(\d+)\.ptx", ptx)]
    return elfs, ptxs


def entry_symbols(cuobjdump: Path, cubin: Path) -> set[str]:
    symbols = run(cuobjdump, ["-symbols", str(cubin)])
    return set(re.findall(r"STO_ENTRY\s+(\w+)", symbols))


def rebuild_kernel(
    cuobjdump: Path, fatbinary: Path, nvrtc: Nvrtc, source: Path, includes: list[Path]
) -> str:
    fatbin = source.with_suffix(".fatbin")
    if not fatbin.is_file():
        raise RuntimeError(f"{fatbin} does not exist")
    text = source.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="fatbin-") as temp:
        temp_path = Path(temp)

        # New: native SASS for every legacy consumer target, from the NVRTC
        # whose toolchain generation still supports them.
        legacy: list[tuple[int, Path]] = []
        for sm in LEGACY_SASS_ARCHES:
            cubin = temp_path / f"{source.stem}.sm_{sm}.cubin"
            cubin.write_bytes(nvrtc.compile_cubin(text, source.name, sm, includes))
            legacy.append((sm, cubin))

        # Kept verbatim: the SASS and PTX the fatbin already carried.
        kept_sass, kept_ptx = extract_images(cuobjdump, fatbin, temp_path)

        # The legacy cubins must export exactly the entry points the modern
        # build does - same source, different toolchain generation.
        reference = entry_symbols(cuobjdump, kept_sass[0][1])
        for sm, cubin in legacy:
            symbols = entry_symbols(cuobjdump, cubin)
            if symbols != reference:
                raise RuntimeError(
                    f"{fatbin.name} sm_{sm}: entry symbols differ from the modern build "
                    f"(missing {sorted(reference - symbols)}, extra {sorted(symbols - reference)})"
                )

        # Ultimate fallback: compute_60 PTX (JITs on anything sm_60+).
        ptx_path = source.with_suffix(f".sm{LEGACY_PTX_ARCH}.ptx")
        ptx_path.write_text(
            nvrtc.compile_ptx(text, source.name, LEGACY_PTX_ARCH, includes),
            encoding="utf-8",
            newline="\n",
        )

        out = temp_path / fatbin.name
        args = ["--64", "--create", str(out)]
        for sm, cubin in [*legacy, *kept_sass]:
            args.append(f"--image3=file={cubin},kind=elf,sm={sm}")
        for sm, ptx in kept_ptx:
            args.append(f"--image3=file={ptx},kind=ptx,sm={sm}")
        args.append(f"--image3=file={ptx_path},kind=ptx,sm={LEGACY_PTX_ARCH}")
        run(fatbinary, args)

        elfs, ptxs = list_images(cuobjdump, out)
        wanted_sass = sorted([sm for sm, _ in legacy] + [sm for sm, _ in kept_sass])
        wanted_ptx = sorted([sm for sm, _ in kept_ptx] + [int(LEGACY_PTX_ARCH)])
        if sorted(elfs) != wanted_sass or sorted(ptxs) != wanted_ptx:
            raise RuntimeError(
                f"repacked {fatbin.name} does not round-trip: "
                f"elf={sorted(elfs)} ptx={sorted(ptxs)} (wanted elf {wanted_sass}, ptx {wanted_ptx})"
            )

        shutil.copyfile(out, fatbin)

    modern = ", ".join(str(sm) for sm, _ in kept_sass)
    legacy_list = ", ".join(str(sm) for sm, _ in legacy)
    return (
        f"{fatbin.name}: SASS legacy[{legacy_list}] + kept[{modern}] "
        f"+ PTX compute_{LEGACY_PTX_ARCH},{','.join(str(s) for s, _ in kept_ptx)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernels", nargs="*", help="kernel names (without .cu); default: every kernel")
    parser.add_argument("--nvrtc-dir", type=Path, default=None, help="directory holding nvrtc64_120_0.dll")
    parser.add_argument("--cuda-bin", type=Path, default=DEFAULT_CUDA_BIN, help="CUDA toolkit bin directory")
    parser.add_argument(
        "--include",
        type=Path,
        action="append",
        default=None,
        help="extra NVRTC include directory (repeatable; defaults to tools/nvrtc_include + the toolkit include)",
    )
    args = parser.parse_args()

    nvrtc_dir = args.nvrtc_dir
    if nvrtc_dir is None:
        nvrtc_dir = DEFAULT_NVRTC_DIR if DEFAULT_NVRTC_DIR.is_dir() else FALLBACK_NVRTC_DIR
    if not (nvrtc_dir / "nvrtc64_120_0.dll").is_file():
        raise SystemExit(f"nvrtc64_120_0.dll not found under {nvrtc_dir}")
    cuobjdump = args.cuda_bin / "cuobjdump.exe"
    fatbinary = args.cuda_bin / "fatbinary.exe"
    for tool in (cuobjdump, fatbinary):
        if not tool.is_file():
            raise SystemExit(f"missing toolkit tool: {tool}")

    includes = args.include if args.include else [ROOT / "tools" / "nvrtc_include", args.cuda_bin.parent / "include"]
    for include in includes:
        if not include.is_dir():
            raise SystemExit(f"include directory not found: {include}")

    sources = (
        [MEDIA / f"{name.removesuffix('.cu')}.cu" for name in args.kernels]
        if args.kernels
        else sorted(MEDIA.glob("*.cu"))
    )
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"no such kernel source: {source}")

    nvrtc = Nvrtc(nvrtc_dir)
    print(f"NVRTC: {nvrtc_dir}")
    for source in sources:
        print(" ", rebuild_kernel(cuobjdump, fatbinary, nvrtc, source, includes))
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
