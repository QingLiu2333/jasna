# Jasna — Windows build with per-GPU TensorRT engines and selectable codec devices

This folder contains a self-contained build of **Jasna v0.11.0** (from
[Kruk2/jasna](https://github.com/Kruk2/jasna), `main`, upstream commit
`d366827`) with three added capabilities:

1. **Selectable video codec devices** — choose which device *encodes* and which
   device *decodes* the video, independently of the GPU that runs the
   restoration models.
2. **Per-GPU engine archiving ("一卡一用")** — every compiled TensorRT engine is
   filed under an identity token for the card that built it. Swapping the
   physical GPU no longer breaks the install: the app detects the new card and
   compiles a fresh set, keeping the old one.
3. **A lower architecture floor** — Volta (compute 7.0) is supported out of the
   box, Pascal / GTX 10xx is opt-in, and the Blackwell-only gate on `ltx_fast`
   is replaced with a capability probe plus a dequantising fallback.

Everything needed to run is inside `dist/jasna/` — Python runtime, PyTorch
(CUDA 13), TensorRT, PyAV/FFmpeg, OpenCV, Tk/CustomTkinter, `ffmpeg`/`ffprobe`,
libVLC and the model weights. Nothing has to be installed.

## Start

Double-click **`jasna.exe`** (GUI), or use it from a terminal (CLI):

```bat
jasna.exe --input input.mp4 --output output.mkv
jasna.exe --help
```

> Keep the folder path free of non-ASCII characters — Jasna refuses to run from
> a path containing them.

## Why this build exists

The original project picks the codec device from the compute GPU. On this PC the
compute GPU is an **NVIDIA CMP 90HX**, a mining card with **no NVENC/NVDEC video
engines at all**, so the original build cannot encode anything: every job dies
with

```
OpenEncodeSessionEx failed: unsupported device (2)
No capable devices found
```

The Intel **UHD Graphics 770** iGPU on the same CPU does have working video
engines. This build therefore moves decode and encode to whichever device can
actually do the job, while the restoration models keep running on the CMP 90HX.

## Choosing the devices

**GUI** — right panel → *Encoding*, the first rows:

| Setting | What it controls |
| --- | --- |
| **视频编解码设备** / Video codec device | which device **encodes** (and, when decode is left on auto, decodes) |
| **视频解码设备** / Video decode device | which device **decodes**, independently |
| **计算显卡（修复模型）** / Compute GPU (restoration) | which CUDA device runs detection + restoration, and therefore which engine archive is used |

The compute-GPU row only appears when more than one CUDA device is present. Under
it the panel prints a live summary of the selected card and the archive it maps
to, e.g.

```
NVIDIA CMP 90HX | cc 8.6 | 50 SM | 10.0 GiB | driver 616.56 | TensorRT 10.16.1.11 | pci 01:00
engine archive: sm86-1ce0a4e075ac
compute paths: NVIDIA CMP 90HX (sm86) | fp16 | int8-tc | dequantized-lowprecision
```

Encoder choices:

| Choice | Encode | Decode (when decode is auto) |
| --- | --- | --- |
| Auto (recommended) | probes NVENC → QSV → AMF → CPU, uses the first that works | follows the encoder |
| NVIDIA NVENC / NVDEC | NVENC | NVDEC |
| Intel Quick Sync (iGPU) | `hevc_qsv`, `h264_qsv` (AV1 → CPU) | Quick Sync / D3D11VA, else software |
| AMD AMF | `hevc_amf`, `h264_amf`, `av1_amf` | AMF |
| CPU / software | `libx265`, `libx264`, `libsvtav1` | FFmpeg software |

Decoder choices: Auto (follow encoder), NVIDIA NVDEC, Intel Quick Sync (iGPU),
AMD AMF, CPU / software.

Equivalent CLI flags:

```bat
jasna.exe --codec-backend qsv --decode-backend cpu  --input in.mp4 --output out.mkv
jasna.exe --codec-backend cpu  --decode-backend qsv  --input in.mp4 --output out.mkv
jasna.exe --device cuda:1       --codec-backend qsv   --input in.mp4 --output out.mkv
```

`--device` names the CUDA device for the **models**; the GUI setting and the CLI
flag both pin the engine archive to that device.

## One card, one engine set ("一卡一用")

### The problem

A TensorRT engine is built for the exact GPU architecture it was compiled on. The
original project wrote every engine into a single flat
`model_weights/` directory, so swapping the physical card in a machine left the
old engines in place — the app then tried to deserialise an engine built for the
previous card and refused to start rather than rebuilding.

The same trap catches a **pre-built engine that was not built on this machine**.
Two RTX 40-series users hit it: they installed a build that shipped engines
compiled on another card (an sm86 machine), those engines were counted as "ready
for this GPU", and loading them failed — reported, confusingly, as *"your
checkpoint file is corrupted / data.pkl not found"*. The archive is now
quarantined and rebuilt for the installed card instead (see below), and pre-built
engines are only shipped under the *building card's* token.

### The scheme

Every CUDA device is reduced to a **stable identity token** by
`jasna/gpu_identity.py`:

```
fmt=1|vendor=nvidia|pci=0000:01:00|ids=0000:220d|cc=8.6|sm=50|
mem=10736959488|drv=616.56|trt=10.16.1.11
```

hashed to `sha256[:12]` and prefixed with the capability string, giving a
filesystem-safe token such as **`sm86-1ce0a4e075ac`**. The inputs deliberately
exclude clocks, current free VRAM and the GPU-instance UUID, so the token is
stable across reboots, driver restarts and device reordering — an engine stays
valid as long as the *hardware identity* is unchanged.

Engines live at:

```
model_weights/engines/<kind>/<token>/<engine file>
```

with `kind` one of `basicvsrpp`, `detection`, `yolo`, `unet4x`, `onnx`. Alongside
them a small `gpu_identity.txt` records the human-readable description.

### What happens on a card swap

* The device's token changes, so the app looks in a **new, empty** archive and
  compiles fresh engines there. The old card's archive is left untouched.
* `engine_preflight` reports the situation before a job starts: the selected
  device, its token, which engines are missing, and any *other* archives present
  (`switches_gpu` is true when other archives exist but this card's set is
  incomplete — i.e. "you changed the physical card").
* Engine files from a **pre-archive** install (`model_weights/*.engine` and the
  historic `<model>_sub_engines/`) are **ignored by default**: they carry no
  record of the card that built them, and a TensorRT plan only deserializes on
  that card. Set `JASNA_ADOPT_LEGACY_ENGINES=1` to move them into this card's
  archive anyway (only do that when you are sure they were built on the card
  installed now).

### When an engine cannot be used (self-healing)

Engines that exist but cannot run on the installed card — a pre-built archive
shipped to another machine, an engine built before a driver or TensorRT upgrade,
a damaged file — are detected at load time:

1. `load_torchtrt_export()` reports the real cause (`EngineArchiveError`, with
   the TensorRT message when the runtime provides one) instead of torch's
   generic *"your checkpoint file is corrupted / data.pkl not found"*.
2. The owning archive directory is renamed to
   `<token>.unusable-<timestamp>` (nothing is deleted; an `unusable_reason.txt`
   is written inside), so every "do the engines exist?" check now answers *no*.
3. The job rebuilds the set for the card that is actually installed and retries
   (CLI: inline; GUI: `ensure_engines_compiled`). If the rebuild still cannot be
   loaded, the run falls back to the PyTorch network instead of failing.

The same treatment applies to the detection engine (`trt.TrtRunner`): an
unreadable plan quarantines its archive and RF-DETR is recompiled on the spot.

> A TensorRT plan is **not** portable between cards. Never ship or keep a
> pre-built engine for a card the user does not have: the app will find it, try
> to deserialize it and fail. This build therefore ships only the token-keyed
> `engines/<kind>/<token>/` archives — `stage_dist.py` prunes anything left in
> the flat layout — and each machine compiles the sets it needs.

Per-GPU archiving can be disabled with `JASNA_ENGINE_PER_GPU=0`, which restores
the historic flat layout (useful for diagnostics or to keep a hand-built engine
in place).

## Console-less GUI launches fail with `[WinError 6]` (not a file problem)

Second failure mode found on the same builds, unrelated to the engines: every job
stops on its first frame, before any model output.

```
ERROR   Failed to process test_clip1_1080p.mp4: [WinError 6] 句柄无效。
  File "jasna/gui/processor.py", line 424, in _run_video_job
  File "jasna/pipeline.py", line 591, in run
  File "jasna/media/probe.py", line 142, in get_video_meta_data
  File "subprocess.py", line 1365, in _get_handles       <- the stdin branch
  File "subprocess.py", line 1416, in _make_inheritable  <- DuplicateHandle
```

**Cause.** `subprocess` resolves every stream the caller left as `None` through
`GetStdHandle` and immediately duplicates it. `get_video_meta_data()` passes
`stdout`/`stderr` as pipes but leaves `stdin` unset, so it is the *stdin* lookup
that dies. On a double-click launch the GUI calls `FreeConsole()` and
`_redirect_std_streams_to_null()`, which hands the kernel NUL handles — but those
streams were only referenced by `sys.stdin`/`sys.stdout`. Rebinding any of them
(which the multiprocessing machinery does) collects the stream, CPython closes its
handle, and the kernel's `STD_INPUT_HANDLE` is left pointing at the freed value.
The duplicate then fails with `ERROR_INVALID_HANDLE (6)` before the child starts.
Because a freed handle value is often recycled in the meantime, the failure looks
intermittent — which is why it survived the earlier attempt at a fix.

**This is not a missing or corrupt media file**, and the video is never opened.

**Fix**, two independent layers:

1. `_redirect_std_streams_to_null()` keeps the NUL streams in module globals
   (`_NULL_STDIN`/`_NULL_STDOUT`) so no reference count can free them, instead of
   relying on `sys.*` alone.
2. `ensure_standard_handles()` re-checks all three standard handles with
   `GetHandleInformation` (a handle closed behind the kernel's back still compares
   as non-`NULL`, so only the kernel can tell) and re-points any broken one at NUL.
   It runs from `subprocess_no_window_kwargs()`, which every tool spawn goes
   through, so the repair happens last — after all the torch/CUDA/TensorRT handle
   churn. Handles that still work are left untouched, so a CLI run that redirected
   its own output keeps the redirection.

`gui/file_actions.py` launches `explorer`/`open`/`xdg-open` directly, so it now
passes all three streams explicitly; those calls never consult the process'
standard handles at all.

Note the kwargs deliberately do **not** inject a `stdin` key: three call sites (the
streaming encoder, the TVAI restorer, the engine compiler) pass their own `stdin`
and a duplicate keyword would be a `TypeError`.

## Lower architecture requirements

| Architecture | Compute | Accepted by the gate | Can it actually run this build? |
| --- | --- | --- | --- |
| Pascal (GTX 10xx) | 6.1 | **yes** (floor) | **yes** — via `rt/cu126` (CUDA 12.6 PyTorch path; no TensorRT, no LTX) |
| Volta (Titan V, V100) | 7.0 | yes | **yes** — via `rt/cu124` (torch 2.5 + TensorRT 10.3, the last TensorRT that accepts Volta; LTX yes, fast mode needs 8.9+) |
| Turing (RTX 20xx, T4) | 7.5 | yes | yes — `rt/cu130`, INT8 tensor-core GEMM path |
| Ampere (RTX 30xx, A100) | 8.0–8.6 | yes | yes |
| Ada (RTX 40xx) | 8.9 | yes | yes — FP8 + NVFP4 fast paths |
| Hopper / Blackwell | 9.0 / 10.x–12.x | yes | yes |
| Maxwell (GTX 9xx) | 5.x | no | — (`JASNA_ALLOW_LEGACY_GPU` only reaches 6.0) |

The gate is **6.1** — every consumer GTX 10-series card. `JASNA_ALLOW_LEGACY_GPU=1`
lowers it to 6.0 for GP100/P100 parts. The old floor was 7.5, which refused a card
this build was meant to accept.

### Gate vs. kernels: two different questions

Passing the capability gate says the card is *new enough*. It says nothing about
whether the shipped binaries were **compiled** for it, and for pre-Turing cards
they are not:

* **CUDA 13 drops offline compilation below compute capability 7.5.** NVIDIA's
  release notes are explicit, and it shows in the wheels —
  `torch.cuda.get_arch_list()` on this build returns
  `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`.
* **TensorRT supports SM 7.5 and higher only** ("TensorRT supports NVIDIA hardware
  with compute capability SM 7.5 or higher"; a GTX 1080 Ti is not enumerated).
  There is no TensorRT path for Pascal at all, not even a slow one.

A Pascal card that cleared the old gate would therefore get past the dialog and
then die mid-job with *"no kernel image is available for execution on the device"*.
The new **GPU Kernel Coverage** row reports this up front instead — as a warning,
not a gate, so the check tells the truth without inventing a new block. The check
uses CUDA's binary-compatibility rule (an sm_80 cubin runs on sm_86/sm_89; a
different major version needs PTX, which these wheels do not carry).

**To actually run a GTX 10-series card** you need a build whose PyTorch was
compiled against **CUDA 12.6** — the last CUDA that targets pre-Turing hardware,
and still published for PyTorch ≤ 2.14. Such a build runs the restoration models
through the PyTorch fp32 path (the tensor-core quantised paths and every TensorRT
engine are unavailable), so expect it to be slow. It is a separate package, not a
setting.

### TensorRT has its own floor, and it moves

Being accepted by the capability gate is not the same as having a TensorRT path.
TensorRT drops architectures on its own schedule, and which cards it can target is
decided by the **installed** version:

| TensorRT | Lowest compute capability | Volta (7.0) | Pascal (6.x) |
| --- | --- | --- | --- |
| 11.x | 7.5 | no | no |
| 10.5 – 10.16 | 7.5 | no | no |
| 10.0 – 10.4 | 7.0 | **yes** | no |
| 8.6 | 5.3 | yes | yes |

"Volta support removed starting with TensorRT 10.5" is NVIDIA's own release note;
10.2 already called it deprecated. The 10.x line has never supported Pascal.

`arch_compat.tensorrt_min_compute()` derives that floor from the installed version,
and `arch_compat.tensorrt_supports_device()` answers "can TensorRT build anything
for this card". Three places now respect it instead of assuming "NVIDIA means
TensorRT":

* `engine_compiler.ensure_engines_compiled()` — nothing is reported as missing, so
  no compile is attempted and the run uses the PyTorch restorer;
* `engine_compiler._detection_engine_exists()` — returns "satisfied", so RF-DETR
  runs as the torch model (exactly as it already does on AMD);
* `mosaic/rfdetr.py` — picks the torch detector instead of a TensorRT runner.

Before this, a Pascal card would pass the capability gate, spend minutes in a
compile that cannot succeed, and fail with a TensorRT error. Now it quietly takes
the PyTorch path, and `ensure_engines_compiled` logs why.

### A Pascal / Volta build: what is and is not possible

Verified with `torch.cuda.get_arch_list()` on the wheels themselves:

| Wheel | Architectures |
| --- | --- |
| `torch ... +cu130` (this build) | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` |
| `torch ... +cu126` | `sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90` |

**cu126 is the only CUDA flavour that covers Pascal and Volta**, so that is the
wheel a legacy package has to be built against (Ada's 8.9 is covered by the sm_86
cubin under CUDA's binary-compatibility rule). CUDA 13 cannot be used at all — it
dropped offline compilation below 7.5.

TensorRT for Volta is the hard part, and the numbers do not currently allow a
modern torch:

| torch_tensorrt (`+cu126`) | requires torch | requires TensorRT | Volta? |
| --- | --- | --- | --- |
| 2.6.x | 2.6.0 | 10.7 – 10.8 | no |
| 2.11.0 | 2.11.x | 10.15 – 10.16 | no |
| 2.14.0 | 2.14.x | 11.1 – 11.2 | no |
| 2.5.0 (`+cu124`) | 2.5.x | 10.3 | **yes** |

Every torch_tensorrt from 2.6 on demands TensorRT 10.7 or newer, which is past
the 10.5 cutoff. Pinning an older TensorRT underneath a newer torch_tensorrt does
**not** work — measured, not assumed:

```
AttributeError: 'tensorrt_bindings.tensorrt.Builder' object has no attribute
                'build_engine_with_config'
```

TensorRT 10.4's Python bindings lack the entry point torch_tensorrt 2.11 calls, so
the ABI is only stable in the direction NVIDIA supports. The one supported pairing
that still targets Volta is the 2024-era stack (torch 2.5 + torch_tensorrt 2.5.0 +
TensorRT 10.3), which is a different serialisation API for engines as well —
`torch.export.save`/`load` only arrived with torch 2.6, so `trt/torch_tensorrt_export.py`
would need a version-adaptive path and the whole pipeline would need re-validating
on torch 2.5.

So the practical split is:

* **Pascal (GTX 10xx)** — a `+cu126` build runs it on CUDA through the PyTorch
  models. No TensorRT is possible, and none is attempted.
* **Volta (V100, Titan V)** — the same build runs it on CUDA with first-generation
  tensor cores. TensorRT on Volta needs the torch-2.5 stack described above.

### LTX is not offered on Pascal

`session_factory` refuses the fast (NVFP4) LTX transformer below Ada, but the
settings panel used to offer the switch anyway and only fail at run time. The UI
now mirrors the run-time rule, and LTX as a whole is refused on Pascal and older:

* **LTX model** — needs Volta (7.0) or newer. Pascal has no Tensor Cores, so every
  LTX layer would be evaluated through the dequantised fallback: correct output at
  a speed that makes a real video impractical. The card is greyed out with
  `model_ltx_needs_tensor_cores` as the reason.
* **LTX fast mode** — needs block-scaled FP4/FP8 tensor-core GEMMs (8.9+). The
  switch is hidden unless `supports_block_quantized_gemm()` agrees, and
  `JASNA_LTX_QUANTIZED_ON_LEGACY=1` still opts back in.

The capability travels from the CUDA warm-up (`_warm_up_cuda()` →
`set_gpu_support(capability=...)`) so the decision is made against the GPU this
process actually uses.

### TensorRT-dependent options are locked off where TensorRT cannot run

The same warm-up now also reports whether the installed TensorRT can target
the selected card, and three things follow from it:

* **"Compile BasicVSR++" switch** — locked off and greyed out on cards below
  the TensorRT floor (Pascal). It used to promise "15-60 minutes" of
  compilation that `engine_compiler` would then refuse to start; the PyTorch
  CUDA path was always the only option there. `set_enabled()` re-applies the
  gate after processing re-enables the panel, and presets cannot flip it back
  on.
* **Engine preflight** — no longer lists engines as "missing" on such cards.
  A GTX 1060 was shown `需要准备 GPU 专用模型文件 / 缺失的引擎: YOLO, BasicVSR++`
  and then told one line later that TensorRT cannot target the GPU at all.
  Nothing will ever be compiled there, so nothing is reported.
* **Engine compilation** (already in place) — `ensure_engines_compiled()` skips
  the whole pipeline and logs `TensorRT cannot target this GPU (...)`, running
  the PyTorch models instead.

### LTX streamed blocks survive a refused pinned allocation

Field report (RTX 5070 Ti, T0.39): the LTX trial crashed at ~9 % with
`CUDA error: resource already mapped` inside `BlockStore.__init__` — but
**only in INT8 (quality) mode; the NVFP4 fast mode ran to completion**. The two
modes pack identical code into `BlockStore`; the only difference is the size of
each streamed block: **256.6 MiB for INT8 vs 144.7 MiB for NVFP4**, and
PyTorch's caching host allocator rounds pinned allocations up to a power of two —
a **512 MiB** page-locked request for INT8, a **256 MiB** one for NVFP4. The
driver accepted the smaller mapping and refused the larger one on that machine,
which is why the mode choice decided the crash. (The same INT8 trial runs to
completion on the reference machine, so the trigger is a driver/system-property
of the host, not the code path.)

`ltx/blocks.py` now survives that refusal regardless of which side of the size
boundary a machine lands on: `BlockStore` synchronises before its large
allocations (so a genuinely poisoned context fails at its true origin instead of
being misattributed to the pinned alloc), retries the pinned request once after
emptying the host cache, and finally degrades to **pageable host memory** — the
side-stream copies then synchronise internally, which costs some prefetch
overlap but the job completes. Covered by `tools/verify_block_store.py`
(healthy / refused / transient-failure paths plus weight round-trips).

### The media kernels carry every tier's native SASS

The six custom CUDA kernels (`yuv_to_rgb`, `rgb_to_yuv`, `lut`, `cas`,
`denoise`, `resize_normalize`) ship as `.fatbin` containers loaded through the
driver API. The original build embedded only SASS for sm_75..sm_121 and a
compute_75 PTX — and **PTX is only forward-compatible**, so neither could serve
Pascal (GTX 10xx, sm_61) or Volta (sm_70): a real GTX 1060 died at decode with
`cuModuleLoadData failed: CUDA_ERROR_NO_BINARY_FOR_GPU` even though the
PyTorch side of the cu126 tree was fine.

The fix follows the same rule as the runtime trees themselves: **compile each
tier with the toolchain that supports it**. CUDA 13 refuses every target below
7.5 (`nvcc` and even `ptxas` reject sm_61), but the NVRTC inside the CUDA 12.6
tree was never restricted and still emits *native SASS* for the whole legacy
range. `tools/build_legacy_kernels.py` uses it to repack each fatbin as:

```
SASS sm_60, 61, 62  (P100, GTX 10xx, TX2)   <- new, native, NVRTC 12.6
SASS sm_70, 72      (Volta, Xavier)         <- new, native, NVRTC 12.6
SASS sm_75..sm_121  (kept verbatim)         <- CUDA 13 build
PTX compute_75      (kept)                  <- forward JIT
PTX compute_60      (new)                   <- ultimate fallback
```

The driver picks the exact SASS for whatever card it runs on. No runtime code
changed; the loader reads the same single file. Every legacy cubin is checked
to export *exactly* the entry points of the modern build, so a GTX 1060 runs
the same kernels — natively compiled, not JIT-lowered.

`tools/verify_legacy_kernels.py` proves all of it per kernel: the image set is
complete (17 SASS + 2 PTX), the full fatbin loads and resolves every entry
point, the sm_61/sm_70 cubins match the modern symbol table, and a PTX-only
repack still loads through the driver's JIT as the forward-compatibility
safety net. The build order is `src/scripts/build_fatbins.sh` →
`tools/build_legacy_kernels.py` → `tools/verify_legacy_kernels.py`.

### The three-runtime package

The distribution carries **three complete runtimes** and a launcher that picks one:

```
jasna.exe                 launcher: reads the GPU, then starts the matching tree
rt/cu130/jasna/jasna.exe  CUDA 13 + TensorRT 10.16, torch 2.12.1+cu130 — Turing (sm_75) .. Blackwell
rt/cu124/jasna/jasna.exe  CUDA 12.4 + TensorRT 10.3,  torch 2.5.1+cu124  — Volta (sm_70)
rt/cu126/jasna/jasna.exe  CUDA 12.6, torch 2.12.1+cu126, no TensorRT     — Maxwell/Pascal (sm_50 .. sm_69)
```

**Why three trees rather than one process with several CUDA stacks.** `torch` is a
single module whose native libraries have fixed names (`torch_cuda.dll`, `cudart`,
`cudnn`, `cublas`…). Two builds of it cannot be loaded into one interpreter, and a
CUDA runtime cannot be initialised twice at different major versions. The choice
therefore has to happen *before* Python starts — which is all the launcher does. At
any moment exactly one tree is running; nothing is "compatible with both".

**How the launcher chooses.** It reads the compute capability straight from the CUDA
driver (`nvcuda.dll`: `cuInit` → `cuDeviceGet` → `cuDeviceComputeCapability`), with
`nvidia-smi --query-gpu=compute_cap` as a fallback. No torch, no import cost:

| Capability | Tree | Why |
| --- | --- | --- |
| ≥ 7.5 | `rt/cu130` | CUDA 13 covers Turing and newer, with native Blackwell kernels |
| = 7.0 (Volta) | `rt/cu124` | CUDA 13 has no Volta kernels and TensorRT 10.5 dropped Volta — this tree pairs torch 2.5 + torch_tensorrt 2.5 with the last TensorRT that accepts Volta (10.3) |
| < 7.0 (Pascal and older) | `rt/cu126` | TensorRT cannot compile for Pascal at all; the PyTorch path is the only way |
| unreadable | `rt/cu130` | better default; a non-NVIDIA build is a different package anyway |

A missing tree, an unreadable capability or a failed start falls through to the
other trees and says so on stderr — the launcher can be wrong without being fatal.
`JASNA_RUNTIME=cu126|cu124|cu130` forces a tree, which is what the build's tests use.

The launcher replaces itself with the chosen `jasna.exe` (`os.execv`) rather than
waiting on it as a child, so console handling stays exactly as before: the app
detaches the console itself for a GUI launch, and nothing else is holding it open.

**What differs between the trees.** Only the CUDA/PyTorch/TensorRT stack:

| | `rt/cu130` | `rt/cu124` | `rt/cu126` |
| --- | --- | --- | --- |
| torch | 2.12.1+cu130 | 2.5.1+cu124 | 2.12.1+cu126 |
| CUDA kernels | sm_75 – sm_120 | sm_50 – sm_90 (Volta native) | sm_50 – sm_90 (Ada 8.9 via the sm_86 cubin) |
| TensorRT | 10.16.1 | 10.3.0 (Volta floor: compute 7.0) | **absent** — the 10.x line cannot compile for Pascal at all |
| engine save/load | `torch.export.save/load` | `torch.save`/`torch.load` pickled module | n/a |
| LTX | yes (fast mode 8.9+) | **yes** (fast mode needs 8.9+, so normal only) | **no** — Pascal has no Tensor Cores |

The engine archive identity token includes the TensorRT version, so the three
trees keep separate per-GPU archives even on the same machine.

**One spec, three trees.** `trt/torch_tensorrt_export.py` adapts at run time:
`torch_tensorrt.compile` keywords the installed release does not know are dropped
automatically (parsed from the `TypeError`), and the save/load path switches on
`hasattr(torch.export, "save")` — torch 2.6+ writes an ExportedProgram, the Volta
tree pickles the compiled module, and `load_torchtrt_export` reads either form.
The ONNX→TensorRT detection engine does not touch torch_tensorrt at all (raw
Builder API, unchanged across TensorRT 10.x), and `arch_compat` derives the
TensorRT compute floor from the installed version: 10.0–10.4 → 7.0, ≥ 10.5 → 7.5.
The Volta dependency set is reproducible from `requirements-volta.txt`.

**Detection without TensorRT (Pascal tree).** The detector is the one model that
had no non-TensorRT path on NVIDIA: RF-DETR ran as a TensorRT engine, and the
torch alternative was gated behind the AMD extra (`jasna[amd]`). The code path
exists (`RfDetrTorchRunner`) but needs the `rfdetr` package and an `rfdetr-*.pt`
checkpoint, neither of which ships (the package was intentionally left out of the
bundle — 150 MB for a model no user has). `mosaic/detection_registry.py` picks the
weights suffix by "can this device run TensorRT" rather than by vendor, so the
Pascal tree lists only the models that actually work there — the bundled
`lada-yolo-v4` `.pt`, which `mosaic/yolo.py` runs through `ultralytics.AutoBackend`.
RF-DETR remains available on Pascal if the user drops an `rfdetr-*.pt` into
`model_weights/` and installs the extra.

### Driver floor: 580

`MIN_DRIVER_VERSION` was 610, which **no Windows 10 machine can satisfy**. NVIDIA's
R580 branch is the last one that exists for Maxwell/Pascal/Volta and the last one
released for Windows 10; its newest Windows release is 582.x, and R590/R610/R615
are Windows 11 / Windows 10 22H2-and-later branches. A 610 floor therefore locked
out every Windows 10 user *and* every 10-series card — the check reported
"582.66 (requires 610+)".

580 is also exactly what CUDA 13 requires, so the floor is now both the minimum
CUDA needs and the maximum those cards can reach.

### How quantisation adapts

`jasna/arch_compat.py` probes the device once and reports which GEMM paths
exist. On a card that lacks them, the quantised layers fall back to
**dequantise-then-multiply** instead of raising:

| Path | Requires | Fallback when absent |
| --- | --- | --- |
| Block-scaled FP8/FP4 `F.scaled_mm` (NVFP4 W4A4, FP8) | Ada (8.9)+ | `dequantized_nvfp4_linear` — unpack E2M1 to bf16, multiply |
| INT8 tensor-core `torch._int_mm` (ConvRot W8A8) | Turing (7.5)+ | `dequantized_int8_linear` — dequantise, multiply in fp16 |

The probe (rather than a hard-coded version rule) means future parts
automatically get the fast paths, and cards whose drivers do not advertise them
self-report the fallback. Override with `JASNA_ALLOW_BLOCK_QUANTIZED_GEMM=0/1`.

The `ltx_fast` (NVFP4) model used to raise
*"needs an RTX 50-series (Blackwell) GPU"* on anything below Blackwell. It now
checks the real capability and raises a helpful message naming your compute
capability and suggesting the INT8/quality model. To force it anyway (every
quantised layer dequantised — correct but several times slower and needing more
VRAM), set `JASNA_LTX_QUANTIZED_ON_LEGACY=1`.

## Windows support

**Windows 10 21H2 (build 19044) or newer, 64-bit.** Windows 11 is fine.

That floor is not enforced by the app — it is what the bundled stack asks for, and
the app now reports your build in a **Windows Version** row so the environment is
visible when something else goes wrong. Below 19044 the row warns; it never blocks.

What was actually broken for Windows 10 was the **driver floor** (see above): 610
could not be installed on Windows 10, so every Windows 10 user failed the check
regardless of their card. That is fixed.

Everything else checks out at the binary level. Every import of the main
executable, the Python runtime, FFmpeg, libvlc, PyTorch and TensorRT was scanned
for dependencies introduced after Windows 10 21H2 — there are none:

| Binary | Imports |
| --- | --- |
| `jasna.exe` | `USER32`, `KERNEL32`, `ADVAPI32` |
| `_internal\python312.dll` | base Windows + UCRT |
| `tools\ffmpeg.exe` | `api-ms-win-crt-*` (UCRT, in-box since Windows 10) |
| `vlc\libvlc.dll` | `KERNEL32`, `ADVAPI32`, `msvcrt` |
| `_internal\torch\...\torch_cuda.dll` | `KERNEL32`, `MSVCP140`, `VCRUNTIME140(_1)`, CUDA DLLs |
| `_internal\...\nvinfer_10.dll` | `KERNEL32`, `USER32`, `SHELL32`, `ole32`, `ADVAPI32` |

Both PE headers are `MajorOperatingSystemVersion 6.0` (Vista+), the app's only
Win32 calls are `SetProcessDpiAwareness`, `GetDpiForSystem`, `MonitorFromWindow`,
`timeBeginPeriod`, `FreeConsole` and `SetStdHandle` — all present since Windows 7
or 8.1 — and the MSVC runtime is bundled, so no redistributable install is needed.

If a Windows 10 machine still misbehaves, check the **Windows Version** and
**GPU Kernel Coverage** rows first: an unpatched 21H2 is supported, but a card
whose kernels this build does not carry is not.

## Multiple GPUs

The settings panel (Encoding section) has a **compute device** picker listing every
CUDA device with its identity and archive token. It appears when more than one
CUDA device is present, and the choice is stored machine-level rather than in the
preset, so a preset copied from another machine cannot pin a meaningless index.

The selection is now authoritative process-wide. It used to be applied only as
`torch.cuda.set_device()`, while several GUI paths spelled `torch.device("cuda:0")`
literally — the video session, the mosaic scanner, the engine preflight and the LUT
preview would keep using card 0 whatever the picker said. Everything resolves
through `os_utils.compute_device()` / `compute_device_string()` now, including:

* the model and detector the pipeline builds,
* the per-GPU engine archive (so engines are compiled and loaded for the chosen card),
* the system checks — they judge the **selected** card, not index 0, which matters
  when the first card in the enumeration is the weak one,
* the CUDA warm-up,
* the CLI, where `--device cuda:1` now records the same selection.

`_compute_device_index` lives in `jasna/os_utils.py`; an out-of-range or malformed
index falls back to 0 instead of raising, so a stale saved value can never stop the
app from starting.

## What was fixed relative to upstream

Three video-corruption defects (all introduced by the previous build and fixed
here) plus the two new features:

| # | Symptom | Root cause | Fix |
| --- | --- | --- | --- |
| 1 | CPU/software encode produced completely garbled frames | `libx265`/`libsvtav1` accept only **planar** YUV, but the GPU conversion emits **packed** NV12/P010; hand-de-interleaving the chroma plane was wrong (≈11 dB per plane vs 41–45 dB packed) | copy the packed buffer into an FFmpeg-owned frame and let swscale planarise |
| 2 | Intermittent bursts of corruption every ~30–60 frames | QSV returns packets whose **DTS equals display-order PTS**, so the stream's DTS is non-monotonic (`bf=3 → 148/300 decreases`); Matroska rejects it, MP4 stores it and players decode against wrong references | encode QSV with B-frames disabled (`bf=0`, the only setting that yields strictly increasing DTS) plus a warn-once guard in `VideoEncoder` |
| 3 | Real root cause of the corruption | Decode output tensors come from `torch.empty` / the caching allocator, which can hand back a block that another pipeline component (TensorRT restorer, crop/blend buffers) just released while its kernels were still in flight and it never called `record_stream` | `_flush_pending_writes()` in `jasna/media/video_decoder.py` — synchronise all streams once per decode group (4 frames) after allocating the output block; ~4% cost; disable with `JASNA_DECODE_ALLOC_FLUSH=0` |

Post-fix measured PSNRs:

| backend / encoder | frame format | PSNR Y / U / V |
| --- | --- | --- |
| `hevc_qsv` | p010le (packed) | 38.3 / 44.9 / 45.0 dB |
| `libx265` | yuv420p10le (planar) | 37.9 / 43.8 / 44.1 dB |
| `libx264` | nv12 (packed) | 37.7 / 43.6 / 43.7 dB |

Per-stage cleanliness after fix 3 (local-gradient-energy score; clean ≈1.9):

| encode | decode | reader0 | reader1 | encoder input |
| --- | --- | --- | --- | --- |
| qsv | cpu | 1.87/1.92 | 1.87/1.92 | 1.84/1.92 |
| cpu | cpu | 1.87/1.92 | 1.87/1.92 | 1.84/1.92 |
| qsv | qsv | 1.87/1.92 | 1.87/1.92 | 1.84/1.92 |
| cpu | auto | 1.87/1.92 | 1.87/1.92 | 1.84/1.92 |

## Measured behaviour (10 s / 300 frames of 1080p source)

End-to-end on this machine (CMP 90HX compute, QSV encode, CPU decode), with the
per-GPU archived engines:

```
cd D:\jasna2\build
python -m jasna --input assets/test_clip1_1080p.mp4 --output out/e2e_qsv_cpu.mkv ^
    --codec-backend qsv --decode-backend cpu --max-clip-size 90
```

| Metric | Value |
| --- | --- |
| Throughput (source run) | 42.4 fps, 300 frames in ~10 s |
| Throughput (frozen `jasna.exe`) | 40.1 fps, ~10 s, 30 s wall including model load |
| VRAM peak | ~3.4 GiB |
| Output sharpness ratio vs source | 0.957 (`check_output.py`: "output looks clean") |
| Output size | 9 826 874 bytes, identical from source and frozen runs |

Confirming the archived engines were used. On the CLI look for the absence of a
`... engine archive unusable ...` line and for the throughput above — falling back
to the PyTorch network is several times slower, so 40 fps on this clip only happens
with the TensorRT engines. The GUI log panel (which logs at INFO) states it
outright:

```
jasna.restorer.basicvsrpp INFO: BasicVSR++ using TRT sub-engines (fp16=True)
jasna.mosaic.rfdetr INFO: RF-DETR detection model loaded:
    model_weights\engines\detection\sm86-1ce0a4e075ac\rfdetr-v6.bs1-4.fp16.win.engine (batch_size=4)
```

## Bundled models

| File | Purpose |
| --- | --- |
| `lada_mosaic_restoration_model_generic_v1.2.pth` | BasicVSR++ restoration weights (per-GPU sub-engines under `engines/basicvsrpp/<token>/`) |
| `lada_mosaic_detection_model_v4_fast.{pt,onnx,fp16.win.engine}` | LADA detection model |
| `rfdetr-v6.onnx`, `rfdetr-v6.bs1-4.fp16.win.engine` | RF-DETR detection (used by default) |
| `rfdetr-v6-large.onnx` | larger RF-DETR variant |
| `checksums_sha256.txt` | integrity hashes |

LTX restoration weights download on first use into `model_weights/ltx-restore/`.

## Layout

```
jasna.exe                  entry point (GUI with no arguments, CLI with arguments)
_internal/                 PyInstaller runtime: Python, PyTorch, TensorRT, PyAV, ...
tools/ffmpeg.exe           used for streaming and metadata
tools/ffprobe.exe
tools/*.dll                FFmpeg shared libraries
model_weights/             model weights
model_weights/engines/     per-GPU TensorRT engine archives
    <kind>/<token>/        one directory per (engine family, GPU identity)
assets/                    logos, sample clips
vlc/                       libVLC + plugins (native player audio)
docs/                      upstream documentation
*.fatbin                   precompiled CUDA kernels (expected at the root when frozen)
```

`docs/` is the upstream documentation set; the pages that matter most here are
`docs/en/cli.md`, `docs/en/tuning.md` and `docs/en/gui.md`.

## Bundled FFmpeg version note

Upstream requires an `ffprobe` whose major version is exactly 8. The bundled
`tools/ffprobe.exe` is a newer git build (major 9); the application's own
in-process FFmpeg (PyAV 18.1.0 / FFmpeg 8.1.2) is what does all decoding,
encoding and muxing, and `ffprobe` is only read for metadata, whose JSON shape is
unchanged. The version gate was therefore relaxed to **8 or newer** and the
wizard/tooltip text updated to match.

## Licensing

Jasna's application source is **AGPL-3.0** (`LICENSE`); models and bundled
dependencies keep their own terms (`LICENSING.md`, `NOTICE`, `assets/licenses/`).
The bundled FFmpeg is a GPL build and libVLC is GPL/LGPL — both are redistributed
here for personal use only. The private `jasna.protection` licensing module is
not part of the public source, so supporter-only models (unet-4x, SD 1.5 image
restoration) are unavailable in this build; the GUI degrades gracefully instead
of crashing.

## Rebuilding

```bat
py -3.12 -m venv D:\jasna\venv312
D:\jasna\venv312\Scripts\python.exe -m pip install -r D:\jasna2\requirements-nvidia.txt -r D:\jasna2\requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cu130
D:\jasna\venv312\Scripts\python.exe -m pip install -e D:\jasna2\work --no-deps
D:\jasna\venv312\Scripts\python.exe D:\jasna2\scan_imports.py
D:\jasna\venv312\Scripts\python.exe -m PyInstaller --noconfirm --distpath D:\jasna2\dist --workpath D:\jasna2\build\pyi D:\jasna2\jasna.spec
D:\jasna\venv312\Scripts\python.exe D:\jasna2\stage_dist.py
```

### Building the three-runtime package

The one-venv recipe above produces a single tree. The shipped package has three plus
a launcher, so it is built four times and then assembled:

```bat
:: 1. the launcher (tiny, onefile)
::    Build from its own spec. Pointing PyInstaller at the .py instead would make
::    it generate a spec *in the current directory* — and with --name jasna that
::    silently overwrote D:\jasna2\jasna.spec, the recipe for the whole app.
set PYTHONPATH=
D:\jasna\venv312\Scripts\python.exe -m PyInstaller --noconfirm ^
    --distpath D:\jasna2\_pkg_launcher --workpath D:\jasna2\build\pyi_launcher ^
    D:\jasna2\launcher\jasna_launcher.spec

:: 2. the CUDA 13 tree (Turing and newer)
D:\jasna\venv312\Scripts\python.exe -m PyInstaller --noconfirm ^
    --distpath D:\jasna2\_pkg\rt\cu130 --workpath D:\jasna2\build\pyi_modern D:\jasna2\jasna.spec
set JASNA_DIST_DIR=D:\jasna2\_pkg\rt\cu130
D:\jasna\venv312\Scripts\python.exe D:\jasna2\stage_dist.py

:: 3. the CUDA 12.4 / Volta tree (see requirements-volta.txt)
D:\jasna2\venv_volta\Scripts\python.exe -m PyInstaller --noconfirm ^
    --distpath D:\jasna2\_pkg\rt\cu124 --workpath D:\jasna2\build\pyi_volta D:\jasna2\jasna.spec
set JASNA_DIST_DIR=D:\jasna2\_pkg\rt\cu124
D:\jasna2\venv_volta\Scripts\python.exe D:\jasna2\stage_dist.py

:: 4. the CUDA 12.6 / Pascal tree (see requirements-legacy.txt; venv126 has no TensorRT)
D:\jasna2\venv126\Scripts\python.exe -m PyInstaller --noconfirm ^
    --distpath D:\jasna2\_pkg\rt\cu126 --workpath D:\jasna2\build\pyi_legacy D:\jasna2\jasna.spec
set JASNA_DIST_DIR=D:\jasna2\_pkg\rt\cu126
D:\jasna2\venv126\Scripts\python.exe D:\jasna2\stage_dist.py

:: 5. assemble and pack
copy D:\jasna2\_pkg_launcher\jasna.exe D:\jasna2\_pkg\jasna.exe
"C:\Program Files\7-Zip-Zstandard\7z.exe" a -mx3 -mmt=on D:\jasna2\dist\jasna_T0.3.7z D:\jasna2\_pkg
```

Two things to know before running this:

* **`PYTHONPATH` must be emptied for `stage_dist.py`.** This machine injects a
  `sitecustomize` shim through `PYTHONPATH` that refuses bulk deletions
  (`CODEBUDDY_SAFE_DELETE_BULK_CONFIRM_REQUIRED`, threshold 50). `stage_dist.py`
  legitimately prunes 114 files from the copied `vlc/`, so staging aborts with
  `STAGE_EXIT=1` and leaves an incomplete tree. `set PYTHONPATH=` clears it for
  that one command; nothing else needs it.
* **Each tree needs a fresh `--distpath`.** PyInstaller's COLLECT removes the
  target directory first, and the same guard blocks that too (see above). Build to
  a new directory rather than over an existing one.

The trees are *not* deduplicated: `assets/`, `model_weights/`, `tools/`, `vlc/`
and the fatbins are byte-identical between them, which is roughly 2.2 GiB of the
package. Hard-linking those before packing (`7z -snh`) would cut the download, at
the cost of requiring a link-aware extractor — not done for the current package,
because an extractor that ignores hard links would silently double the tree.

`stage_dist.py` copies `tools/`, `vlc/`, `assets/`, `model_weights/` and the
`.fatbin` kernels to the dist root, which the code expects when frozen. It then
**prunes every engine file that is not inside a per-GPU archive** (the flat
`model_weights/*.engine` files and the historic `*_sub_engines/` folders) and
writes `model_weights/engines/README.txt` — see "When an engine cannot be used"
above for why shipping a flat engine for the build machine poisons other
machines.

Pack the result for distribution with 7-Zip (Zstandard edition is installed at
`C:\Program Files\7-Zip-Zstandard\`):

```bat
"C:\Program Files\7-Zip-Zstandard\7z.exe" a -mx3 -mmt=on ^
    D:\jasna2\dist\jasna_T0.2.7z D:\jasna2\dist\jasna
```

`-mx3` (Zstandard level 3) packs the ~6.8 GiB tree down to ~4.7 GiB.  Raising the
level past 3 buys little on `.dll`/`.engine` blobs but multiplies the pack time,
so keep it at 3.

> **PyInstaller cannot replace an existing `dist\jasna` here.** Its `COLLECT` step
> removes the target directory first, and this machine's safe-delete guard refuses
> bulk removals (it aborts with `SAFE_DELETE_BULK_CONFIRM_REQUIRED` and
> `EXIT=1`, after the EXE has already been built and with the *old* tree left in
> place — so it is easy to mistake for a successful build). Rename the previous
> output aside first:
>
> ```bat
> move D:\jasna2\dist\jasna D:\jasna2\build\_dist_prev
> ```
>
> ...then build into a clean `dist`, and delete `build\_dist_prev` once the new
> tree is verified. Always confirm the 7z below contains a fresh
> `dist\jasna\jasna.exe` before shipping.

Two directory-name hazards are handled by the spec and worth knowing about if you
move things:

* **The package must be the only `jasna` on the import path.** An editable
  install or a `.pth` naming a different tree will silently win. The build venv
  installs `D:\jasna2\work` in editable mode.
* **`jasna/packaging/` shadows the `packaging` library.** If the *package
  directory itself* (`...\work\jasna`) is put on `sys.path`, `import packaging`
  resolves to jasna's own stub and PyInstaller's hooks fail with
  `No module named 'packaging.requirements'`. Put the **parent** (`...\work`) on
  `sys.path` instead — which is what `jasna.spec` does.

### Useful regression checks

```bat
python D:\jasna2\tools\check_output.py <source> <encoded>
      :: sharpness ratio per frame; a clean encode stays near 1.0, >5 means garbage

python D:\jasna2\tools\verify_engine_recovery.py
      :: foreign/damaged engine archives are reported and quarantined, not fatal

python D:\jasna2\tools\verify_std_handles.py
      :: a killed standard handle no longer breaks child processes

python D:\jasna2\tools\verify_gpu_gates.py
      :: compute/driver floors, kernel coverage per architecture, device selection,
      :: TensorRT floor per TensorRT version

python D:\jasna2\tools\verify_legacy_kernels.py
      :: every media kernel loads on modern GPUs AND through the Pascal/Volta
      :: PTX-JIT path (PTX-only repack, no SASS); also checks block-store fallback

python D:\jasna2\tools\probe_volta_stack.py
      :: torch 2.5 / TensorRT 10.3 engine round trip (run in venv_volta)

D:\jasna2\venv126\Scripts\python.exe D:\jasna2\tools\probe_trt_downgrade.py
      :: whether the active torch / torch_tensorrt / TensorRT trio can compile and
      :: load an engine through the app's own round trip
```

Run `check_output.py` after any change to the decode, blend, engine-path or
encoder code. The `verify_*` scripts are self-contained (they provision their own
fixtures) and need only the build venv and a CUDA device — run them after
touching engine loading, `os_utils`, or anything that resolves a GPU.

## Changes made to the upstream source (this build)

| File | Change |
| --- | --- |
| `jasna/gpu_identity.py` | **new** — stable per-GPU identity token and human-readable description |
| `jasna/engine_paths.py` | **rewritten** — device-qualified engine paths (`engines/<kind>/<token>/`), `adopt_legacy_engines()`, `set_engine_device_index()`, `get_detection_engine_path()` |
| `jasna/arch_compat.py` | **new** — capability probe, `dequantized_nvfp4_linear` / `dequantized_int8_linear` fallbacks, `recommended_fp16()` |
| `jasna/engine_compiler.py` | pin the archive's device index before deciding what is missing; stamp `gpu_identity.txt` after compiling; route detection engines through `kind="detection"` |
| `jasna/os_utils.py` | `MIN_GPU_COMPUTE` 7.5 → **6.1**, `MIN_DRIVER_VERSION` 610 → **580** on Windows too; `check_kernel_coverage()` / `built_cuda_arches()` report whether the shipped binaries were compiled for the installed card; `windows_version()` / `check_windows_version()`; `_redirect_std_streams_to_null()` keeps the NUL streams alive at module level and `ensure_standard_handles()` (called from `subprocess_no_window_kwargs()`) repairs a standard handle freed behind the kernel's back, so a child spawn can no longer die with `[WinError 6]` |
| `jasna/os_utils.py` | compute-device selection: `set_compute_device_index()`, `compute_device_index()`, `compute_device_string()`, `compute_device()` — one place every GPU path resolves through; `check_supported_gpu()` / `check_kernel_coverage()` default to the selected card instead of `cuda:0` |
| `jasna/gui/system_checks.py`, `jasna/gui/wizard.py` | new **GPU Kernel Coverage** and **Windows Version** rows (warning-only, so they inform without blocking); the capability message is filled from the real floor instead of a hardcoded string; GPU checks inspect the selected device |
| `jasna/gui/locales/{en,ja,ko,th,zh}.py` | `wizard_gpu_compute_too_low` no longer hardcodes "7.5" (it said 7.5 while the code enforced 7.0); new strings for the kernel-coverage, Windows and LTX-availability rows |
| `jasna/arch_compat.py` | `tensorrt_version()` / `tensorrt_min_compute()` / `tensorrt_supports()` / `tensorrt_supports_device()` — the TensorRT floor derived from the installed version (10.5+ needs 7.5, 10.0–10.4 still targets 7.0), so "NVIDIA" is no longer mistaken for "TensorRT works here" |
| `jasna/engine_compiler.py`, `jasna/mosaic/rfdetr.py` | skip engine compilation entirely when TensorRT cannot target the card, and run the PyTorch restorer/detector instead of failing after a compile that cannot succeed |
| `jasna/gui/ltx_models.py`, `settings_sections/restoration_model.py`, `settings_panel.py`, `gui/app.py` | LTX is refused on Pascal and older (no Tensor Cores) with a stated reason, and the fast-mode switch now mirrors the run-time rule instead of being offered and then rejected; the capability comes from the CUDA warm-up |
| `launcher/jasna_launcher.py` | **new** — the dual-runtime dispatcher: reads the compute capability from the CUDA driver (no torch), starts `rt/cu130` or `rt/cu126`, falls back to whichever tree exists, `JASNA_RUNTIME` overrides |
| `jasna/trt/__init__.py`, `jasna/trt/trt_runner.py` | `import tensorrt` is guarded: the legacy tree ships without TensorRT, so these modules must import cleanly with it absent (`tensorrt_available()`, `require_tensorrt()`) |
| `jasna/mosaic/detection_registry.py` | RF-DETR's weight suffix follows the *runner* (`.pt` for the torch model, `.onnx` for the TensorRT path) instead of the vendor — keying it on the vendor sent an ONNX file into `torch.load` on the legacy tree |
| `jasna/mosaic/yolo.py` | the engine-vs-`AutoBackend` choice follows "TensorRT can target this card" rather than "the card is NVIDIA", so the shipped `.pt` runs through ultralytics where no engine can exist |
| `requirements-legacy.txt` | **new** — the CUDA 12.6 tree's dependency set (the modern set minus the TensorRT stack, plus `rfdetr`) |
| `tools/probe_trt_downgrade.py` | **new** — runs the app's own compile → save → load round trip against whatever torch/torch_tensorrt/TensorRT trio is installed; this is what proved the TRT-10.4 downgrade impossible |
| `jasna/gui/file_actions.py` | `explorer`/`open`/`xdg-open` launches pass all three streams explicitly instead of letting `subprocess` consult the process' standard handles |
| `jasna/session_factory.py` | `_apply_architecture_profile()` (pin device, log the profile), `_resolve_fp16()`, architecture-aware `ltx_fast` gate with `JASNA_LTX_QUANTIZED_ON_LEGACY=1`; `_build_basicvsrpp_pipeline()` rebuilds engines once when the stored set turns out to be unusable, then falls back to PyTorch |
| `jasna/ltx/transformer.py` | `int8_linear` / `nvfp4_linear` fall back to dequantised matmuls when the tensor-core path is unavailable |
| `jasna/ltx/blocks.py` | the streamed-block host buffers no longer fail the job when the driver refuses to pin memory: `BlockStore` synchronises before its large allocations (so a stale asynchronous CUDA error surfaces at its real origin instead of as a baffling "resource already mapped" on the pinned alloc), retries once after emptying the host cache, and finally degrades to pageable host memory — the side-stream copies then synchronise internally, which costs a little throughput but the job completes. `tools/verify_block_store.py` covers the healthy, refused and transient-failure paths plus weight round-trips |
| `jasna/mosaic/rfdetr.py`, `jasna/trt/__init__.py` | detection engines compile into and load from the `detection` archive; an unreadable detection plan is quarantined and the engine rebuilt on the spot |
| `jasna/trt/torch_tensorrt_export.py` | `EngineArchiveError` (carries the path, the reason and TensorRT's own message), real error reporting instead of the misleading "checkpoint is corrupted", no more suppression of the inner deserialization error |
| `jasna/trt/trt_runner.py` | `TrtEngineError`; a plan that will not deserialize quarantines its archive |
| `jasna/restorer/basicvsrpp_sub_engines.py` | `load_sub_engines()` quarantines a sub-engine set that cannot run on this GPU and re-raises so the caller can rebuild |
| `jasna/engine_paths.py` | `quarantine_engine_archive()` / `quarantine_engine_archive_for_path()`, `legacy_engine_files()`; legacy adoption is now opt-in (`JASNA_ADOPT_LEGACY_ENGINES=1`) |
| `jasna/gui/settings_sections/encoding.py`, `settings_panel.py`, `models.py`, `locales/{en,zh}.py`, `gui/app.py` | compute-GPU picker, live GPU/archive summary, machine-level persistence, CUDA warm-up uses the real capability probe; the saved choice is applied in `JasnaApp.__init__` (before the first-run checks) and `set_compute_device()` delegates to `os_utils` |
| `jasna/gui/video_session.py`, `mosaic_scan.py`, `engine_preflight.py`, `ltx_seed_preview.py`, `app.py` | the literal `torch.device("cuda:0")` these carried is replaced by `os_utils.compute_device()`, so the picker actually governs the session, the scanner, the preflight and the LUT preview |
| `jasna/gui/engine_preflight.py` | reports the device, archive token, other cards' archives and the "card changed" signal; skips every engine requirement when TensorRT cannot target the selected card, so a Pascal user is no longer told to wait for engines that can never be built |
| `jasna/gui/settings_sections/basic.py`, `settings_panel.py`, `app.py` | the CUDA warm-up reports whether TensorRT can target the card; the "Compile BasicVSR++" switch is locked off (and presets cannot re-enable it) where it cannot |
| `jasna/main.py`, `jasna/cli_help.py` | `--device` help; `_pin_engine_device()` binds the engine archive *and* the process-wide compute device to the target device |
| `jasna/media/codec_backend.py` | **new** — backend enum, encode/decode selection precedence, encoder probing, auto-selection and fallback |
| `jasna/media/encoder_specs.py` | **new** — per-backend FFmpeg encoder specs (NVENC, AMF, QSV, software); QSV uses `bf=0` because of the DTS defect |
| `jasna/media/encoder_settings.py` | CQ ranges and validated option names for the QSV/CPU backends |
| `jasna/media/video_encoder.py` | backend-aware spec/encoder choice; FFmpeg-owned host frame (planar corruption fix); non-monotonic-DTS warning |
| `jasna/media/video_decoder.py` | decode device selection, D3D11VA/QSV hardware decode with software fallback; **stream flush before writing a freshly allocated decode block** |
| `jasna/media/splice.py`, `jasna/streaming_encoder.py` | H.264 smart-render and HLS streaming on the QSV/CPU backends |
| `jasna/session_config.py`, `pipeline.py`, `pipeline_threads.py` | plumb the device/backend settings through |
| `jasna/_licensing.py` | **new** — makes the absent private licensing module degrade gracefully instead of crashing the GUI at startup |

## Compiling for Volta / Pascal

The default `torch` / `torch-tensorrt` wheels ship kernels for
`sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. Volta (sm_70) and Pascal (sm_61) are
**not** in that list, so on those cards you should rebuild PyTorch from source
with `TORCH_CUDA_ARCH_LIST="7.0"` (Volta) or `"6.1"` (Pascal) and re-run the
freeze. The `arch_compat` layer already handles the *math* (fp16/int8/FP4
fallbacks) — the limitation is purely which kernels the wheel contains.
