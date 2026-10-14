"""Verify the GPU gates: 6.1 compute floor, 580 driver floor, kernel coverage.

The three things that were wrong, and what each check pins down:

* the compute floor still said 7.5 while the localized message said so too, so a
  GTX 10-series card (6.1) was refused by a card that the build was supposed to
  accept;
* the driver floor was 610, which no Windows 10 machine can satisfy - NVIDIA's
  R580 branch is the last one that exists for Windows 10 (and for
  Maxwell/Pascal/Volta), and its newest Windows release is 582.x;
* passing a capability gate says nothing about whether the shipped CUDA binaries
  were compiled for the card. The CUDA 13 wheels start at sm_75, so a Pascal card
  that clears the gate still cannot execute a single kernel - which is exactly
  what the new coverage check reports instead of leaving "no kernel image is
  available for execution on the device" to surface mid-job.

Usage:
    python tools/verify_gpu_gates.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\jasna2\work")

from jasna import os_utils

FAILURES: list[str] = []
STDOUT = sys.__stdout__


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}", file=STDOUT)
    if not ok:
        FAILURES.append(name)


# --- floors ---------------------------------------------------------------

check("compute floor is 6.1", os_utils.MIN_GPU_COMPUTE == (6, 1), str(os_utils.MIN_GPU_COMPUTE))
check("driver floor is 580", os_utils.MIN_DRIVER_VERSION == 580, str(os_utils.MIN_DRIVER_VERSION))
check(
    "minimum_compute() reports the 6.1 floor",
    os_utils.minimum_compute() == (6, 1),
    str(os_utils.minimum_compute()),
)


def capability(value):
    return lambda device="cuda:0": value


# --- kernel coverage ------------------------------------------------------

real_capability = os_utils._compute_capability
real_arches = os_utils.built_cuda_arches
try:
    built = real_arches()
    check("build advertises sm targets", bool(built), ", ".join(built))

    # Every architecture the shipped wheels actually carry must be reported covered,
    # including the "lower minor, same major" binary-compatibility case (sm_80 cubin
    # runs on sm_86, sm_89).
    for capability_value in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)):
        os_utils._compute_capability = capability(capability_value)
        covered, _detail = os_utils.check_kernel_coverage()
        check(
            f"sm_{capability_value[0]}{capability_value[1]} is covered",
            covered,
            _detail,
        )

    # The cards that clear the 6.1 gate but have no kernels in a CUDA 13 build.
    for capability_value in ((6, 1), (6, 0), (7, 0)):
        os_utils._compute_capability = capability(capability_value)
        covered, detail = os_utils.check_kernel_coverage()
        check(
            f"sm_{capability_value[0]}{capability_value[1]} is reported uncovered",
            not covered,
            detail,
        )
finally:
    os_utils._compute_capability = real_capability
    os_utils.built_cuda_arches = real_arches


# --- the capability gate itself -------------------------------------------

def gpu_result_for(capability_value):
    original = os_utils._compute_capability
    os_utils._compute_capability = capability(capability_value)
    try:
        return os_utils.check_supported_gpu("cuda:0")
    finally:
        os_utils._compute_capability = original


ok, result = gpu_result_for((6, 1))
check("a GTX 10-series card (6.1) passes the capability gate", ok is True, str(result))
ok, result = gpu_result_for((5, 2))
check(
    "a Maxwell card (5.2) is still refused",
    ok is False and result[0] == "compute_too_low",
    str(result),
)


# --- Windows version ------------------------------------------------------

real_windows_version = os_utils.windows_version
try:
    class _Info:
        def __init__(self, build):
            self.major = 10
            self.minor = 0
            self.build = build

    for build, expected_label, expected_ok in (
        (19045, "Windows 10 22H2", True),
        (19044, "Windows 10 21H2", True),
        (19043, "Windows 10 21H1", False),
        (22000, "Windows 11 (build 22000)", True),
    ):
        os_utils.windows_version = lambda b=build: (10, 0, b, (
            "Windows 11 (build %d)" % b if b >= 22000
            else os_utils._WINDOWS_10_RELEASES.get(b, f"Windows 10 (build {b})")
        ))
        _, _, _, label = os_utils.windows_version()
        ok, detail = os_utils.check_windows_version()
        check(
            f"build {build} -> {expected_label}, warning={not expected_ok}",
            label == expected_label and ok is expected_ok,
            f"{label} / {detail}",
        )
finally:
    os_utils.windows_version = real_windows_version

_, _, _, live_label = os_utils.windows_version()
check("live Windows is reported", bool(live_label), live_label)


# --- device selection -----------------------------------------------------

check("compute device defaults to cuda:0", os_utils.compute_device_string() == "cuda:0")
applied = os_utils.set_compute_device_index(1)
check(
    "an out-of-range index falls back to 0 instead of raising",
    applied == 0 and os_utils.compute_device_index() == 0,
    f"applied={applied}",
)
check(
    "a bogus index does not raise",
    os_utils.set_compute_device_index("not-a-number") == 0,
)
check(
    "compute_device() follows the recorded index",
    str(os_utils.compute_device()) == os_utils.compute_device_string(),
    f"{os_utils.compute_device()} / {os_utils.compute_device_string()}",
)


# --- the driver check must be satisfiable on Windows 10 --------------------

ok, detail = os_utils.check_gpu_driver_version()
check("driver check runs and reports a version", bool(detail), detail)
if ok:
    major = int(str(detail).split(".")[0])
    check(
        f"live driver {detail} clears the 580 floor",
        major >= 580,
        f"major={major}",
    )


# --- TensorRT capability --------------------------------------------------

from jasna import arch_compat

real_trt_version = arch_compat.tensorrt_version
try:
    live = arch_compat.tensorrt_min_compute()
    check("installed TensorRT reports a compute floor", live is not None, f"min={live}")

    # The floor follows the TensorRT line: 10.5+ starts at Turing, 10.0-10.4 still
    # targeted Volta, and Pascal is below both wherever the version sits.
    for version, expected_floor, vol, pascal in (
        ((10, 16), (7, 5), False, False),
        ((10, 4), (7, 0), True, False),
        ((10, 5), (7, 5), False, False),
        ((11, 1), (7, 5), False, False),
        # The 8.6 line still covered Maxwell/Pascal, so a build on it would give a
        # GTX 10-series card a TensorRT path. None of our builds use it.
        ((8, 6), (5, 3), True, True),
    ):
        arch_compat.tensorrt_version = lambda v=version: v
        floor = arch_compat.tensorrt_min_compute()
        check(
            f"TRT {version[0]}.{version[1]} floor is {expected_floor}",
            floor == expected_floor,
            f"got {floor}",
        )
        check(
            f"TRT {version[0]}.{version[1]}: Volta supported={vol}",
            arch_compat.tensorrt_supports((7, 0)) is vol,
        )
        check(
            f"TRT {version[0]}.{version[1]}: Pascal supported={pascal}",
            arch_compat.tensorrt_supports((6, 1)) is pascal,
        )
finally:
    arch_compat.tensorrt_version = real_trt_version
    arch_compat.tensorrt_version.cache_clear()

check(
    "with the live TensorRT, this machine is TRT-capable",
    arch_compat.tensorrt_supports_device(),
    f"min={arch_compat.tensorrt_min_compute()}",
)


# --- localized messages no longer hardcode 7.5 ----------------------------

REQUIRED = "wizard_gpu_compute_too_low"
for code in ("en", "ja", "ko", "th", "zh"):
    module = __import__(f"jasna.gui.locales.{code}", fromlist=["_"])
    strings = next(
        value for value in vars(module).values()
        if isinstance(value, dict) and "wizard_check_gpu" in value
    )
    message = strings.get(REQUIRED, "")
    check(
        f"{code}: gate message uses the real floor",
        "{required}" in message and "7.5" not in message,
        message,
    )
    for key in ("wizard_check_os", "wizard_check_gpu_kernels", "wizard_gpu_kernels_ok",
                "wizard_gpu_kernels_missing"):
        check(f"{code}: has {key}", key in strings)

print("", file=STDOUT)
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}", file=STDOUT)
    raise SystemExit(1)
print("All GPU gate checks passed.", file=STDOUT)
