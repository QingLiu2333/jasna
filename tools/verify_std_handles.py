"""Verify that child processes survive a broken standard handle.

The bug this guards against. After ``FreeConsole()`` the process' standard handles
are dead, so ``_redirect_std_streams_to_null()`` re-points them at NUL. Anything that
then frees one of those NUL handles again - a stray ``CloseHandle``, or the stream
object being collected - leaves the kernel's ``STD_*_HANDLE`` entry pointing at a
released value. ``subprocess`` resolves a stream the caller left as ``None`` through
``GetStdHandle`` and duplicates it, so the spawn dies with
``[WinError 6] The handle is invalid`` before the child starts. In the frozen GUI the
caller is ``probe.get_video_meta_data``, which leaves ``stdin`` unset, so every job
stops on its first frame.

A released handle value may be recycled before the spawn reaches it, which is why the
real failure looks intermittent; this script uses a permanently invalid value instead
(an odd number - Windows handle values are always 4-byte aligned, so it can never be a
real handle and can never be re-issued) to make the reproduction deterministic.

Checks:
  1. a healthy spawn works at all
  2. an unusable STD_INPUT_HANDLE reproduces the user-visible ``[WinError 6]``
  3. ``subprocess_no_window_kwargs()`` repairs it and the same spawn succeeds
  4. the detector recognises the unusable handle, and stops complaining once repaired
  5. the returned kwargs stay compatible with the call sites that pass their own
     ``stdin`` (a duplicate keyword would be a TypeError)
  6. the NUL streams stay alive when ``sys.stdin`` is rebound and collected

Usage:
    python tools/verify_std_handles.py
"""
from __future__ import annotations

import ctypes
import gc
import os
import subprocess
import sys

sys.path.insert(0, r"D:\jasna2\work")

from jasna import os_utils

# Odd on purpose: never a valid handle, never recycled, so the failure is reproducible
# rather than dependent on whether the released value was handed out again.
UNUSABLE_HANDLE = 0xDEAD0001

FAILURES: list[str] = []
STDOUT = sys.__stdout__

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
_kernel32.SetStdHandle.restype = ctypes.c_int


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}", file=STDOUT)
    if not ok:
        FAILURES.append(name)


def set_stdin_handle(value: int | None) -> None:
    _kernel32.SetStdHandle(os_utils.STD_INPUT_HANDLE, ctypes.c_void_p(value))


def spawn_with(**kwargs) -> str:
    try:
        proc = subprocess.Popen(
            ["cmd", "/c", "echo hi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        proc.communicate()
        return f"ok rc={proc.returncode}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# Window suppression only - no repair, so this is the state a child sees before the fix.
bare = {
    "startupinfo": os_utils.get_subprocess_startup_info(),
    "creationflags": subprocess.CREATE_NO_WINDOW,
}

# 1 - healthy baseline
result = spawn_with(**bare)
check("healthy spawn succeeds", result == "ok rc=0", result)

# 2 - an unusable standard input handle reproduces the reported failure
set_stdin_handle(UNUSABLE_HANDLE)
result = spawn_with(**bare)
check(
    "unusable STD_INPUT_HANDLE reproduces [WinError 6]",
    "WinError 6" in result or "句柄无效" in result,
    result,
)

# 3 - the repair hook makes the very same spawn work again
kwargs = os_utils.subprocess_no_window_kwargs()
result = spawn_with(**kwargs)
check("repair hook fixes the spawn", result == "ok rc=0", result)
print(f"       STD_INPUT_HANDLE after repair: {hex(os_utils._kernel32_api().GetStdHandle(-10) or 0)}", file=STDOUT)

# 4 - the detector agrees, and clears once the handle is usable again
set_stdin_handle(UNUSABLE_HANDLE)
check("detector reports the handle unusable", not os_utils._standard_handle_is_usable(os_utils.STD_INPUT_HANDLE))
os_utils.ensure_standard_handles()
check("detector clears after repair", os_utils._standard_handle_is_usable(os_utils.STD_INPUT_HANDLE))

# 5 - call sites that pass their own stdin must not collide with the kwargs
kwargs = os_utils.subprocess_no_window_kwargs()
check("kwargs carry no 'stdin' key", "stdin" not in kwargs, f"keys={sorted(kwargs)}")
try:
    proc = subprocess.Popen(
        ["cmd", "/c", "echo piped"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    proc.communicate(b"")
    check("explicit stdin= still combines with the kwargs", proc.returncode == 0)
except TypeError as exc:
    check("explicit stdin= still combines with the kwargs", False, str(exc))

# 6 - the NUL streams survive a rebind of sys.stdin (the module-level reference)
saved_streams = (sys.stdin, sys.stdout, sys.stderr)
which_all = (os_utils.STD_INPUT_HANDLE, os_utils.STD_OUTPUT_HANDLE, os_utils.STD_ERROR_HANDLE)
try:
    os_utils._redirect_std_streams_to_null()
    check("redirect_std_streams_to_null yields usable handles", all(os_utils._standard_handle_is_usable(w) for w in which_all))

    set_stdin_handle(UNUSABLE_HANDLE)
    sys.stdin = open(os.devnull, "r")
    gc.collect()
    check("unusable handle is still repaired after rebinding sys.stdin", spawn_with(**os_utils.subprocess_no_window_kwargs()) == "ok rc=0")
    check("NUL streams keep valid fds", os_utils._stream_has_valid_fd(os_utils._NULL_STDIN) and os_utils._stream_has_valid_fd(os_utils._NULL_STDOUT))
finally:
    sys.stdin, sys.stdout, sys.stderr = saved_streams

print("", file=STDOUT)
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}", file=STDOUT)
    raise SystemExit(1)
print("All standard-handle checks passed.", file=STDOUT)
