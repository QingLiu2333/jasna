"""Reproduce the frozen-GUI standard-handle state that breaks ffprobe.

A double-click launch of the frozen ``jasna.exe`` follows the GUI branch of
``jasna/__main__.py``::

    drop_console_window()          -> FreeConsole()
                                   -> _redirect_std_streams_to_null()

``_redirect_std_streams_to_null`` exists because ``subprocess.Popen`` resolves
every stream the caller left as ``None`` through ``GetStdHandle``; after
``FreeConsole`` those handles are dead and ``DuplicateHandle`` fails with
``[WinError 6] The handle is invalid``.  ``probe.get_video_meta_data`` leaves
``stdin`` unset, so it is the stdin lookup that dies.

This script emulates that exact sequence, runs the TensorRT engine-load capture
that was added in this round, and finishes with the very ``Popen`` call that
``probe.get_video_meta_data`` makes.  Run it with a python that has no console
of its own to be faithful - see the report for what each step leaves behind.

Usage:
    python tools/repro_std_handles.py <report.json>
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import sys
from ctypes import wintypes

REPORT_PATH = sys.argv[1] if len(sys.argv) > 1 else "std_handle_report.json"
sys.path.insert(0, r"D:\jasna2\work")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetStdHandle.argtypes = [wintypes.DWORD]
k32.GetStdHandle.restype = wintypes.HANDLE
k32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
k32.SetStdHandle.restype = wintypes.BOOL
k32.GetCurrentProcess.argtypes = []
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.DuplicateHandle.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
    ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
]
k32.DuplicateHandle.restype = wintypes.BOOL
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CloseHandle.restype = wintypes.BOOL

PROC = k32.GetCurrentProcess()
STD_STREAMS = (("stdin", -10), ("stdout", -11), ("stderr", -12))


def check() -> dict:
    """Report, for each standard stream, whether a child could inherit it."""
    result: dict = {}
    for name, which in STD_STREAMS:
        handle = k32.GetStdHandle(which)
        ctypes.set_last_error(0)
        duplicate = wintypes.HANDLE()
        ok = k32.DuplicateHandle(PROC, handle, PROC, ctypes.byref(duplicate), 0, True, 0x00000002)
        err = ctypes.get_last_error()
        if ok:
            k32.CloseHandle(duplicate)
        result[name] = {
            "handle": None if handle is None else int(handle),
            "duplicate_ok": bool(ok),
            "duplicate_error": 0 if ok else err,
        }
    for fd in (0, 1, 2):
        try:
            os.fstat(fd)
            result[f"crt_fd_{fd}"] = "valid"
        except OSError as exc:
            result[f"crt_fd_{fd}"] = f"{type(exc).__name__}(errno={exc.errno})"
    return result


def probe_popen() -> str:
    """Exactly what probe.get_video_meta_data does: stdin left unset."""
    try:
        p = subprocess.Popen(
            ["cmd", "/c", "echo hi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p.communicate()
        return f"ok rc={p.returncode}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


report: dict = {}
try:
    report["1_attached_console"] = check()
    report["1_popen"] = probe_popen()

    k32.FreeConsole()
    report["2_after_freeconsole"] = check()

    from jasna.os_utils import _redirect_std_streams_to_null

    _redirect_std_streams_to_null()
    report["3_after_redirect_std_streams_to_null"] = check()
    report["3_popen"] = probe_popen()

    try:
        from jasna.trt.torch_tensorrt_export import _capture_process_stderr
    except Exception as exc:  # pragma: no cover - environment problem
        report["4_import_capture"] = f"{type(exc).__name__}: {exc}"
    else:
        report["4_import_capture"] = "ok"
        try:
            with _capture_process_stderr() as lines:
                pass
            report["5_engine_load_capture"] = {"ran": True, "captured_lines": lines}
        except Exception as exc:
            report["5_engine_load_capture"] = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
        report["6_after_capture"] = check()
        report["6_popen"] = probe_popen()

    # The shipped NUL streams are only kept alive by sys.std*.  Replacing sys.stdin
    # drops the last reference; CPython closes the file and its handle with it,
    # while the kernel's STD_INPUT_HANDLE keeps pointing at the freed value.
    sys.stdin = open(os.devnull, "r")
    gc.collect()
    report["7_after_sys_stdin_replaced"] = check()
    report["7_popen"] = probe_popen()
except Exception as exc:  # pragma: no cover - always leave a report behind
    report["fatal"] = f"{type(exc).__name__}: {exc}"

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)
