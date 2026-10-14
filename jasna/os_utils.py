import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from jasna._frozen import is_frozen

logger = logging.getLogger(__name__)

MIN_GPU_COMPUTE = (6, 1)
# Pascal (6.1) is the floor: every consumer GTX 10-series card is 6.1, and the
# build's CUDA kernels have to cover it. What a 6.1 card can actually run is a
# separate question - see ``check_kernel_coverage`` - because TensorRT is SM 7.5+
# only and the CUDA 13 PyTorch wheels carry no Pascal kernels, so a 10-series
# card gets the fp32 PyTorch path.
# ``JASNA_ALLOW_LEGACY_GPU=1`` lowers the gate to 6.0 for GP100/Tesla P100 parts.
MIN_GPU_COMPUTE_LEGACY = (6, 0)
ALLOW_LEGACY_GPU_ENV = "JASNA_ALLOW_LEGACY_GPU"
# CUDA 13.x requires driver R580 or newer, and R580 is the last driver branch
# that exists for Maxwell/Pascal/Volta - so 580 is both the floor CUDA needs and
# the ceiling those cards can reach. Windows 10 tops out here too: R590/R610/R615
# are Windows 11 / Windows 10 22H2-and-later branches, which is why a 610 floor
# locked every Windows 10 machine out.
MIN_DRIVER_VERSION = 580

# The media stack (PyAV) bundles FFmpeg 8.x, and ``ffprobe`` supplies the video
# metadata in the same JSON shape from 8 onward. Anything older predates the
# current CLI/JSON behaviour; anything newer (a git build of 9, say) is
# compatible, so the gate is a floor rather than an exact match.
FFMPEG_MIN_MAJOR = 8

def minimum_compute() -> tuple[int, int]:
    """The compute-capability floor this run enforces.

    Pascal (6.1) by default, i.e. every consumer GTX 10-series card is accepted.
    ``JASNA_ALLOW_LEGACY_GPU=1`` lowers it to 6.0 for the GP100/P100 parts. Note
    the floor is only a *gate*: whether the build actually has kernels for the
    card is a separate question answered by :func:`check_kernel_coverage`.
    """
    import os as _os

    if _os.environ.get(ALLOW_LEGACY_GPU_ENV, "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        return MIN_GPU_COMPUTE_LEGACY
    return MIN_GPU_COMPUTE


def built_cuda_arches() -> list[str]:
    """The ``sm_XX`` targets this build's CUDA runtime was compiled for."""
    try:
        import torch

        return [entry for entry in torch.cuda.get_arch_list() if entry.startswith("sm_")]
    except Exception:
        return []


def _compute_capability(device: str) -> tuple[int, int]:
    import torch

    return tuple(torch.cuda.get_device_capability(device))


def check_kernel_coverage(device: str | None = None) -> tuple[bool, str]:
    """Does this build's CUDA runtime actually contain kernels for *device*?

    A capability gate says "this card is new enough"; it says nothing about
    whether the shipped PyTorch/CUDA binaries were compiled for it. The CUDA 13
    wheels cover Turing (7.5) and newer only, so a Pascal card that passes the
    gate would still die later with *"no kernel image is available for execution
    on the device"* — mid-job, with no hint about the cause.

    Coverage follows CUDA's binary-compatibility rule: a cubin built for sm_XY
    runs on sm_XZ when Z >= Y (higher minor, same major). Anything across a major
    version needs PTX, which these wheels do not carry for older architectures,
    so it is reported as uncovered.

    Returns ``(covered, detail)`` with a human-readable ``detail`` either way.
    """
    try:
        import torch
        from jasna.accelerator import AcceleratorVendor, vendor_for_device
    except ImportError:
        return True, ""

    if not torch.cuda.is_available():
        return True, ""
    device = device or compute_device_string()
    if vendor_for_device(device) is AcceleratorVendor.AMD:
        # Handled separately by check_supported_gpu (arch_unsupported).
        return True, ""

    built = built_cuda_arches()
    if not built:
        return True, ""

    major, minor = _compute_capability(device)
    parsed: list[tuple[int, int]] = []
    for entry in built:
        digits = entry[3:]
        if len(digits) >= 2 and digits.isdigit():
            parsed.append((int(digits[:-1]), int(digits[-1])))
    if not parsed:
        return True, ""

    if any(built_major == major and built_minor <= minor for built_major, built_minor in parsed):
        return True, "covered by " + ", ".join(entry for entry in built)

    shown = ", ".join(entry for entry in built)
    return False, f"the build has kernels for {shown}, not for {major}.{minor}"


def check_supported_gpu(
    device: str | None = None,
) -> tuple[bool, str] | tuple[bool, tuple[str, int, int]]:
    """Validate the GPU supported by the current vendor-specific build.

    ``device`` defaults to the compute device this process selected, so a
    multi-GPU machine is judged on the card it will actually use.
    """
    try:
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
        from jasna.accelerator import AcceleratorVendor, vendor_for_device
    except ImportError:
        return False, "no_cuda"
    if not torch.cuda.is_available():
        return False, "no_cuda"
    device = device or compute_device_string()
    if vendor_for_device(device) is AcceleratorVendor.AMD:
        # A ROCm build has no kernels for a GPU arch it was not compiled for, and on
        # Windows the first kernel launch then crashes the process (issue #408).
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        built_archs = {a.split(":")[0] for a in torch.cuda.get_arch_list()}
        if built_archs and arch not in built_archs:
            return False, ("arch_unsupported", arch)
        return True, torch.cuda.get_device_name(device)
    capability = _compute_capability(device)
    if capability < minimum_compute():
        return False, ("compute_too_low", capability[0], capability[1])
    return True, torch.cuda.get_device_name(device)


def gpu_check_error(result: str | tuple) -> str:
    """English CLI message for a failed check_supported_gpu result."""
    if result == "no_cuda":
        return "No compatible GPU was found for this Jasna build."
    if result[0] == "arch_unsupported":
        return f"This AMD GPU ({result[1]}) is not supported by this Jasna build."
    _, major, minor = result
    floor = minimum_compute()
    return (
        f"Compute capability {floor[0]}.{floor[1]}+ required (GPU: {major}.{minor}). "
        f"Cards below that (Maxwell, compute capability 5.x) cannot run this build."
    )


# ---------------------------------------------------------------------------
# Which GPU this process uses
# ---------------------------------------------------------------------------

_compute_device_index = 0


def set_compute_device_index(index: int) -> int:
    """Pin the CUDA device for the whole process and return the index applied.

    An out-of-range or non-numeric index falls back to 0 rather than raising:
    a saved selection must never be able to stop the app from starting.
    """
    global _compute_device_index
    try:
        resolved = max(0, int(index))
    except (TypeError, ValueError):
        resolved = 0
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if resolved >= count:
                resolved = 0
            torch.cuda.set_device(resolved)
    except Exception:
        logger.debug("Could not select CUDA device %s", resolved, exc_info=True)
    try:
        from jasna.engine_paths import set_engine_device_index

        set_engine_device_index(resolved)
    except Exception:
        logger.debug("Could not pin the engine archive device", exc_info=True)
    try:
        # The capability probes are cached, so switching cards has to invalidate
        # them: fp16/int8/FP8 answers for one GPU are wrong for another.
        from jasna.arch_compat import reset_capability_cache

        reset_capability_cache()
    except Exception:
        logger.debug("Could not reset the capability cache", exc_info=True)
    _compute_device_index = resolved
    return resolved


def compute_device_index() -> int:
    """The CUDA device index this process uses. 0 until something selects one."""
    return _compute_device_index


def compute_device_string() -> str:
    """``"cuda:N"`` for the selected device, for APIs that take a device string."""
    return f"cuda:{_compute_device_index}"


def compute_device():
    """``torch.device`` for the selected device.

    Every GPU path should resolve its device through this instead of spelling
    ``"cuda:0"``: on a multi-GPU machine the literal would silently ignore the
    user's selection in the settings panel.
    """
    import torch

    return torch.device("cuda", _compute_device_index)


def _bundled_exe_filename(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def _find_bundled_executable(name: str) -> Path | None:
    if not is_frozen():
        return None

    exe = _bundled_exe_filename(name)
    # Nuitka standalone has no _internal/ (a PyInstaller-ism); bundled tools sit at the
    # dist root next to the binary, alongside model_weights/ and assets/.
    base = Path(sys.executable).parent

    if name in {"ffmpeg", "ffprobe"}:
        p = base / "tools" / exe
        return p if p.is_file() else None

    return None


_COMMON_EXECUTABLE_LOCATIONS: dict[str, tuple[str, ...]] = {
    "nvidia-smi": (
        # Windows
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        # Linux / WSL
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",
    ),
}


def _find_in_common_locations(name: str) -> str | None:
    for candidate in _COMMON_EXECUTABLE_LOCATIONS.get(name, ()):
        if os.path.isfile(candidate):
            return candidate
    return None


def find_executable(name: str) -> str | None:
    bundled = _find_bundled_executable(name)
    if bundled is not None:
        return str(bundled)
    found = shutil.which(name)
    if found is not None:
        return found
    return _find_in_common_locations(name)


def resolve_executable(name: str) -> str:
    found = find_executable(name)
    return found if found is not None else name



def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12

# The NUL streams handed to the kernel as the standard handles. They are kept at
# module level on purpose: if the only reference to one is ``sys.stdin``, then
# anything that rebinds ``sys.stdin`` collects the stream, CPython closes its
# handle, and the kernel's STD_INPUT_HANDLE is left pointing at the freed value.
# The next child process then dies with ``[WinError 6] The handle is invalid``.
_NULL_STDIN = None
_NULL_STDOUT = None

_kernel32_api_cache = None


def _stream_has_valid_fd(stream) -> bool:
    try:
        os.fstat(stream.fileno())
    except (OSError, ValueError):
        return False
    return True


def _null_streams():
    """The module-level NUL streams, re-opened if their fd was closed underneath us."""
    global _NULL_STDIN, _NULL_STDOUT
    if _NULL_STDIN is None or not _stream_has_valid_fd(_NULL_STDIN):
        _NULL_STDIN = open(os.devnull, "r")
    if _NULL_STDOUT is None or not _stream_has_valid_fd(_NULL_STDOUT):
        _NULL_STDOUT = open(os.devnull, "w")
    return _NULL_STDIN, _NULL_STDOUT


def _kernel32_api():
    """kernel32 with explicit signatures, so handle-sized values are not truncated.

    ctypes' default is to pass a plain Python int as a C ``int``, which silently
    cuts a 64-bit handle value in half.
    """
    global _kernel32_api_cache
    if _kernel32_api_cache is None:
        import ctypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.GetStdHandle.argtypes = [ctypes.c_uint32]
        api.GetStdHandle.restype = ctypes.c_void_p
        api.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        api.SetStdHandle.restype = ctypes.c_int
        api.GetHandleInformation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        api.GetHandleInformation.restype = ctypes.c_int
        _kernel32_api_cache = api
    return _kernel32_api_cache


def _standard_handle_is_usable(which: int) -> bool:
    """True when a child process could inherit the standard handle for *which*.

    A handle closed behind the kernel's back still compares as non-NULL, so the
    only reliable test is to ask the kernel about it — with the same call that
    ``subprocess._make_inheritable`` uses and fails on.
    """
    if os.name != "nt":
        return True
    import ctypes

    api = _kernel32_api()
    handle = api.GetStdHandle(which)
    if not handle or handle == ctypes.c_void_p(-1).value:
        return False
    flags = ctypes.c_uint32()
    return bool(api.GetHandleInformation(handle, ctypes.byref(flags)))


def _point_standard_handle_at_null(which: int, stream) -> bool:
    import ctypes
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(stream.fileno())
    except OSError:
        return False
    if handle in (-1, None) or handle == ctypes.c_void_p(-1).value:
        return False
    return bool(_kernel32_api().SetStdHandle(which, ctypes.c_void_p(handle)))


def ensure_standard_handles() -> None:
    """Repair any standard handle that no longer points at something usable.

    ``subprocess`` resolves every stream the caller left as ``None`` through
    ``GetStdHandle`` and immediately duplicates it. On a frozen GUI launch the
    inherited console handles die with ``FreeConsole``, and later a stray
    ``CloseHandle`` — or a collected stream object — can free the NUL handle that
    replaced them, leaving ``STD_INPUT_HANDLE``/``STD_OUTPUT_HANDLE``/
    ``STD_ERROR_HANDLE`` pointing at freed memory. The duplicate then fails and the
    spawn dies with ``[WinError 6] The handle is invalid`` before the child starts:
    ffprobe never runs, so every job stops on its first frame.

    ``subprocess_no_window_kwargs()`` calls this, and every tool spawn goes through
    it, so a child is only ever started against handles that work — repaired at the
    last possible moment, after all the torch/CUDA/TensorRT handle churn. Handles
    that are still valid are left alone, so a CLI run that redirected its own output
    keeps that redirection.
    """
    if os.name != "nt":
        return

    null_r, null_w = _null_streams()
    for which, stream in (
        (STD_INPUT_HANDLE, null_r),
        (STD_OUTPUT_HANDLE, null_w),
        (STD_ERROR_HANDLE, null_w),
    ):
        if _standard_handle_is_usable(which):
            continue
        if _point_standard_handle_at_null(which, stream):
            logger.debug("Standard handle %s was unusable; re-pointed it at NUL", which)


def _redirect_std_streams_to_null() -> None:
    """Point sys.stdout/stderr/stdin AND the process's OS-level standard handles at NUL.

    After FreeConsole the inherited console handles are dead, and two things break.
    First, a stray `print()` or a logging StreamHandler writing to the real stream raises
    `OSError: [WinError 6] The handle is invalid` — repointing sys.* at NUL discards those
    writes. Second — and this is what randomly fails the system check and later crashes
    ffprobe — `subprocess.Popen` calls `GetStdHandle()` for every stream the
    caller left as None (our tools run with stdout/stderr=PIPE but stdin unset), then
    `DuplicateHandle`s that dangling handle and dies with `[WinError 6]`/`[WinError 50]`.
    Repointing the OS std handles at NUL gives each child a valid, inheritable handle to
    duplicate; fixing sys.* alone does not.

    The NUL streams are held by this module, not just by ``sys.*`` — see
    ``_NULL_STDIN``. ``ensure_standard_handles()`` covers the case where a handle dies
    anyway, so neither mechanism has to be perfect on its own.
    """
    null_r, null_w = _null_streams()
    sys.stdin = null_r
    sys.stdout = null_w
    sys.stderr = null_w
    if os.name != "nt":
        return
    for which, stream in (
        (STD_INPUT_HANDLE, null_r),
        (STD_OUTPUT_HANDLE, null_w),
        (STD_ERROR_HANDLE, null_w),
    ):
        if not _point_standard_handle_at_null(which, stream):
            logger.warning("Could not point standard handle %s at NUL", which)


def drop_console_window() -> None:
    """Detach the console on a frozen GUI launch (Windows). The frozen binary is built
    console-subsystem (`--windows-console-mode=force`) so the CLI blocks cmd and stdout/stderr
    work; for a GUI/double-click launch we don't want that console window, so release it.
    No-op off Windows and in dev, where the console is the developer's own terminal.

    Set ``JASNA_GUI_CONSOLE=1`` to keep the console attached. That is the only
    way to see a startup traceback from a frozen GUI build, so it is worth
    keeping around for diagnostics."""
    if sys.platform != "win32" or not is_frozen():
        return
    if os.environ.get("JASNA_GUI_CONSOLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    import ctypes

    ctypes.windll.kernel32.FreeConsole()
    _redirect_std_streams_to_null()


def subprocess_no_window_kwargs() -> dict:
    """Popen/run kwargs that suppress a child's console window on Windows.

    The GUI drops its own console (FreeConsole), so a console-subsystem child like
    ffmpeg/ffprobe would otherwise pop its own cmd window. We capture their output
    via pipes, so they never need an inherited console. No-op (empty) off Windows.

    Every tool spawn goes through this helper, so it is also where the process's
    standard handles get repaired — see ``ensure_standard_handles``. A stream the
    caller left unset is resolved through ``GetStdHandle``, and a dead handle makes
    the spawn fail with ``[WinError 6] The handle is invalid`` before the child even
    starts. Note this deliberately does *not* inject a ``stdin`` key: three call
    sites (the streaming encoder, the TVAI restorer, the engine compiler) pass their
    own ``stdin`` and a duplicate keyword would be a TypeError.
    """
    if os.name != "nt":
        return {}
    ensure_standard_handles()
    return {
        "startupinfo": get_subprocess_startup_info(),
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


def _parse_ffmpeg_major_version(version_output: str) -> int:
    def parse_major_from_libavutil(out: str) -> int:
        m = re.search(r"(?m)^\s*libavutil\s+(\d+)\.", out)
        if not m:
            raise ValueError("Could not parse libavutil version from ffmpeg output")
        libavutil_major = int(m.group(1))
        return libavutil_major - 52

    first_line = version_output.splitlines()[0] if version_output else ""
    m = re.match(r"^\s*(?:ffmpeg|ffprobe)\s+version\s+(\S+)", first_line)
    if not m:
        raise ValueError(f"Unexpected ffmpeg/ffprobe version output: {first_line!r}")

    ver = m.group(1)
    if ver.startswith(("N-", "git-", "GIT-")):
        return parse_major_from_libavutil(version_output)

    m = re.search(r"(\d+)", ver)
    if not m:
        raise ValueError(f"Could not parse ffmpeg major version from: {ver!r}")
    return int(m.group(1))


FFMPEG_DOWNLOAD_LINKS = (
    "Linux: https://www.ffmpeg.org/download.html  "
    "Windows: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z"
)


def check_required_executables() -> None:
    """Check that required external tools are available in PATH and callable."""
    missing: list[str] = []
    wrong_version: list[str] = []
    checks = {
        "ffprobe": ["-version"],
    }
    for exe, args in checks.items():
        exe_path = find_executable(exe)
        if exe_path is None:
            missing.append(exe)
            continue
        cmd = [exe_path] + list(args)
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            missing.append(exe)
            continue
        if completed.returncode != 0:
            logger.error(
                "%s failed (exit code %s). stdout:\n%s\nstderr:\n%s",
                exe,
                completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
            )
            missing.append(exe)
            continue

        if exe == "ffprobe":
            try:
                major = _parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
            except ValueError:
                wrong_version.append(f"{exe} (could not detect major version)")
                continue
            if major < FFMPEG_MIN_MAJOR:
                wrong_version.append(f"{exe} (detected major={major})")

    if missing:
        msg = f"Error: Required executable(s) not found in PATH or not callable: {', '.join(missing)}"
        print(msg)
        logger.info("%s", msg)
        print("Please install them and ensure they are available in your system PATH and runnable.")
        logger.info("Please install them and ensure they are available in your system PATH and runnable.")
        if "ffprobe" in missing:
            print(FFMPEG_DOWNLOAD_LINKS)
            logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)
    if wrong_version:
        msg = (
            f"Error: ffprobe major version must be at least {FFMPEG_MIN_MAJOR} (or detectable as such): "
            + ", ".join(wrong_version)
        )
        print(msg)
        logger.info("%s", msg)
        print(FFMPEG_DOWNLOAD_LINKS)
        logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)


_CUDA_SYSMEM_FALLBACK_POLICY_ID = 0x10ECECC9
_PREFER_NO_SYSMEM_FALLBACK = 1
_PREFER_SYSMEM_FALLBACK = 2


def _read_drs_setting(setting_id: int) -> int | None:
    import struct

    drs_path = os.path.join(
        os.environ.get("ProgramData", r"C:\ProgramData"),
        "NVIDIA Corporation",
        "Drs",
        "nvdrsdb0.bin",
    )
    with open(drs_path, "rb") as f:
        data = f.read()
    target = struct.pack("<I", setting_id)
    pos = data.find(target)
    if pos < 0:
        return None
    # DRS binary record: ID(4) + type_flags(4) + value(4) + ...
    value_offset = pos + 8
    if value_offset + 4 > len(data):
        return None
    return struct.unpack("<I", data[value_offset : value_offset + 4])[0]


def check_windows_nvidia_sysmem_fallback_policy() -> tuple[bool, str]:
    if sys.platform != "win32":
        return True, "N/A"

    try:
        value = _read_drs_setting(_CUDA_SYSMEM_FALLBACK_POLICY_ID)
    except OSError as e:
        return False, str(e)

    if value == _PREFER_NO_SYSMEM_FALLBACK:
        return True, "Prefer No Sysmem Fallback"
    if value == _PREFER_SYSMEM_FALLBACK:
        return False, "Prefer Sysmem Fallback (recommended: Prefer No Sysmem Fallback)"
    return False, "Driver Default (recommended: Prefer No Sysmem Fallback)"


# Windows 10 21H2. The CUDA 13 / TensorRT 10 stack documents this as its floor,
# and it is also the oldest Windows 10 that still receives current NVIDIA drivers.
WINDOWS_MIN_BUILD = 19044

_WINDOWS_10_RELEASES = {
    19041: "Windows 10 2004",
    19042: "Windows 10 20H2",
    19043: "Windows 10 21H1",
    19044: "Windows 10 21H2",
    19045: "Windows 10 22H2",
}


def windows_version() -> tuple[int, int, int, str]:
    """``(major, minor, build, label)`` for the running Windows.

    ``sys.getwindowsversion()`` reads through ``RtlGetVersion``, so unlike the
    manifest-sensitive ``GetVersionEx`` it reports the real build even when the
    app carries no Windows 10 compatibility entry.
    """
    if sys.platform != "win32":
        return 0, 0, 0, sys.platform
    info = sys.getwindowsversion()
    build = int(getattr(info, "build", 0))
    if build >= 22000:
        label = f"Windows 11 (build {build})"
    elif build in _WINDOWS_10_RELEASES:
        label = _WINDOWS_10_RELEASES[build]
    elif build:
        label = f"Windows 10 (build {build})"
    else:
        label = "Windows"
    return int(info.major), int(info.minor), build, label


def check_windows_version() -> tuple[bool, str]:
    """Whether this Windows is new enough for the bundled CUDA/TensorRT stack.

    Reported as a warning rather than a gate: an older build usually still works,
    and the user is better served by knowing what they are on than by being
    stopped before the GPU checks can speak.
    """
    if sys.platform != "win32":
        return True, "N/A"
    _, _, build, label = windows_version()
    if build and build < WINDOWS_MIN_BUILD:
        return False, f"{label} (recommended: Windows 10 21H2 / build {WINDOWS_MIN_BUILD} or newer)"
    return True, label


def check_gpu_driver_version() -> tuple[bool, str]:
    import torch

    hip_version = torch.version.hip
    if hip_version:
        if not torch.cuda.is_available():
            return False, f"ROCm {hip_version} is installed but no AMD GPU is available"
        return True, f"ROCm {hip_version}"

    nvidia_smi = find_executable("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found"
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            **subprocess_no_window_kwargs(),
        )
    except OSError as e:
        return False, str(e)
    if result.returncode != 0:
        return False, f"nvidia-smi exited with code {result.returncode}"
    version_str = result.stdout.strip().split("\n")[0].strip()
    m = re.match(r"(\d+)\.(\d+)", version_str)
    if not m:
        return False, f"Could not parse driver version: {version_str!r}"
    major = int(m.group(1))
    if major < MIN_DRIVER_VERSION:
        return False, f"{version_str} (requires {MIN_DRIVER_VERSION}+)"
    return True, version_str


def check_ascii_install_path() -> tuple[bool, str]:
    if is_frozen():
        install_path = Path(sys.executable).parent
    else:
        install_path = Path(__file__).resolve().parent
    path_str = str(install_path)
    try:
        path_str.encode("ascii")
    except UnicodeEncodeError:
        return False, path_str
    return True, path_str


def get_user_config_dir(app_name: str) -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / app_name
        return Path.home() / app_name

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / app_name
    return Path.home() / ".config" / app_name
