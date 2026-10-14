"""First-run dependency checks, run off the UI thread by the wizard."""

import logging
import os
import subprocess
from collections.abc import Iterable

from jasna import os_utils
from jasna.gui.locales import t

logger = logging.getLogger(__name__)

WARNING_ONLY_CHECKS = {"sysmem", "os", "gpu_kernels"}


def evaluate_check_results(
    results: dict[str, tuple[bool, str]], keys: Iterable[str]
) -> tuple[bool, bool]:
    """Return ``(all_passed, has_required_failure)`` for the displayed check keys.

    A key missing from ``results`` counts as a failure: a check that never ran (e.g. the
    check thread died early) must read as failed, never as passed — otherwise the wizard
    shows red rows while still reporting "ready to use" and enabling Get Started."""
    passed = {key: results.get(key, (False, ""))[0] for key in keys}
    all_passed = all(passed.values())
    has_required_failure = any(
        not ok and key not in WARNING_ONLY_CHECKS for key, ok in passed.items()
    )
    return all_passed, has_required_failure


def run_system_checks(results: dict[str, tuple[bool, str]]) -> None:
    """Fill ``results`` one check at a time, so a caller that stops waiting still sees the finished ones."""
    results["ascii_path"] = os_utils.check_ascii_install_path()
    results["ffprobe"] = check_ffprobe()
    results["gpu"] = check_gpu()
    results["cuda"] = check_cuda()
    results["gpu_kernels"] = check_gpu_kernels()
    results["driver"] = os_utils.check_gpu_driver_version()
    if os.name == "nt":
        results["os"] = os_utils.check_windows_version()
        results["sysmem"] = os_utils.check_windows_nvidia_sysmem_fallback_policy()


def check_ffprobe() -> tuple[bool, str]:
    path = os_utils.find_executable("ffprobe")
    if not path:
        return False, t("wizard_not_found")
    completed = subprocess.run(
        [path, "-version"],
        capture_output=True,
        text=True,
        check=False,
        **os_utils.subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        logger.error(
            "ffprobe failed (exit code %s). stdout:\n%s\nstderr:\n%s",
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )
        return False, t("wizard_not_callable", path=path)
    try:
        major = os_utils._parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
    except ValueError:
        return False, t("wizard_found_no_major", path=path)
    if major < os_utils.FFMPEG_MIN_MAJOR:
        return False, t("wizard_found_bad_major", path=path, major=major)
    return True, t("wizard_found_major", path=path, major=major)


def check_gpu() -> tuple[bool, str]:
    try:
        # Judged on the card this process will actually use, not on index 0: a
        # multi-GPU machine may well have the weak card first in the list.
        ok, result = os_utils.check_supported_gpu()
        if ok:
            return True, result
        if result == "no_cuda":
            return False, t("wizard_no_cuda")
        if result[0] == "arch_unsupported":
            return False, t("wizard_gpu_arch_unsupported", arch=result[1])
        _, major, minor = result
        floor = os_utils.minimum_compute()
        return False, t(
            "wizard_gpu_compute_too_low",
            required=f"{floor[0]}.{floor[1]}",
            major=major,
            minor=minor,
        )
    except Exception as e:
        return False, str(e)


def check_gpu_kernels() -> tuple[bool, str]:
    """Whether the shipped CUDA binaries were built for this card at all.

    Reported as a warning, not a gate: passing the capability check says the card
    is new enough, while this says whether the build can actually execute on it.
    A GTX 10-series card clears the former and fails the latter, because the CUDA
    13 wheels start at Turing (7.5) and TensorRT supports SM 7.5+ only.
    """
    try:
        covered, _detail = os_utils.check_kernel_coverage()
        try:
            major, minor = os_utils._compute_capability(os_utils.compute_device_string())
            version = f"{major}.{minor}"
        except Exception:
            version = "?"
        if covered:
            return True, t("wizard_gpu_kernels_ok", version=version)
        built = ", ".join(os_utils.built_cuda_arches()) or t("wizard_not_available")
        return False, t("wizard_gpu_kernels_missing", version=version, built=built)
    except Exception as e:
        return True, str(e)


def check_cuda() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            version = torch.version.hip or torch.version.cuda
            if torch.version.hip:
                return True, f"ROCm {version}"
            capability = torch.cuda.get_device_capability(os_utils.compute_device_string())
            return True, t(
                "wizard_cuda_version_compute",
                version=version,
                major=capability[0],
                minor=capability[1],
            )
        return False, t("wizard_not_available")
    except Exception as e:
        return False, str(e)
