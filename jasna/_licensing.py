"""Access to the optional, private licensing module (``jasna.protection``).

Only the maintainer's release builds contain ``jasna.protection``; the public
source checkout does not, and neither does anything built from it. Licensing
state is nevertheless read on paths that always run — the GUI paints a
"supporter" chip while it builds its status bar — so every access has to degrade
gracefully instead of raising ``ImportError``.

Use :func:`license_store` for the store, :func:`protection_module` when the
richer API is needed, and :func:`protection_error_types` when building ``except``
clauses, which cannot be wrapped in a try block.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_MODULE_NAME = "jasna.protection"


class LicenseUnavailableError(RuntimeError):
    """Raised when a supporter feature is used in a build without licensing."""


class _UnavailableLicenseStore:
    """Stand-in for ``license_store`` when the private module is absent."""

    available = False

    def load_license(self) -> tuple[str, str] | None:
        return None

    def is_licensed(self) -> bool:
        return False

    def set_license(self, email: str, key: str) -> None:
        raise LicenseUnavailableError(
            "This build has no licensing module, so supporter keys cannot be "
            "activated. Free models and all normal processing are unaffected."
        )


_unavailable_store = _UnavailableLicenseStore()
_module: Any = None
_resolved = False


def protection_module() -> Any | None:
    """The private module, or ``None`` when this build does not include it."""
    global _module, _resolved
    if _resolved:
        return _module
    _resolved = True
    try:
        import importlib

        _module = importlib.import_module(_MODULE_NAME)
    except ImportError:
        _module = None
        log.info(
            "Optional licensing module %s is not present; supporter features are "
            "unavailable in this build.",
            _MODULE_NAME,
        )
    return _module


def protection_available() -> bool:
    return protection_module() is not None


def license_store() -> Any:
    """The real licence store, or a safe stand-in that reports 'not licensed'."""
    module = protection_module()
    if module is None:
        return _unavailable_store
    store = getattr(module, "license_store", None)
    return store if store is not None else _unavailable_store


def protection_error_types() -> tuple[type[BaseException], ...]:
    """Exception classes from the private module, for ``except`` clauses."""
    module = protection_module()
    if module is None:
        return (LicenseUnavailableError,)
    names = (
        "ProtectionError",
        "LicenseError",
        "MalformedLicenseError",
        "ForgedLicenseError",
        "RetiredLicenseError",
    )
    found = tuple(
        getattr(module, name) for name in names if isinstance(getattr(module, name, None), type)
    )
    return found or (LicenseUnavailableError,)


def is_licensed() -> bool:
    return bool(license_store().is_licensed())
