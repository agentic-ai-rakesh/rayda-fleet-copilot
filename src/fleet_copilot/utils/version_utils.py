"""Parse OS version strings into comparable parts.

Examples from the dataset:
    macOS   "14.5"     -> major=14, minor=5, patch=0
    Windows "11 23H2"  -> major=11, release="23H2"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DOTTED_VERSION = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$")
_WINDOWS_RELEASE = re.compile(r"^(\d+)\s+([0-9]{2}H[12])$", re.IGNORECASE)


@dataclass(frozen=True)
class OsVersion:
    major: int | None = None
    minor: int | None = None
    patch: int | None = None
    release: str | None = None  # Windows feature release, e.g. "23H2"

    @property
    def is_parsed(self) -> bool:
        return self.major is not None


def parse_os_version(version: str | None) -> OsVersion:
    """Return the parsed version, or an empty OsVersion when the format is unknown."""
    if not version:
        return OsVersion()
    text = version.strip()

    dotted = _DOTTED_VERSION.match(text)
    if dotted:
        major, minor, patch = dotted.groups()
        return OsVersion(major=int(major), minor=int(minor or 0), patch=int(patch or 0))

    windows = _WINDOWS_RELEASE.match(text)
    if windows:
        major, release = windows.groups()
        return OsVersion(major=int(major), release=release.upper())

    return OsVersion()
