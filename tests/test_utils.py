from datetime import UTC, datetime, timedelta, timezone

import pytest

from fleet_copilot.utils.time_utils import ensure_utc, to_compact_utc, to_iso_utc
from fleet_copilot.utils.unit_utils import bytes_to_gb, percent
from fleet_copilot.utils.version_utils import OsVersion, parse_os_version


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("14.5", OsVersion(major=14, minor=5, patch=0)),
        ("15.4.1", OsVersion(major=15, minor=4, patch=1)),
        ("13", OsVersion(major=13, minor=0, patch=0)),
        ("11 23H2", OsVersion(major=11, release="23H2")),
        ("10 22h2", OsVersion(major=10, release="22H2")),
        ("Sonoma beta", OsVersion()),
        ("", OsVersion()),
        (None, OsVersion()),
    ],
)
def test_parse_os_version(raw, expected):
    assert parse_os_version(raw) == expected


def test_unparsed_version_is_flagged():
    assert not parse_os_version("unknown").is_parsed
    assert parse_os_version("14.5").is_parsed


def test_macos_versions_compare_numerically():
    assert parse_os_version("13.6").major < 15 <= parse_os_version("15.3").major


def test_percent_handles_missing_and_zero():
    assert percent(25, 100) == 25.0
    assert percent(1, 3) == 33.33
    assert percent(None, 100) is None
    assert percent(5, 0) is None


def test_bytes_to_gb():
    assert bytes_to_gb(34359738368) == 32.0
    assert bytes_to_gb(None) is None


def test_time_helpers_normalise_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    local = datetime(2026, 5, 14, 15, 24, tzinfo=ist)
    assert to_iso_utc(local) == "2026-05-14T09:54:00Z"
    assert to_compact_utc(local) == "20260514T0954Z"
    assert ensure_utc(datetime(2026, 1, 1)).tzinfo == UTC


def test_linear_slope_and_daily_slope():
    from fleet_copilot.utils.stats_utils import daily_slope, linear_slope, share

    assert linear_slope([0, 1, 2], [10, 8, 6]) == -2.0
    assert linear_slope([1, 1], [3, 4]) is None
    assert linear_slope([1], [3]) is None
    assert daily_slope(["2026-05-01T00:00:00Z", "2026-05-03T00:00:00Z"], [10, 6]) == -2.0
    assert share([True, False, True, True]) == 0.75
    assert share([]) == 0.0
