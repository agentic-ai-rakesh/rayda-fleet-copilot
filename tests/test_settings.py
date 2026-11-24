from pathlib import Path

import pytest

from fleet_copilot.config.settings import load_settings
from fleet_copilot.exceptions import ConfigError


def test_missing_file_falls_back_to_defaults(tmp_path: Path):
    settings = load_settings(tmp_path / "absent.yaml", project_root=tmp_path)
    assert settings.ingestion.max_reject_ratio == 0.05
    assert settings.paths.fleet_db == tmp_path / "data" / "processed" / "fleet.db"


def test_relative_paths_resolve_from_project_root(tmp_path: Path):
    config = tmp_path / "settings.yaml"
    config.write_text("paths:\n  fleet_db: custom/fleet.db\n")
    settings = load_settings(config, project_root=tmp_path)
    assert settings.paths.fleet_db == tmp_path / "custom" / "fleet.db"


@pytest.mark.parametrize(
    "content",
    [
        "ingestion:\n  max_reject_ratio: 2\n",  # out of range
        "ingestion:\n  max_reject_ratoi: 0.1\n",  # typo -> unknown key
        "- just\n- a list\n",  # wrong top-level type
        "paths: [unclosed\n",  # not valid YAML
    ],
)
def test_invalid_settings_fail_loudly(tmp_path: Path, content: str):
    config = tmp_path / "settings.yaml"
    config.write_text(content)
    with pytest.raises(ConfigError):
        load_settings(config, project_root=tmp_path)
