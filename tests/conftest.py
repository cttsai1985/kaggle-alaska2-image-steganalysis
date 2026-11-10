"""Public CPU fixtures share the exact builder used by the executed walkthrough."""

from pathlib import Path

import pytest

from tools.fixtures import build_fixture


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory) -> dict:
    return build_fixture(tmp_path_factory.mktemp("alaska2_fixture"))


@pytest.fixture
def fixture_cfg(fixture_data):
    from alaska2.config import load_config

    def make(*extra: str) -> dict:
        repo = Path(__file__).resolve().parents[1]
        return load_config(
            repo / "configs" / "fixture_cpu.toml", [f'data.root="{fixture_data["root"]}"', *extra]
        )

    return make
