"""Lab preparation is safe to repeat and the generated container contract preserves auth/isolation."""

import json
import stat
import subprocess
from pathlib import Path

import pytest

from tools.lab import prepare


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="alaska2"\nversion="0.1.0"\n')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


def test_prepare_reuses_secret_and_only_exposes_loopback(project):
    first = prepare(project)
    token_file = Path(first["token_file"])
    secret = token_file.read_text()
    second = prepare(project)
    assert len(secret.strip()) >= 32 and token_file.read_text() == secret
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert secret.strip() not in json.dumps(second)
    override = json.loads(Path(second["compose_file"]).read_text())
    service = override["services"]["lab"]
    assert service["ports"] == ["127.0.0.1:8894:8888"]
    mounts = {mount["target"]: mount for mount in service["volumes"]}
    assert mounts["/data"]["read_only"] is True
    assert mounts["/outputs"].get("read_only", False) is False
    assert mounts["/run/secrets/alaska2_lab_token"]["read_only"] is True


def test_missing_supplied_data_is_not_created(project):
    data = project / "missing-data"
    with pytest.raises(ValueError, match="data"):
        prepare(project, data_dir=data)
    assert not data.exists()


def test_existing_empty_token_is_repaired_automatically(project):
    path = project / ".local/jupyter_token"
    path.parent.mkdir()
    path.write_text("")
    prepare(project)
    assert len(path.read_text().strip()) >= 32


def test_jupyter_accepts_the_actual_server_configuration(tmp_path, monkeypatch):
    from jupyter_server.serverapp import ServerApp
    from traitlets.config.loader import PyFileConfigLoader

    token = tmp_path / "token"
    token.write_text("x" * 40)
    monkeypatch.setenv("ALASKA2_LAB_TOKEN_FILE", str(token))
    source_config = Path(__file__).resolve().parents[1] / "docker/jupyter_server_config.py"
    config_dir = tmp_path / "checkout/docker"
    config_dir.mkdir(parents=True)
    (config_dir / source_config.name).write_bytes(source_config.read_bytes())
    config = PyFileConfigLoader("jupyter_server_config.py", path=str(config_dir)).load_config()
    app = ServerApp(config=config)
    assert set(config.ServerApp) <= set(app.traits()), "unknown ServerApp configuration"
    assert app.root_dir == str(config_dir.parent)
    assert app.log_level == 30
    assert config.IdentityProvider.token == "x" * 40
    assert app.open_browser is False


def test_outputs_cannot_overlap_source_or_read_only_inputs(project):
    data = project.parent / "inputs"
    data.mkdir()
    for output in (project / "outputs", data / "outputs", project.parent):
        with pytest.raises(ValueError, match="overlap"):
            prepare(project, data_dir=data, output_dir=output)


def test_existing_short_token_is_repaired_automatically(project):
    token = project / ".local/jupyter_token"
    token.parent.mkdir()
    token.write_text("short")
    prepare(project)
    assert len(token.read_text().strip()) >= 32
