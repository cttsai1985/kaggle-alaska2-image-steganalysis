"""Require a private nonempty token; Compose publishes this service on loopback only."""

import os
from pathlib import Path

c = get_config()  # noqa: F821 - supplied by Jupyter's configuration loader
secret_file = Path(os.environ["ALASKA2_LAB_TOKEN_FILE"])
secret = secret_file.read_text().strip()
if len(secret) < 32:
    raise ValueError("prepare a nonempty private Lab token before startup")
c.IdentityProvider.token = secret
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/workspace"
c.ServerApp.log_level = "WARN"
c.ServerApp.show_banner = False
