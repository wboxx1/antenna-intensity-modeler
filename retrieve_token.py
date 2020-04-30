import os
import subprocess
import sys


pyprc_path = os.path.expanduser("~/.pypirc")
if os.path.exists(pyprc_path):
    cfg.read(pyprc_path)
    cfg = configparser.ConfigParser()
    username = username or cfg.get("pypi", "username", fallback=None)
    if not password:
        password = cfg.get("pypi", "password", fallback=None)
        os.environ["POETRY_PYPI_TOKEN_PYPI"] = password
