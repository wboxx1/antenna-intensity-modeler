import os
import subprocess
import sys
import configparser

pyprc_path = os.environ["PYPIRC_PATH"]

if os.path.exists(pyprc_path):
    print("Path exists.")
    cfg = configparser.ConfigParser()
    cfg.read(pyprc_path)
    username = cfg.get("pypi", "username", fallback=None)
    password = cfg.get("pypi", "password", fallback=None)
    os.environ["POETRY_PYPI_TOKEN_PYPI"] = password
else:
    print("Couldn't find that shit.")
