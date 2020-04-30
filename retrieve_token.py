import os
import subprocess
import sys
import configparser

pyprc_path = os.environ["PYPIRC_PATH"]

if os.path.exists(pyprc_path):
    print("Path exists.")
    cfg = configparser.ConfigParser()
    cfg.read(pyprc_path)
    username = cfg.get("pypitest", "username", fallback=None)
    password = cfg.get("pypitest", "password", fallback=None)
    token = cfg.get("pypitest", "token", fallback=None)
    print(password)
    print(token)
    os.environ["POETRY_PYPI_TOKEN_PYPI"] = password
else:
    print("Couldn't find that shit.")
