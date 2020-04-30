import os
import subprocess
import sys
import configparser

pyprc_path = os.environ["PYPIRC_PATH"]

if os.path.exists(pyprc_path):
    print("Path exists.")
    cfg = configparser.ConfigParser()
    print(cfg.read(pyprc_path))
    print(cfg.sections())
    for sec in cfg.sections():
        for key in sec:
            print(key)
    username = cfg.get("pypi", "username", fallback=None)
    password = cfg.get("pypi", "password", fallback=None)
    token = cfg.get("pypi", "token", fallback=None)
    print(password)
    print(token)
    os.environ["POETRY_PYPI_TOKEN_PYPI"] = password
else:
    print("Couldn't find that shit.")
