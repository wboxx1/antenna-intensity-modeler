import os
import subprocess
import sys


pyprc_path = os.path.expanduser("~/.pypirc")
print("first: {}".format(pyprc_path))
pyprc_path = os.environ["PYPIRC_PATH"]
print("second: {}".format(pyprc_path))
if os.path.exists(pyprc_path):
    print("Path exists.")
    cfg = configparser.ConfigParser()
    cfg.read(pyprc_path)
    username = username or cfg.get("pypi", "username", fallback=None)
    if not password:
        password = cfg.get("pypi", "password", fallback=None)
        os.environ["POETRY_PYPI_TOKEN_PYPI"] = password
else:
    print("Couldn't find that shit.")
