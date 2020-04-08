
import re
import sys
import toml

if __name__ == "__main__":
    key = sys.argv[1]
    data = toml.load("pyproject.toml")
    for part in key.split("."):
        data = data[part]

    for name, version in data.items():
        if re.match("^[0-9]", version):
            version = f"=={version}"
        reqstr = f"{name}{version}"
        print(reqstr)
