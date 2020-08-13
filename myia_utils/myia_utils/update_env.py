"""
Utility script to update a conda environment file.
- allow to extend environment file with a requirements.txt-like file.
- simplify file to make it work correctly with command:
  `conda env update --file <env-file>`
"""
import argparse

from yaml import dump, load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="YAML file path")
    parser.add_argument(
        "-p",
        "--packages",
        help="File containing packages requirements to add to YAML file.",
    )
    parser.add_argument(
        "-o", "--output", help="Output YAML file. By default, print output."
    )
    args = parser.parse_args()
    with open(args.file) as file:
        data = load(file, Loader=Loader)
    if args.packages:
        packages = []
        with open(args.packages) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                packages.append(line)
        dependencies = data.get("dependencies", [])
        packages_added = False
        for index, element in enumerate(dependencies):
            if isinstance(element, dict):
                dependencies = (
                    dependencies[:index] + packages + dependencies[index:]
                )
                packages_added = True
                break
        if not packages_added:
            dependencies.extend(packages)
        data["dependencies"] = dependencies
    data.pop("name", None)
    output = dump(data, Dumper=Dumper, sort_keys=False)
    if args.output:
        with open(args.output, "w") as file:
            file.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
