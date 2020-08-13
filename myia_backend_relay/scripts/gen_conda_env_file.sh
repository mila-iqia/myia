#!/bin/bash

# NB: This script should be run from package folder containing pyproject.toml and scripts/ sub-folder.
# NB: package myia_utils must be installed.

# Install python dependencies required to run this script.
pip install poetry2conda==0.3.0

# Generate environment.yml from pyproject using poetry2conda
poetry2conda pyproject.toml environment.yml

# Update environment.yml to make it work with conda and add additional requirements
# (especially: tvm package from conda channel mila-iqia).
python -m myia_utils.update_env environment.yml -p relay-extras.conda -o environment.yml
