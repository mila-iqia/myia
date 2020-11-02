#!/bin/bash

# NB: This script should be run from package folder containing pyproject.toml
# NB: package myia_utils must be installed.

# Install python dependencies required to run this script.
pip install poetry2conda==0.3.0

# Generate environment.yml from pyproject using poetry2conda
poetry2conda pyproject.toml environment.yml

# Remove name field from environment file.
python -m myia_utils.update_env environment.yml -o environment.yml
