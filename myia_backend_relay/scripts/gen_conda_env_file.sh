#!/bin/bash

# NB: This script should be run from myia_backend_relay folder containing pyproject.toml and scripts/ sub-folder.

# Install python dependencies required to run this script.
pip install poetry2conda==0.3.0
pip install pyyaml

# Generate environment.yml from pyproject using poetry2conda
poetry2conda pyproject.toml environment.yml

# Update environment.yml to make it work with conda and add additional requirements
# (especially: tvm package from conda channel mila-iqia).
python scripts/update_env.py environment.yml -p relay-extras.conda -o environment.yml
