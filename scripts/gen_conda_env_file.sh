#!/bin/bash

# NB: This script should be run from package folder containing pyproject.toml

# Install python dependencies required to run this script.
pip install poetry2conda==0.3.0

# Generate environment.yml from pyproject using poetry2conda
poetry2conda pyproject.toml --dev -E pytorch environment.yml
