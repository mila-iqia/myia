#!/bin/bash

# NB: This script should be run from myia_backend_relay folder containing pyproject.toml and scripts/ sub-folder.

# Install python dependencies required to run this script.
pip install poetry2conda==0.3.0
pip install pyyaml

# Generate environment.yml from pyproject using poetry2conda
poetry2conda pyproject.toml --dev -E pytorch -E cpu environment-cpu.yml
poetry2conda pyproject.toml --dev -E pytorch -E gpu environment-gpu.yml

# Generate conda environment files for CPU and GPU.
python scripts/update_env.py environment-cpu.yml -p cpu-extras.conda -o environment-cpu.yml
python scripts/update_env.py environment-gpu.yml -p gpu-extras.conda -o environment-gpu.yml
