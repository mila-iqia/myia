#!/bin/bash

set -x
set -e

DEV=cpu

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu)
    DEV=gpu
    shift
    ;;
esac
done

if [ ! -d $HOME/miniconda ]; then
    wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-`uname -s`-`uname -m`.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
fi
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda install pip
conda init
. $HOME/miniconda/etc/profile.d/conda.sh
conda env remove -n test
conda create -n test python=3.7

# Activate conda environment and install poetry.
conda activate test
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
source $HOME/.poetry/env

# Install myia_utils. It's used later to re-generate conda environment files.
cd myia_utils
poetry install
cd ..

# Install myia, backend and frontend plugins using poetry.
poetry install

cd myia_backend_pytorch
poetry install

cd ../myia_backend_relay
poetry install

cd ../myia_frontend_pytorch
poetry install

cd ..

# Complete installation with specific conda packages using environment files.
# Re-generate environment files before using them.
./scripts/gen_conda_env_file.sh
conda env update --file environment.yml

cd myia_backend_pytorch
./scripts/gen_conda_env_file.sh
conda env update --file environment-${DEV}.yml

cd ../myia_backend_relay
./scripts/gen_conda_env_file.sh
conda env update --file environment.yml

cd ../myia_frontend_pytorch
./scripts/gen_conda_env_file.sh
conda env update --file environment-${DEV}.yml

cd ..
