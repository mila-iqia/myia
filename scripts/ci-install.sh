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
pip install poetry2conda>=0.3.0
poetry2conda pyproject.toml --dev -E pytorch -E $DEV -E relay env.yml
cat $DEV-extras.conda relay-extras.conda >> env.yml
# conda doesn't like overwriting envs anymore, but doesn't complain
# if you remove a non-existent env.
conda env remove -n test
conda env create -n test -f env.yml

# Install backend plugins.
conda activate test
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
source $HOME/.poetry/env
cd myia_backend_pytorch
poetry install
cd ../myia_backend_relay
poetry install
cd ..
