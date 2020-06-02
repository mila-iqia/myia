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
conda init
. $HOME/miniconda/etc/profile.d/conda.sh
pip install poetry2conda>=0.3.0
poetry2conda pyproject.toml --dev -E pytorch -E $DEV -E relay env.yml
cat $DEV-extras.conda relay-extras.conda >> env.yml
conda env create -n test -f env.yml
