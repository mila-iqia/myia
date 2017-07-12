#!/bin/sh

# Abort on first error
set -e

# Run PEP8 linter
pep8 --ignore=E265,E251 --show-source --show-pep8 myia/*.py tests/*.py

# Run myia tests
export PYTHONPATH=$PYTHONPATH:`pwd`
pytest
