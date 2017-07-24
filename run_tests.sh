#!/bin/sh

# Abort on first error
set -e

# Run PEP8 linter
# Ignore E701 for now because it triggers on some type annotations in Py3.6
pep8 --ignore=E265,E251,E701 --show-source --show-pep8 myia/*.py tests/*.py

# Run myia tests
export PYTHONPATH=$PYTHONPATH:`pwd`
pytest
