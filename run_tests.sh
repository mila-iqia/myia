#!/bin/sh

# Abort on first error
set -e

# Run PEP8 linter
# Ignore E402 because circular imports sometimes requires imports at the bottom
#     of the file
# Ignore E701 for now because it triggers on some type annotations in Py3.6
pep8 --ignore=E265,E251,E402,E701 --show-source myia tests

# Run mypy type checker
mypy --ignore-missing-imports myia

# Run myia tests
export PYTHONPATH=$PYTHONPATH:`pwd`
pytest
