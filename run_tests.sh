#!/bin/sh

# Abort on first error
set -e

# Run myia tests
python3 tests.py

# Run PEP8 linter
pep8 --ignore=E265 --show-source --show-pep8 myia/*.py *.py
