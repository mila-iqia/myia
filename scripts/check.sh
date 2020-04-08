#!/bin/sh
set -ex

flake8
pydocstyle myia
isort -c -df
black --check .
