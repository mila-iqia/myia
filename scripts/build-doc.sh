#!/bin/sh
set -ex

cd docs
sphinx-apidoc -fo source/ ../myia/
sphinx-build -W -b html . _build
