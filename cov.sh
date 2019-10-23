#!/bin/sh
set -eux

echo $PATH
which python

if ls .coverage.* 1>/dev/null 2>&1; then
    coverage combine -a .coverage.*
fi

coverage report -m
