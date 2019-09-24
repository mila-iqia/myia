#!/bin/sh
set -eux

NAME=$1
shift;

"$@"

if ls .coverage.* 1>/dev/null 2>&1; then
    coverage combine -a .coverage.*
fi

mv .coverage $NAME.coverage
