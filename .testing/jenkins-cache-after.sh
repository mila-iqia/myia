#!/bin/bash

BASE=/var/cache/jenkins
PROJECT=myia


if [ ! -d $BASE ]; then
    echo "Base cache dir doesn't exist"
    exit 0;
fi

NAME=${JOB_NAME//[\/]/_}

echo "Archiving artifacts" "$@" "to $NAME"
tar -czf $BASE/$NAME.tgz -C workspace "$@"

if [ x"$GIT_BRANCH" == x"master" ]; then
    echo "Master branch; updating master cache"
    cp $BASE/$NAME.tgz $BASE/$PROJECT-master.tgz
