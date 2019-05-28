#!/bin/bash

BASE=/var/cache/jenkins
PROJECT=myia


if [ ! -d $BASE ]; then
    echo "Base cache dir doesn't exist"
    exit 0;
fi

NAME=${JOB_NAME//[\/]/_}

if [ -f $BASE/$NAME.tgz ]; then
    echo "Expanding cache for $NAME"
    tar -xzf $BASE/$NAME.tgz -C $WORKSPACE
elif [ -f $BASE/$PROJECT-master.tgz ]; then
    echo "Expanding cache from master"
    tar -xzf $BASE/$PROJECT-master.tgz -C $WORKSPACE
else
    echo "No caches found"
fi
