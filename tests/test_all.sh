#!/bin/sh

mkdir -p test_log
for f in ./inputs/* ; do
    basename=$(basename $f)
    echo "Testing $basename"
    touch test_log/$basename.log
    python testbot.py $f &> test_log/$basename.log
    echo $?
    sh clean.sh
done
