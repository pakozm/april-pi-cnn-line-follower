#!/bin/bash
dir=$HOME/april-pi-cnn-line-folower
pid=$($dir/getpid.sh)
if [ -z $pid ]; then
    echo "ERROR: NOT FOUND"
else
    kill -9 $pid && echo "Ok"
fi
