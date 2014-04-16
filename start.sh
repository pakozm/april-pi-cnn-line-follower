#!/bin/bash
dir=$HOME/april-pi-cnn-line-follower
pid=$($dir/getpid.sh)
if [ -z $pid ]; then
    screen -d -m python $dir/capture.py && echo "Ok"
else
    echo "ERROR: ALREADY RUNNING!!"
fi
