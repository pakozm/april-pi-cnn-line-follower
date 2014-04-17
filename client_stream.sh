#!/bin/bash
server=$1
if [[ -z $server ]]; then
    server=192.168.1.20
fi
w=$2
h=$3
if [[ -z $w ]]; then
    w=32
fi
if [[ -z $h ]]; then
    h=32
fi
raspivid -w $w -h $h -t 60000 -o - | nc $server 8000
