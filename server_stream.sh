#!/bin/bash
nc -l 8000 | vlc --demux h264 -
