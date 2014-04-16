#!/bin/bash
ps aux | grep "python.*capture.py$" | grep -v "SCREEN" | awk '{ print $2 }'
