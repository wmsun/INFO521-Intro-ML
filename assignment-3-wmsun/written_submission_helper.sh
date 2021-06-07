#!/bin/bash

# Script that checks if a file is pressent, if not, will fail a travis build
FILE=$1
if [ -f "$FILE" ]; then
    echo "$FILE was found"
else
    echo "$FILE is not versioned"
    exit 1
fi
