#!/bin/bash
data_dir=$1

if [ ! -d "$data_dir" ]; then
    mkdir -p "$data_dir"
fi

