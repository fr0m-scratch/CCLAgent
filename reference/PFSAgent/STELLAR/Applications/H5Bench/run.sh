#! /bin/bash

h5_config_file=$1
h5_data_dir=$2
h5_results_dir=$3

# delete the contents of the results directory(if it exists)
if [ -d "$h5_results_dir" ]; then
    rm -rf $h5_results_dir/*
fi

# if data directory does not exist, create it
if [ ! -d "$data_dir" ]; then
    mkdir -p $h5_data_dir
fi

cd $h5_data_dir

# Get start time in nanoseconds
start_time=$(date +%s.%N)
h5bench --debug $h5_config_file
# Get end time in nanoseconds
end_time=$(date +%s.%N)

# Calculate walltime in seconds with high precision
# Using bc for floating point arithmetic
walltime=$(echo "scale=6; ($end_time - $start_time)" | bc)

echo "walltime: $walltime" > $h5_results_dir/total_walltime.txt

