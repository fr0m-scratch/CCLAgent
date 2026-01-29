#!/bin/bash


ior_path=$1
data_file=$2
results_path=$3
shift 3  # Remove first three arguments
run_params="$@"  # Capture all remaining arguments

echo "mpiargs: $IOR_MPIARGS"
echo "ior_path: $ior_path"
echo "data_file: $data_file"
echo "results_path: $results_path"
echo "run_params: $run_params"


if [ -z "$run_params" ]; then
    echo "Usage: $0 <ior_path> <data_dir> <results_path> <run_params>"
    exit 1
fi


cd $ior_path

echo "pwd: $(pwd)"

mpirun $IOR_MPIARGS ./ior $run_params -o $data_file > $results_path





