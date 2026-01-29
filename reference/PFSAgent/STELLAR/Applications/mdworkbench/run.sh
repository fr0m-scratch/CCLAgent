#! /bin/bash

benchmark_dir=$1
data_dir=$2
results_path=$3
shift 3  # Remove first three arguments
run_params="$@"  # Capture all remaining arguments

echo "benchmark_dir: $benchmark_dir"
echo "data_dir: $data_dir"
echo "results_path: $results_path"
echo "run_params: $run_params"

if [ -z "$run_params" ]; then
    echo "Usage: $0 <benchmark_dir> <results_dir> <run_params>"
    exit 1
fi

cd $benchmark_dir
echo "pwd: $(pwd)"

echo "running"
echo "MPIARGS: $MDWORKBENCH_MPIARGS"

mpirun $MDWORKBENCH_MPIARGS ./md-workbench -2 -3 $run_params -o $data_dir > $results_path


