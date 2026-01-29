#! /bin/bash

benchmark_dir=$1
data_dir=$2
shift 2  # Remove first two arguments
run_params="$@"  # Capture all remaining arguments

echo "benchmark_dir: $benchmark_dir"
echo "data_dir: $data_dir"
echo "run_params: $run_params"

if [ -z "$run_params" ]; then
    echo "Usage: $0 <benchmark_dir> <data_dir> <run_params>"
    exit 1
fi

cd $benchmark_dir
echo "pwd: $(pwd)"

# precreate env
echo "precreating env"
echo "RUN_PARAMS: $run_params"
mpirun $MDWORKBENCH_MPIARGS ./md-workbench -1 $run_params -o $data_dir


