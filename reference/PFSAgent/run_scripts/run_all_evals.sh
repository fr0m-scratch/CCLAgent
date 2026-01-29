#!/bin/bash


time_file="./eval_times.txt"

# create eval_results directory and plots subdirectory
mkdir -p /custom-install/eval_results
mkdir -p /custom-install/eval_results/plots

########################################################
# run all eval scripts
########################################################

# run eval 1
start_time=$(date +%s)
./run_eval1_workloads.sh
end_time=$(date +%s)
echo "Eval 1 took $((end_time - start_time)) seconds" >> $time_file

# run eval 2
start_time=$(date +%s)
./run_eval2_workloads.sh
end_time=$(date +%s)
echo "Eval 2 took $((end_time - start_time)) seconds" >> $time_file

# run eval 3
start_time=$(date +%s)
./run_eval3_workloads.sh
end_time=$(date +%s)
echo "Eval 3 took $((end_time - start_time)) seconds" >> $time_file