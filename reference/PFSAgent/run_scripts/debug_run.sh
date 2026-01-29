#!/bin/bash

cwd=$(pwd)

cd /custom-install/PFSAgent

echo "Starting debug run with no agents"
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/debug.json --run_type no_change --iterations 1 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
echo "Debug run complete"

echo "Starting debug run with agents"
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/debug.json --run_type default --iterations 1 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --rerun_with_best_iterations 1
echo "Debug run complete"

cd $cwd
echo "Debug runs complete"