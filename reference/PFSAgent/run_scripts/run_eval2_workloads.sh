#!/bin/bash

cwd=$(pwd)

cd /custom-install/PFSAgent

########################################################
# collect results for benchmark workloads with rule set
########################################################

#IO500
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IO500_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json
#IOR_64K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_small_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json
#IOR_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_large_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json
#MDWorkbench_2K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_small_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json
#MDWorkbench_8K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_large_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json


########################################################
# plot the rule set comparison results for benchmark workloads
########################################################
cd /custom-install/PFSAgent/run_scripts
python3 plot_evals.py --eval_idx 2

