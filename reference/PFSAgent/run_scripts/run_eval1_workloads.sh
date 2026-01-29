#!/bin/bash

cwd=$(pwd)

cd /custom-install/PFSAgent

########################################################
# Collect baseline results using default config
# workload will be run (iterations+1) times
########################################################

#IO500
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IO500_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#IOR_64K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_small_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#IOR_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_large_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#MDWorkbench_2K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_small_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#MDWorkbench_8K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_large_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json


########################################################
# Collect results using expert config
# workload will be run (iterations+1) times
########################################################

#IO500
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IO500_config.json --starting-params /custom-install/PFSAgent/starter_configs/IO500_expert_params.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#IOR_64K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_small_config.json --starting-params /custom-install/PFSAgent/starter_configs/IOR_small_expert_params.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#IOR_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_large_config.json --starting-params /custom-install/PFSAgent/starter_configs/IOR_large_expert_params.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#MDWorkbench_2K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_small_config.json --starting-params /custom-install/PFSAgent/starter_configs/MDWorkbench_small_expert_params.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#MDWorkbench_8K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_large_config.json --starting-params /custom-install/PFSAgent/starter_configs/MDWorkbench_large_expert_params.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json


########################################################
# Collect results using STELLAR tuning
# best generate config will be run (iterations+1) times
########################################################

#IO500
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IO500_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --rerun_with_best_iterations 3 --speedup_result_path /custom-install/eval_results/tuning_speedups.json
#IOR_64K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_small_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --rerun_with_best_iterations 3 --speedup_result_path /custom-install/eval_results/tuning_speedups.json
#IOR_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/IOR_large_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --rerun_with_best_iterations 3 --speedup_result_path /custom-install/eval_results/tuning_speedups.json
#MDWorkbench_2K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_small_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --rerun_with_best_iterations 3 --speedup_result_path /custom-install/eval_results/tuning_speedups.json
#MDWorkbench_8K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/MDWorkbench_large_config.json --run_type default --iterations 5 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --rerun_with_best_iterations 3 --speedup_result_path /custom-install/eval_results/tuning_speedups.json



########################################################
# plot the collected results
########################################################
cd /custom-install/PFSAgent/run_scripts
python3 plot_evals.py --eval_idx 1









