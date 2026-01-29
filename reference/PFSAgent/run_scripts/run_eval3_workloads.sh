#!/bin/bash

cwd=$(pwd)

cd /custom-install/PFSAgent


########################################################
# collect baseline results for real-app workloads
########################################################

#amrex
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/Amrex_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#macsio_512K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/macsio_512k_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json
#macsio_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/macsio_16m_config.json --run_type no_change --iterations 2 --eval_result_type aggregate --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json


########################################################
# collect results for real-app workloads without rule set
########################################################

#amrex
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/Amrex_config.json --run_type default --iterations 5 --eval_result_type speedup --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json
#macsio_512K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/macsio_512k_config.json --run_type default --iterations 5 --eval_result_type speedup --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json
#macsio_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/macsio_16m_config.json --run_type default --iterations 5 --eval_result_type speedup --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json


########################################################
# collect results for real-app workloads with rule set
########################################################

#amrex
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/Amrex_config.json --run_type default --iterations 5 --eval_result_type speedup --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json
#macsio_512K
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/macsio_512k_config.json --run_type default --iterations 5 --eval_result_type speedup --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json
#macsio_16M
python3 run.py --stellar-config /custom-install/PFSAgent/AgentConfigs/macsio_16m_config.json --run_type default --iterations 5 --eval_result_type speedup --aggregate_result_path /custom-install/eval_results/baseline_tuning_performance.json --speedup_result_path /custom-install/eval_results/tuning_speedups.json --use_rule_set --rule_set_path /custom-install/PFSAgent/rule_sets/combined_rule_set.json


########################################################
# plot the rule set comparison results for real-app workloads
########################################################
cd /custom-install/PFSAgent/run_scripts
python3 plot_evals.py --eval_idx 3