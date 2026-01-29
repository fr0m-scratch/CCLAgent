from STELLAR.Utils.stellar_config import StellarConfig
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stellar-config", type=str)
    parser.add_argument("--starting-params", type=str, default="./starter_configs/default_lustre_v3_params.json")
    parser.add_argument("--run_type", type=str, default="default", choices=["no_change", "default"])
    parser.add_argument("--use_rule_set", action="store_true")
    parser.add_argument("--rule_set_path", type=str, default="./rule_sets/combined_rule_set.json")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--eval_result_type", type=str, required=True, choices=["aggregate", "speedup"])
    parser.add_argument("--aggregate_result_path", type=str, default="./eval_results/baseline_results.json")
    parser.add_argument("--speedup_result_path", type=str)
    parser.add_argument("--rerun_with_best_iterations", type=int, default=0)
    args = parser.parse_args()

    # Set the config path before any possible instantiation
    if args.stellar_config:
        StellarConfig.set_default_config_path(args.stellar_config)
    
    # Now get the config instance
    run_config = StellarConfig.get_instance()
    from STELLAR.main import StellarAgent
    # Load the starting params
    with open(args.starting_params, "r") as f:
        starting_params = json.load(f)

    enable_analysis = args.run_type != "no_change"

    eval_result_source = None
    if "expert" in args.starting_params:
        eval_result_source = "expert"
    elif args.run_type == "no_change":
        eval_result_source = "default"
    else:
        eval_result_source = "STELLAR"

    use_rule_set = False
    if args.use_rule_set:
        use_rule_set = True
    else:
        use_rule_set = False

    agent = StellarAgent(run_config, 
                            starting_params=starting_params, 
                            enable_runtime_analysis=enable_analysis, 
                            use_rule_set=use_rule_set, 
                            eval_result_type=args.eval_result_type, 
                            aggregate_results_file=args.aggregate_result_path,
                            speedup_results_file=args.speedup_result_path,
                            eval_result_source=eval_result_source,
                            rule_set_path=args.rule_set_path)

    if args.run_type == "no_change":
        agent.run_no_change(max_steps=args.iterations)
    elif args.run_type == "default":
        agent.run(max_steps=args.iterations, test_config_iterations=args.rerun_with_best_iterations)
    else:
        raise ValueError(f"Invalid run type: {args.run_type}")






    
    
    

