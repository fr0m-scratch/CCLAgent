from STELLAR.Applications import Application
import os
import json
import random
from STELLAR.Utils.logger import LOGGERS


logger = LOGGERS["application"]


class IOR(Application):
    name = "IOR"
    description = "IOR is a parallel I/O benchmark that can be used to test the performance of parallel file systems using various interfaces and access patterns."
    runtime_description = None
    score_metric = ["read_mean_bw", "write_mean_bw", "total_bw"]
    score_metric_descriptions = {
        "read_max_bw": "Maximum read bandwidth in MiB/s",
        "read_min_bw": "Minimum read bandwidth in MiB/s",
        "read_mean_bw": "Mean read bandwidth in MiB/s",
        "read_mean_time": "Mean time for read operations in seconds",
        "write_max_bw": "Maximum write bandwidth in MiB/s",
        "write_min_bw": "Minimum write bandwidth in MiB/s",
        "write_mean_bw": "Mean write bandwidth in MiB/s",
        "write_mean_time": "Mean time for write operations in seconds",
        "write_max_iops": "Maximum write IOPS",
        "write_min_iops": "Minimum write IOPS",
        "write_mean_iops": "Mean write IOPS",
        "write_stddev_iops": "Standard deviation of write IOPS",
        "total_bw": "Combined read and write mean bandwidth in MiB/s"
    }
    has_pre_execution_steps = True

    def __init__(self, config_name: str = None):
        super().__init__(config_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.results_path = os.path.join(self.results_dir, "results.txt")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_file = os.path.join(self.data_dir, "ior_data_file")


    def build(self):
        pass


    def parse_config_content_to_string(self, config_content: dict) -> str:
        args = []
        for key, value in config_content.items():
            if value is True:
                args.append(f"-{key}")
            else:
                args.append(f"-{key} {value}")
        return " ".join(args)
    

    def parse_config(self):
        selected_config = f"{self.config_name}.json"
        configs_folder = os.path.join(os.path.dirname(__file__), "configs")
        config_files = os.listdir(configs_folder)
        if selected_config not in config_files:
            raise ValueError(f"Config file {selected_config} not found in {configs_folder}")
        config_path = os.path.join(configs_folder, selected_config)
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Convert config to command line arguments
        config_content = self.parse_config_content_to_string(config)
        
        return {
            "config_path": config_path,
            "config_content": config_content
        }
    
    def get_pre_execution_steps(self, application_config_dict: dict) -> list[dict]:
        return [
            {"command": [
                os.path.join(os.path.dirname(__file__), "pre_execution.sh"), 
                self.data_dir
            ]}
        ]
    

    def get_run_command(self, application_config_dict: dict) -> list:
        return [
            os.path.join(os.path.dirname(__file__), "run.sh"),
            self.application_root_dir,
            self.data_file,
            self.results_path,
            application_config_dict["config_content"],
        ]
    

    def parse_results(self):
        with open(self.results_path, "r") as f:
            results_text = f.read()

        # Parse the summary format
        # Example:
        # Summary of all tests:
        # Operation   Max(MiB)   Min(MiB)  Mean(MiB)     StdDev   Max(OPs)   Min(OPs)  Mean(OPs)     StdDev    Mean(s) Stonewall(s) Stonewall(MiB) Test# #Tasks tPN reps fPP reord reordoff reordrand seed segcnt   blksiz    xsize aggs(MiB)   API RefNum
        # write         865.71     865.71     865.71       0.00      13.53      13.53      13.53       0.00  141.94207         NA            NA     0     60  10    1   1     1        1         0    0      1 2147483648 67108864  122880.0 MPIIO      0
        # read          942.99     942.99     942.99       0.00      14.73      14.73      14.73       0.00  130.30905         NA            NA     0     60  10    1   1     1        1         0    0      1 2147483648 67108864  122880.0 MPIIO      0

        # Parse the summary format using column positions from the header
        lines = results_text.strip().split('\n')
        if len(lines) < 3:  # Need at least header + write + read lines
            logger.warning("Summary section doesn't contain enough lines")
            return {
                "write": {"max_bw": 0.0, "min_bw": 0.0, "mean_bw": 0.0, "stddev_bw": 0.0, "max_iops": 0.0, "min_iops": 0.0, "mean_iops": 0.0, "stddev_iops": 0.0, "mean_time": 0.0},
                "read": {"max_bw": 0.0, "min_bw": 0.0, "mean_bw": 0.0, "stddev_bw": 0.0, "max_iops": 0.0, "min_iops": 0.0, "mean_iops": 0.0, "stddev_iops": 0.0, "mean_time": 0.0}
            }
        
        # Find lines containing write and read operations
        write_line = None
        read_line = None
        for line in lines[1:]:  # Skip header
            if line.strip().startswith('write'):
                write_line = line
            elif line.strip().startswith('read'):
                read_line = line
        
        results = {
            "write": {},
            "read": {}
        }
        
        # Parse write line
        if write_line:
            values = write_line.split()
            if len(values) >= 10:  # Ensure we have enough values
                results["write"]["max_bw"] = float(values[1])
                results["write"]["min_bw"] = float(values[2])
                results["write"]["mean_bw"] = float(values[3])
                results["write"]["stddev_bw"] = float(values[4])
                results["write"]["max_iops"] = float(values[5])
                results["write"]["min_iops"] = float(values[6])
                results["write"]["mean_iops"] = float(values[7])
                results["write"]["stddev_iops"] = float(values[8])
                results["write"]["mean_time"] = float(values[9])
            else:
                logger.warning("Write line doesn't contain enough values")
                # Set default values as in the original code
                results["write"]["max_bw"] = 0.0
                results["write"]["min_bw"] = 0.0
                results["write"]["mean_bw"] = 0.0
                results["write"]["stddev_bw"] = 0.0
                results["write"]["max_iops"] = 0.0
                results["write"]["min_iops"] = 0.0
                results["write"]["mean_iops"] = 0.0
                results["write"]["stddev_iops"] = 0.0
                results["write"]["mean_time"] = 0.0
        
        # Parse read line
        if read_line:
            values = read_line.split()
            if len(values) >= 10:  # Ensure we have enough values
                results["read"]["max_bw"] = float(values[1])
                results["read"]["min_bw"] = float(values[2])
                results["read"]["mean_bw"] = float(values[3])
                results["read"]["stddev_bw"] = float(values[4])
                results["read"]["max_iops"] = float(values[5])
                results["read"]["min_iops"] = float(values[6])
                results["read"]["mean_iops"] = float(values[7])
                results["read"]["stddev_iops"] = float(values[8])
                results["read"]["mean_time"] = float(values[9])
            else:
                logger.warning("Read line doesn't contain enough values")
                # Set default values as in the original code
                results["read"]["max_bw"] = 0.0
                results["read"]["min_bw"] = 0.0
                results["read"]["mean_bw"] = 0.0
                results["read"]["stddev_bw"] = 0.0
                results["read"]["max_iops"] = 0.0
                results["read"]["min_iops"] = 0.0
                results["read"]["mean_iops"] = 0.0
                results["read"]["stddev_iops"] = 0.0
                results["read"]["mean_time"] = 0.0
        
        logger.info(f"Parsed IOR summary results: {json.dumps(results, indent=2)}")
        return results

    def get_score(self, results):
        score_metric = self.score_metric

        # Map metric names to their values in the results dictionary
        metric_mapping = {
            "read_max_bw": results["read"]["max_bw"],
            "read_min_bw": results["read"]["min_bw"],
            "read_mean_bw": results["read"]["mean_bw"],
            "read_mean_time": results["read"]["mean_time"],
            "write_max_bw": results["write"]["max_bw"],
            "write_min_bw": results["write"]["min_bw"],
            "write_mean_bw": results["write"]["mean_bw"],
            "write_mean_time": results["write"]["mean_time"],
            "write_max_iops": results["write"]["max_iops"],
            "write_min_iops": results["write"]["min_iops"],
            "write_mean_iops": results["write"]["mean_iops"],
            "write_stddev_iops": results["write"]["stddev_iops"],
            "read_max_iops": results["read"]["max_iops"],
            "read_min_iops": results["read"]["min_iops"],
            "total_bw": (results["read"]["mean_bw"] + results["write"]["mean_bw"]) / 2
        }

        if isinstance(score_metric, str):
            if score_metric in self.score_metric_descriptions:
                score_value = metric_mapping.get(score_metric)
                if score_value is None:
                    raise ValueError(f"Invalid score metric: {score_metric}")
                score_description = self.score_metric_descriptions[score_metric]
            else:
                raise ValueError(f"Invalid score metric: {score_metric}")
                
            return {
                "value": score_value,
                "description": score_description
            }
        
        elif isinstance(score_metric, list):
            score_value = {}
            score_description = {}
            
            for metric in score_metric:
                if metric in self.score_metric_descriptions:
                    score_value[metric] = metric_mapping.get(metric)
                    if score_value[metric] is None:
                        raise ValueError(f"Invalid score metric: {metric}")
                    score_description[metric] = self.score_metric_descriptions[metric]
                else:
                    raise ValueError(f"Invalid score metric: {metric}")
            
            return {
                "value": score_value,
                "description": score_description
            }
        
        else:
            raise ValueError(f"Invalid score metric type: {type(score_metric)}")




