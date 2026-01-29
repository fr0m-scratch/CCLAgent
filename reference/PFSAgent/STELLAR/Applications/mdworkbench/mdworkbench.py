from STELLAR.Applications import Application
from STELLAR.FS import FSConfig
import os
import subprocess
import re
import random
from STELLAR.Utils.logger import LOGGERS
import json


logger = LOGGERS["application"]


class MDWorkbench(Application):
    name = "MDWorkbench"
    description = "The MDWorkbench benchmark is an MPI-parallel benchmark to measure metadata (together with small object) performance. It aims to simulate actual user activities on a file system such as compilation."
    runtime_description = None
    score_metric = ["iops_rate", "tp"]
    score_metric_descriptions = {
        "max": "The maximum runtime for any process",
        "min": "The minimum runtime for any process",
        "mean": "The arithmetic mean runtime across all processes",
        "balance": "The minimum runtime divided by the maximum runtime, a balance of 50% means that the minimum process took 50% of the runtime of the longest running process. A high value is favorable. ",
        "stddev": "The standard deviation of runtime. A low value is favorable.",
        "iops_rate": "Given in iops/s, each operation like stat, create, read, delete is counted as one IOP, this number is computed based on the global timer",
        "objects": "Total number of objects written/accessed",
        "obj_rate": "Given in obj/s, this number is computed based on the global timer",
        "tp": "The throughput given the file size, an indicative value",
        "op-max": "The maximum runtime of any I/O operation, i.e., the slowest I/O operation."
    }
    has_pre_execution_steps = True
    
    def __init__(self, config_name: str = None):
        super().__init__(config_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.results_path = os.path.join(self.results_dir, "results.txt")


    def build(self):
        pass

    def get_pre_execution_steps(self, application_config_dict: dict) -> list[dict]:
        return [
            {"command": [
                os.path.join(os.path.dirname(__file__), "pre_execution.sh"), 
                self.application_root_dir, 
                self.data_dir, 
                application_config_dict["config_content"]
            ],
            "method": self.clear_all_caches}
        ]

    def get_run_command(self, application_config_dict: dict) -> list:
        return [
            os.path.join(os.path.dirname(__file__), "run.sh"),
            self.application_root_dir,
            self.data_dir,
            self.results_path,
            application_config_dict["config_content"]
        ]
    

    def parse_config_content_to_string(self, config_content: dict) -> str:
        return " ".join([f"-{key} {value}" for key, value in config_content.items()])

    def parse_config(self):
        selected_config = f"{self.config_name}.json"
        configs_folder = os.path.join(os.path.dirname(__file__), "configs")
        config_files = os.listdir(configs_folder)
        if selected_config not in config_files:
            raise ValueError(f"Config file {selected_config} not found in {configs_folder}")
        config_path = os.path.join(configs_folder, selected_config)
        with open(config_path, "r") as f:
            config = json.load(f)
        config_content = self.parse_config_content_to_string(config)
        return {
            "config_path": config_path,
            "config_content": config_content
        }


    def parse_results(self):
        with open(self.results_path, "r") as f:
            results = f.read()
        
        parsed_results = {
            'processes': {},
            'total_runtime': None
        }
        
        # Regular expressions for matching
        process_pattern = r'(?:benchmark|cleanup) process max:(\d+\.\d+)s min:(\d+\.\d+)s mean: (\d+\.\d+)s balance:(\d+\.\d+) stddev:(\d+\.\d+) rate:(\d+\.\d+) iops/s objects:(\d+) rate:(\d+\.\d+) obj/s tp:(\d+\.\d+) MiB/s op-max:(\d+\.\d+e[-+]?\d+)s'
        runtime_pattern = r'Total runtime: (\d+)s'
        
        # Extract process information
        count = 0
        for match in re.finditer(process_pattern, results):
            count += 1
            logger.info(f"Match: {match.group(0)}")
            if len(parsed_results['processes']) == 0:
                process_info = {
                    'max': float(match.group(1)),
                    'min': float(match.group(2)),
                    'mean': float(match.group(3)),
                    'balance': float(match.group(4)),
                    'stddev': float(match.group(5)),
                    'iops_rate': float(match.group(6)),
                    'objects': int(match.group(7)),
                    'obj_rate': float(match.group(8)),
                    'tp': float(match.group(9)),
                    'op_max': float(match.group(10))
                }
                parsed_results['processes'] = process_info
            else:
                parsed_results['processes']['max'] += float(match.group(1))
                parsed_results['processes']['min'] += float(match.group(2))
                parsed_results['processes']['mean'] += float(match.group(3))
                parsed_results['processes']['balance'] += float(match.group(4))
                parsed_results['processes']['stddev'] += float(match.group(5))
                parsed_results['processes']['iops_rate'] += float(match.group(6))
                parsed_results['processes']['objects'] += int(match.group(7))
                parsed_results['processes']['obj_rate'] += float(match.group(8))
                parsed_results['processes']['tp'] += float(match.group(9))
                parsed_results['processes']['op_max'] += float(match.group(10))
        
        if count > 0 and len(parsed_results['processes']) > 0:
            parsed_results['processes']['max'] /= count
            parsed_results['processes']['min'] /= count
            parsed_results['processes']['mean'] /= count
            parsed_results['processes']['balance'] /= count
            parsed_results['processes']['stddev'] /= count
            parsed_results['processes']['iops_rate'] /= count
            parsed_results['processes']['objects'] /= count
            parsed_results['processes']['obj_rate'] /= count
            parsed_results['processes']['tp'] /= count
            parsed_results['processes']['op_max'] /= count
        
        logger.info(f"Parsed results: {parsed_results}")

        # Extract total runtime
        runtime_match = re.search(runtime_pattern, results)
        if runtime_match:
            parsed_results['total_runtime'] = float(runtime_match.group(1))
        
        logger.info(f"Parsed results: {parsed_results}")
        
        return parsed_results


    def get_score(self, results):
        score_metric = self.score_metric
        score_metric_descriptions = self.score_metric_descriptions
        
        # Extract the maximum op_max value from all processes
        if type(score_metric) == str:
            if score_metric in list(score_metric_descriptions.keys()):
                score_value = results['processes'][score_metric]
                score_description = score_metric_descriptions[score_metric]
                return {
                    "value": score_value,
                    "description": score_description
                }
            else:
                raise ValueError(f"Score metric {score_metric} not found in results. Available metrics: {list(score_metric_descriptions.keys())}")
        elif type(score_metric) == list:
            score_value = {}
            score_description = {}
            for metric in score_metric:
                if metric in list(score_metric_descriptions.keys()):
                    score_value[metric] = results['processes'][metric]
                    score_description[metric] = score_metric_descriptions[metric]
                else:
                    raise ValueError(f"Score metric {metric} not found in results. Available metrics: {list(score_metric_descriptions.keys())}")
            return {
                "value": score_value,
                "description": score_description
            }
        else:
            raise ValueError(f"Invalid score metric: {score_metric}")




