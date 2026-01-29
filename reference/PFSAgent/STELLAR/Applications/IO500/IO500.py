from STELLAR.Applications import Application
import os
import re
import configparser
import random
from STELLAR.Utils.logger import LOGGERS
import json


logger = LOGGERS["application"]


class IO500(Application):
    name = "IO500"
    description = "IO500 is a benchmark that measures the performance of storage systems."
    runtime_description = None
    score_metric = ["bandwidth", "iops", "total"]
    score_metric_descriptions = {
        "total": "Total score, combining the bandwidth and iops scores",
        "bandwidth": "Bandwidth score",
        "iops": "IOPS score"
        }


    def __init__(self, config_name: str = None):
        super().__init__(config_name)
        self.set_config_var("global", "datadir", self.data_dir)
        self.set_config_var("global", "resultdir", self.results_dir)

    def build(self):
        pass

    def parse_config(self):
        selected_config = f"{self.config_name}.ini"
        configs_folder = os.path.join(os.path.dirname(__file__), "configs")
        config_files = os.listdir(configs_folder)
        if selected_config not in config_files:
            raise ValueError(f"Config file {selected_config} not found in {configs_folder}")
        config_path = os.path.join(configs_folder, selected_config)
        config_content = configparser.ConfigParser()
        config_content.read(config_path)
        config_dict = {
            "config_path": config_path,
            "config_content": config_content
        }
        return config_dict
    

    def set_config_var(self, section, key, value):
        config_dict = self.parse_config()
        config_content = config_dict["config_content"]
        config_path = config_dict["config_path"]
        if not section in config_content:
            raise ValueError(f"Section {section} not found in {config_path}")
        if not key in config_content[section]:
            raise ValueError(f"Key {key} not found in section {section} in {config_path}")
        config_content[section][key] = value
        with open(config_path, "w") as f:
            config_content.write(f)
        
    
    def get_run_command(self, application_config_dict: dict) -> list:
        return [os.path.join(os.path.dirname(__file__), "run.sh"), application_config_dict["config_path"], self.application_root_dir]


    def parse_results(self):
        results_dir = self.results_dir
        # get the most recent timestamped directory in results_dir
        timestamped_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        most_recent_dir = max(timestamped_dirs, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
        results_summary_file = os.path.join(results_dir, most_recent_dir, "result_summary.txt")
        results = {
            "phases": {},
            "overall": {
                "score": None,
                "scorex": None
            }
        }

        with open(results_summary_file, "r") as f:
            lines = f.readlines()

        # Regex patterns for parsing
        # RESULT line example:
        # [RESULT] ior-easy-write        0.016383 GiB/s : time 35.387 seconds [INVALID]
        # We'll use a regex to capture: 
        # test_name, value, unit, time, and optional INVALID
        result_pattern = re.compile(
            r'^\[RESULT\]\s+([^\s]+)\s+([\d\.]+)\s+([A-Za-z\/]+)\s*:\s*time\s+([\d\.]+)\s+seconds(.*)$'
        )
        
        # SCORE line example:
        # [SCORE ] Bandwidth 0.162411 GiB/s : IOPS 0.000000 kiops : TOTAL 0.000000 [INVALID]
        score_pattern = re.compile(
            r'^\[SCORE\s*\]\s+Bandwidth\s+([\d\.]+)\s+GiB/s\s*:\s+IOPS\s+([\d\.]+)\s+kiops\s*:\s+TOTAL\s+([\d\.]+)(.*)$'
        )
        
        # SCOREX line is similar:
        scorex_pattern = re.compile(
            r'^\[SCOREX\]\s+Bandwidth\s+([\d\.]+)\s+GiB/s\s*:\s+IOPS\s+([\d\.]+)\s+kiops\s*:\s+TOTAL\s+([\d\.]+)(.*)$'
        )

        for line in lines:
            line = line.strip()

            # Parse RESULT lines
            m = result_pattern.match(line)
            if m:
                test_name = m.group(1)
                value = float(m.group(2))
                unit = m.group(3)
                time_val = float(m.group(4))
                remainder = m.group(5)
                valid = "[INVALID]" not in remainder
                if value > 0:
                    results["phases"][test_name] = {
                        "value": value,
                        "unit": unit,
                        "time": time_val,
                        "valid": valid
                    }
                continue

            # Parse SCORE line
            s = score_pattern.match(line)
            if s:
                bw = float(s.group(1))
                iops = float(s.group(2))
                total = float(s.group(3))
                remainder = s.group(4)
                valid = "[INVALID]" not in remainder
                results["overall"]["score"] = {
                    "bandwidth": bw,
                    "iops": iops,
                    "total": total,
                    "valid": valid
                }
                continue

            # Parse SCOREX line
            sx = scorex_pattern.match(line)
            if sx:
                bw = float(sx.group(1))
                iops = float(sx.group(2))
                total = float(sx.group(3))
                remainder = sx.group(4)
                valid = "[INVALID]" not in remainder
                results["overall"]["scorex"] = {
                    "bandwidth": bw,
                    "iops": iops,
                    "total": total,
                    "valid": valid
                }
                continue
        logger.info(f"Complete run results: {json.dumps(results, indent=4)}")
        return results
    
    
    def get_score(self, results):
        score_metric = self.score_metric

        if type(score_metric) == str:
            if score_metric in results["overall"]["score"] and score_metric in self.score_metric_descriptions:
                score_value = results["overall"]["score"][score_metric]
                score_description = self.score_metric_descriptions[score_metric]
            else:
                raise ValueError(f"Invalid score metric: {score_metric}")
        elif type(score_metric) == list:
            score_value = {}
            score_description = {}
            for metric in score_metric:
                if metric in results["overall"]["score"] and metric in self.score_metric_descriptions:
                    score_value[metric] = results["overall"]["score"][metric]
                    score_description[metric] = self.score_metric_descriptions[metric]
                else:
                    raise ValueError(f"Invalid score metric: {metric}")
        else:
            raise ValueError(f"Invalid score metric: {score_metric}")
        return {
            "value": score_value,
            "description": score_description
        }



