from STELLAR.FS import FSConfig
from STELLAR.Applications import Application
from STELLAR.Utils.logger import LOGGERS
from typing import Union, Dict
import uuid
import time
import json
import subprocess
from datetime import datetime

logger = LOGGERS["candidate"]

class Candidate():
    id: str
    tuning_config: FSConfig
    application: Application
    status: str = None
    score: Union[float, Dict[str, float]] = None
    score_description: Union[str, Dict[str, str]] = None
    description: str = None
    complete_results: Union[dict, None] = None

    def __init__(self, tuning_config: FSConfig, application: Application, enable_runtime_analysis: bool = False):
        logger.info(f"INITIALIZING CANDIDATE")
        self.id = f"{time.time()}|{uuid.uuid4()}"
        self.tuning_config = tuning_config
        self.application = application
        self.enable_runtime_analysis = enable_runtime_analysis

    def __lt__(self, other):
        if type(self.score) == float:
            return self.score < other.score
        else:
            return self.score["total"] < other.score["total"]
    
    def __gt__(self, other):
        if type(self.score) == float:
            return self.score > other.score
        else:
            return self.score["total"] > other.score["total"]

    def __eq__(self, other):
        if type(self.score) == float:
            return self.score == other.score
        else:
            return self.score["total"] == other.score["total"]
    
    def get_config_parameters(self):
        return self.tuning_config.model_dump_json()
    
    def get_score_and_description(self):
        return {
            "score": self.score,
            "description": self.score_description
        }
    
    
    def get_candidate_description(self):
        return self.description

    def __str__(self):
        output = ""
        if self.description is not None:
            output += f"Description: {self.description}\n"

        output += f"Configuration_parameters:\n{self.tuning_config.model_dump_json(indent=4)}\n"
        
        if self.score is not None:
            output += f"Score: {self.score}\n"

        return output
    
    def to_json(self):
        json_dict = {
            "description": self.description,
            "config_parameters": self.tuning_config.model_dump_json(),
            "score": self.score,
            "score_description": self.score_description
        }
        return json.dumps(json_dict, indent=4)
    


    def combine_complete_results(self, new_complete_results):
        logger.info(f"Combining complete results: {new_complete_results}")
        for phase in new_complete_results["phases"].keys():
            logger.info(f"Phase: {phase}")
            if phase not in self.complete_results["phases"].keys():
                logger.info(f"Phase not in complete results")
                self.complete_results["phases"][phase] = new_complete_results["phases"][phase]
            else:
                logger.info(f"Phase in complete results")
                self.complete_results["phases"][phase]["value"] += new_complete_results["phases"][phase]["value"]
        for metric in new_complete_results["overall"]["score"].keys():
            logger.info(f"Metric: {metric}")
            if metric not in self.complete_results["overall"]["score"].keys():
                logger.info(f"Metric not in complete results")
                self.complete_results["overall"]["score"][metric] = new_complete_results["overall"]["score"][metric]
            else:
                logger.info(f"Metric in complete results")
                self.complete_results["overall"]["score"][metric] += new_complete_results["overall"]["score"][metric]

    def init_and_run(self):
        # clear cache with ior
        #self.application.clear_cache_with_ior()
        # clear io500 dir and remount lustre
        self.application.reset_FS()
        # initialize the tuning config
        self.tuning_config.initialize()
        # log the output of "df -h"
        command = ['df', '-h']
        output = subprocess.run(command, capture_output=True, text=True)
        logger.info(f"Output of {command}: {output.stdout}")

        # set the log aggregation settings if they are dependent on the current time
        if self.application.log_aggregation_settings and self.application.log_aggregation_settings != "all":
            current_time = datetime.now()
            self.application.set_log_aggregation_settings(current_time)

        score, complete_results = self.application.run_and_score(runtime_analysis=self.enable_runtime_analysis)
        logger.info(f"Complete results: {complete_results}")
        logger.info(f"Score: {score}")
        if self.complete_results is None:
            logger.info(f"Complete results is None")
            self.complete_results = complete_results

        else:
            logger.info(f"Complete results is not None: {self.complete_results}")
            self.combine_complete_results(complete_results)


        if score is not None:
            if self.score is None:
                logger.info(f"Score is None")
                self.score = score["value"]
            else:
                logger.info(f"Score is not None: {self.score}")
                if type(self.score) == float:
                    logger.info("score is Float")
                    self.score += score["value"]
                else:
                    logger.info(f"score is {type(self.score)}")
                    for metric in self.score.keys():
                        logger.info(f"Metric: {metric}")
                        self.score[metric] += score["value"][metric]
            self.score_description = score["description"]
        else:
            raise Exception("Score is None")
        return True
    
    def calculate_average_score(self, run_count: int):
        logger.info(f"Calculating average score: {self.score}")
        if self.score is not None:
            logger.info(f"Score is not None: {self.score}")
            if type(self.score) == float:
                logger.info(f"Score is Float: {self.score}")
                self.score = self.score / run_count
            else:
                logger.info(f"Score is Dict: {self.score}")
                for metric in self.score.keys():
                    logger.info(f"Metric: {metric}")
                    self.score[metric] = self.score[metric] / run_count
        
        if self.complete_results is not None:
            logger.info(f"Complete results is not None: {self.complete_results}")
            # Average the phase results
            for phase in self.complete_results["phases"].keys():
                logger.info(f"Phase: {phase}")
                self.complete_results["phases"][phase]["value"] = self.complete_results["phases"][phase]["value"] / run_count
            # Average the overall scores
            for metric in self.complete_results["overall"]["score"].keys():
                logger.info(f"Metric: {metric}")
                self.complete_results["overall"]["score"][metric] = self.complete_results["overall"]["score"][metric] / run_count


    def generate_description(self):
        pass





