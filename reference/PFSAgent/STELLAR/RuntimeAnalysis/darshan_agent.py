from .utils import parse_darshan_log, parse_darshan_to_csv
from STELLAR.LLM import DarshanAnalysisPrompt, DarshanSummaryPrompt, DarshanQAPrompt
from STELLAR.FS import FSConfig
from STELLAR.Utils.logger import LOGGERS
from .base_agent import BaseAnalysisAgent
from datetime import datetime
import os
import json
import subprocess
import traceback
from typing import Literal

logger = LOGGERS["analysis_agent"]



class DarshanAnalysisAgent(BaseAnalysisAgent):
    def __init__(self, fs_config_class: FSConfig = None, log_aggregation_settings: None | Literal["all"] | datetime = None):
        super().__init__(fs_config_class)
        self.log_aggregation_settings = log_aggregation_settings
        self.summary_prompt = DarshanSummaryPrompt
        

    def initialize_analysis_env(self):
        self.analysis_dir = self.process_to_dir()
        self.darshan_modules = self.list_darshan_modules()
        self.analysis_file = os.path.join(self.analysis_dir, "analysis.json")
        self.summary_file = os.path.join(self.analysis_dir, "summary.txt")
        

    def process_to_dir(self):
        logger.info(f"Starting process_to_dir for application: {self.application_name}")
        try:
            log_contents, log_paths = parse_darshan_log(self.agent_config['Darshan']['log_dir'], 
                                                     self.agent_config['Darshan']['date_formatted_dir'],
                                                     self.log_aggregation_settings)
            logger.info(f"Successfully parsed Darshan log from {', '.join(log_paths)}")
            
            new_dir_name = f"{self.application_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            if self.application_name in self.analysis_dir.split("/")[-1]:
                root_analysis_dir = "/".join(self.analysis_dir.split("/")[:-1])
                new_dir = os.path.join(root_analysis_dir, new_dir_name)
            else:
                new_dir = os.path.join(self.analysis_dir, new_dir_name)
            logger.info(f"Creating new directory for processed data: {new_dir}")
            
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                logger.info(f"Created new directory: {new_dir}")
                
            parse_darshan_to_csv(log_contents, new_dir)
            logger.info(f"Successfully parsed Darshan data to CSV in {new_dir}")
            return new_dir
        except Exception as e:
            logger.error(f"Error in process_to_dir: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    


    def list_darshan_modules(self):
        logger.info("Starting to list Darshan modules")
        try:
            modules = []
            for file in os.listdir(self.analysis_dir):
                if file.endswith(".csv"):
                    module_name = file.split(".")[0]
                    modules.append(module_name)
            logger.info(f"Found Darshan modules: {modules}")
            return modules
        except Exception as e:
            logger.error(f"Error listing Darshan modules: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    

    def prepare_interpreter_session(self):
        logger.info("Starting to prepare interpreter session")
        try:
            executed_code = \
"""
import pandas as pd
import numpy as np
import os

header = open('header.txt', 'r').read()

"""     
            forbidden_chars = ["-", " "]
            for module_name in self.darshan_modules:
                new_module_name = module_name
                for char in forbidden_chars:
                    new_module_name = new_module_name.replace(char, "_")
                executed_code += f"{new_module_name}_data = pd.read_csv('{module_name}.csv')\n"
                executed_code += f"{new_module_name}_description = open('{module_name}_description.txt', 'r').read()\n"
            
            logger.info("Executing setup code in interpreter")
            self.interpreter.computer.run("python", executed_code, display=True)
            logger.debug(f"Executed code: {executed_code}")
            return executed_code
        except Exception as e:
            logger.error(f"Error in prepare_interpreter_session: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


    def setup_qa_environment(self):
        self.initialize_analysis_env()
        try:
            os.chdir(self.analysis_dir)
            logger.info(f"Changed working directory to: {self.analysis_dir}")
            setup_code = self.prepare_interpreter_session()
            return setup_code
        except Exception as e:
            logger.error(f"Error in setup_qa_environment: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def run_qa_analysis(self, question: str, current_step: int):
        load_new_environment = False
        working_dir = os.getcwd()
        try:
            kwargs = {"question": question}
            if current_step is None:
                raise ValueError("current_step is required for QA analysis")
            elif current_step > 0:
                load_new_environment = True
            if load_new_environment:
                setup_code = self.setup_qa_environment()
                kwargs["setup_code"] = setup_code
            kwargs["new_environment"] = load_new_environment
            prompt = DarshanQAPrompt(**kwargs)
            messages = self.interpreter.chat(prompt.get_messages()[0]["content"])
            logger.log_messages_json(messages)
            return messages
        except Exception as e:
            logger.error(f"Error in run_qa_analysis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if load_new_environment:
                os.chdir(working_dir)


    def interpret_all_modules(self):
        logger.info("Starting module interpretation")
        working_dir = os.getcwd()
        logger.info(f"Current working directory: {working_dir}")
        try:
            os.chdir(self.analysis_dir)
            logger.info(f"Changed working directory to: {self.analysis_dir}")
            
            setup_code = self.prepare_interpreter_session()
            
            logger.info("Preparing analysis prompt")
            prompt = DarshanAnalysisPrompt(self.darshan_modules, setup_code, 
                                         self.fs_config_class.describe())
            
            logger.info("Starting interpreter chat session")
            messages = self.interpreter.chat(prompt.get_messages()[0]["content"])
            
            logger.info(f"Saving analysis to file: {self.analysis_file}")
            with open(self.analysis_file, "w") as f:
                json.dump(messages, f, indent=4)
            
            logger.info("Logging analysis messages")
            logger.log_messages_json(messages)
            
            return messages
        except Exception as e:
            logger.error(f"Error in interpret_all_modules: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            logger.info(f"Changing working directory back to: {working_dir}")
            os.chdir(working_dir)

    
        

    def run(self):
        logger.info("Starting DarshanAnalysisAgent run")
        self.initialize_analysis_env()
        try:
            if self.agential_analysis:
                analysis_messages = self.interpret_all_modules()
                logger.info("Successfully completed module interpretation")
                logger.log_messages_json(analysis_messages)
                analysis_summary = self.summarize_insights()
                logger.info("Successfully generated analysis summary")
                logger.debug(f"Analysis summary: {analysis_summary}")
                return analysis_messages, analysis_summary
            else:
                analysis_result = self.run_analysis_script(os.path.join(os.path.dirname(__file__), "scripts", "darshan_agent"), "analyze.py")
                return None, analysis_result

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    

    def run_darshan_parser(self, log_path):
        logger.info(f"Running darshan-parser on: {log_path}")
        try:
            result = subprocess.run(
                ['darshan-parser', log_path],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Successfully ran darshan-parser")
            logger.debug(f"Parser stdout: {result.stdout[:200]}...")  # Log first 200 chars
            return {
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running darshan-parser: {str(e)}")
            logger.error(f"Parser stderr: {e.stderr}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
