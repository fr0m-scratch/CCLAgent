from STELLAR.LLM import (
    ElbenchoAnalysisPrompt,
    ElbenchoSummaryPrompt
)
from STELLAR.FS import FSConfig
from STELLAR.Utils.logger import LOGGERS
from .base_agent import BaseAnalysisAgent
from datetime import datetime
import json
import os
import traceback
import shutil

logger = LOGGERS["analysis_agent"]


class ElbenchoAnalysisAgent(BaseAnalysisAgent):
    def __init__(self, fs_config_class: FSConfig = None):
        super().__init__(fs_config_class)
        self.analysis_dir = os.path.join(self.agent_config['RuntimeAnalysis']['analysis_dir'], f"{self.application_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.docs_file = os.path.join(self.agent_config['Application_Configs']['Elbencho']['docs_file'])
        self.summary_prompt = ElbenchoSummaryPrompt


    def initialize_analysis_env(self):
        self.setup_analysis_dir()
        self.analysis_file = os.path.join(self.analysis_dir, "analysis.json")
        self.summary_file = os.path.join(self.analysis_dir, "summary.json")

    def setup_analysis_dir(self):
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
            logger.info(f"Created analysis directory: {self.analysis_dir}")
        else:
            logger.info(f"Analysis directory already exists: {self.analysis_dir}")
        
        results_file = os.path.join(self.agent_config['Application_Configs']['Elbencho']['results_dir'], "results.csv")
        # copy results file to analysis dir
        shutil.copy(results_file, os.path.join(self.analysis_dir, "results.csv"))
        logger.info(f"Copied results file to analysis directory: {os.path.join(self.analysis_dir, 'results.csv')}")
        self.results_file = os.path.join(self.analysis_dir, "results.csv")



    def analyze_elbencho_results(self):
        working_dir = os.getcwd()
        logger.info(f"Current working directory: {working_dir}")
        try:
            os.chdir(self.analysis_dir)
            setup_code = self.prepare_interpreter_session()
            prompt = ElbenchoAnalysisPrompt(setup_code, self.fs_config_class.describe())

            logger.info("Starting interpreter chat session")
            messages = self.interpreter.chat(prompt.get_messages()[0]["content"])

            logger.info(f"Saving analysis to file: {self.analysis_file}")
            with open(self.analysis_file, "w") as f:
                json.dump(messages, f, indent=4)

            return messages
        except Exception as e:
            logger.error(f"Error in analyze_elbencho_results: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            logger.info(f"Changing working directory back to: {working_dir}")
            os.chdir(working_dir)


    def setup_qa_environment(self):
        self.initialize_analysis_env()

    def run_qa_analysis(self, question: str, current_step: int):
        messages = self.interpreter.chat(question)
        logger.log_messages_json(messages)
        return messages


    def get_qa_prompt(self, question: str):
        #prompt = ElbenchoQAPrompt(question)
        #return prompt.get_messages()[0]["content"]
        return question

    def prepare_interpreter_session(self):
        logger.info("Starting to prepare interpreter session")
        try:
            executed_code = \
f"""
import pandas as pd
import numpy as np
import os

results = pd.read_csv('{self.results_file}')
column_descriptions = open('{self.docs_file}', 'r').read()


"""     
            logger.info("Executing setup code in interpreter")
            self.interpreter.computer.run("python", executed_code, display=True)
            logger.debug(f"Executed code: {executed_code}")
            return executed_code
        except Exception as e:
            logger.error(f"Error in prepare_interpreter_session: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def run(self):
        logger.info("Starting ElbenchoAnalysisAgent run")
        self.initialize_analysis_env()
        try:
            if self.agential_analysis:
                analysis_messages = self.analyze_elbencho_results()
                logger.info("Successfully completed elbencho results analysis")
                logger.log_messages_json(analysis_messages)
                
                analysis_summary = self.summarize_insights()
                logger.info("Successfully generated analysis summary")
                logger.debug(f"Analysis summary: {analysis_summary}")
                
                return analysis_messages, analysis_summary
            else:
                self.run_analysis_script(os.path.join(os.path.dirname(__file__), "scripts", "elbencho_agent"), "analyze.py")
        except Exception as e:
            logger.error(f"Error in run method: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        

