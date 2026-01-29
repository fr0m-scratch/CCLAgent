from interpreter import OpenInterpreter
from STELLAR.Utils.stellar_config import StellarConfig
from STELLAR.Utils.logger import LOGGERS
from STELLAR.FS import FSConfig
from STELLAR.LLM import generate_completion
import json
import os
import subprocess
import shutil
import traceback

logger = LOGGERS["analysis_agent"]


def init_code_interpreter():
    interpreter = OpenInterpreter()
    interpreter.llm.model = "gpt-4.1"
    interpreter.auto_run = True
    interpreter.loop = True
    interpreter.llm.stream = False
    interpreter.llm.max_tokens = 8192
    interpreter.llm.context_window = 200000
    return interpreter

class BaseAnalysisAgent:
    def __init__(self, fs_config_class: FSConfig = None):
        self.agent_config = StellarConfig.get_instance().config
        self.analysis_dir = self.agent_config['RuntimeAnalysis']['analysis_dir']
        self.agential_analysis = self.agent_config['RuntimeAnalysis']['agential']
        self.application_name = self.agent_config['Application']
        self.interpreter = init_code_interpreter()
        self.fs_config_class = fs_config_class
        self.summary_prompt = None
        self.analysis_file = None
        self.summary_file = None


    def prepare_interpreter_session(self):
        raise NotImplementedError("prepare_interpreter_session must be implemented by the subclass")


    def run(self):
        raise NotImplementedError("run must be implemented by the subclass")

    def run_analysis_script(self, script_path: str, script_name: str):
        logger.info("Starting analysis script")
        # analysis script is a python which is at ./scripts/darshan_agent/analyze.py
        # the script needs to be copied to the analysis_dir
        working_dir = os.getcwd()
        try:
            shutil.copy(script_path, self.analysis_dir)
            os.chdir(self.analysis_dir)
            # run the script
            result = subprocess.run(
                ['python', script_name],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Successfully ran analysis script")
            logger.debug(f"Script stdout: {result.stdout[:200]}...")  # Log first 200 chars
            return result.stdout
        except Exception as e:
            logger.error(f"Error in run_analysis_script: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise 
        finally:
            os.chdir(working_dir)


    def parse_completion(self, completion):
        logger.debug(f"Analysis Summary Completion: {completion}")
        try:
            # First try to parse the completion as JSON
            completion_dict = json.loads(completion)
            
            # Check if we have a nested structure with a single key
            if len(list(completion_dict.keys())) == 1:
                completion_dict = completion_dict[list(completion_dict.keys())[0]]
                
            return completion_dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse completion as JSON: {str(e)}")
            logger.error(f"Raw completion: {completion}")
            raise Exception(f"Failed to parse completion as JSON: {str(e)}")
    

    def summarize_insights(self):
        logger.info("Starting to summarize insights")
        try:
            logger.info(f"Reading analysis from: {self.analysis_file}")
            with open(self.analysis_file, "r") as f:
                analysis_messages = json.load(f)
            
            logger.info("Preparing summary prompt")
            summary_prompt = self.summary_prompt(analysis_messages)
            
            logger.info("Generating completion for summary")

            analysis_summary = generate_completion(
                self.agent_config['Agent']['model'], 
                summary_prompt.get_messages()
            )
            
            logger.info(f"Saving summary to: {self.summary_file}")
            with open(self.summary_file, "w") as f:
                f.write(analysis_summary)
            
            return analysis_summary
        except Exception as e:
            logger.error(f"Error in summarize_insights: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Error in summarize_insights: {str(e)}")
        

    
    def run_qa_analysis(self, question: str, current_step: int):
        raise NotImplementedError("run_qa_analysis must be implemented by the subclass")
    

    def agential_setting_question_answer(self, question: str, current_step: int):
        logger.info(f"Answering question: {question}")
        try:
            messages = self.run_qa_analysis(question, current_step)
            logger.log_messages_json(messages)
            return messages[-1]["content"]
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Error in answer_question: {str(e)}")
        
    def script_setting_question_answer(self, question: str):
        raise NotImplementedError("script_setting_question_answer must be implemented by the subclass")
    

    def answer_question(self, question: str, current_step: int = None):
        if self.agential_analysis:
            return self.agential_setting_question_answer(question, current_step)
        else:
            return self.script_setting_question_answer(question)

        

