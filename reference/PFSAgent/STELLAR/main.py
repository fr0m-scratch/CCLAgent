from STELLAR.Candidates import Candidate, CandidateDB
from STELLAR.Applications import APPLICATIONS, TUNING_EXPERIENCE
from STELLAR.FS import get_lustre_config
from STELLAR.LLM.prompt import InitialGenerationPrompt, PredictionIterationPrompt, AnalysisIterationPrompt, ConfigDistillationPrompt, ExperienceSynthesisPrompt
from STELLAR.LLM.completions import generate_completion
from STELLAR.LLM.messages import MessageThread
from STELLAR.Utils.logger import StellarLogger, LOGGERS
from STELLAR.Utils.stellar_config import StellarConfig
from STELLAR.response_formats import CreatePredictionFormat, ExperienceSynthesisFormat
from STELLAR.tools import get_tools_and_functions
import json
import re
import traceback
import os
logger = LOGGERS["main"]
tuning_knowledge_logger = LOGGERS["tuning_knowledge"]
runtime_description_logger = LOGGERS["runtime_description"]
config_distillation_logger = StellarLogger("config_distillation")
stellar_config_logger = StellarLogger("stellar_config")


class StellarAgent:

    def __init__(self, run_config: StellarConfig, 
                 starting_params: dict = None, 
                 enable_runtime_analysis: bool = True, 
                 use_rule_set: bool = False,
                 eval_result_type: str = None,
                 aggregate_results_file: str = None,
                 speedup_results_file: str = None,
                 eval_result_source: str = None,
                 rule_set_path: str = None):
        self.eval_result_type = eval_result_type
        self.aggregate_results_file = aggregate_results_file
        self.speedup_results_file = speedup_results_file
        self.eval_result_source = eval_result_source
        self.use_rule_set = use_rule_set
        self.rule_set_path = rule_set_path

        # Load the run config
        self.run_config = run_config
        stellar_config_logger.info(f"Run config: {json.dumps(self.run_config.config, indent=4)}")

        # Load the tuning config class
        self.fs_config = get_lustre_config()
        if starting_params:
            valid_fields = set(self.fs_config.__annotations__.keys())
            filtered_params = {k: v for k, v in starting_params.items() if k in valid_fields}
            self.default_config = self.fs_config(**filtered_params)
        else:
            #initialize from current file system settings
            self.default_config = self.fs_config.read()


        # Load the application
        application_name = self.run_config.config["Application"]
        application_config_name = self.run_config.config["Application_Config"]
        if application_name not in APPLICATIONS:
            raise ValueError(f"Invalid application: {application_name}")
        self.application_class = APPLICATIONS[application_name]
        self.previous_tuning_experience = None
        if self.use_rule_set:
            self.previous_tuning_experience = self.load_tuning_rule_set(self.rule_set_path)
        self.application = self.application_class(application_config_name)
        self.application.init_analysis_agent(self.fs_config)
        self.application.build()


        self.agential_analysis = self.run_config.config["RuntimeAnalysis"]["agential"]
        self.PredictionFormat = CreatePredictionFormat(self.fs_config)
        tool_definitions, tool_functions = get_tools_and_functions(self.application)
        tool_definitions.append({"type": "function", "function": {"name": "generate_tuning_prediction", 
                                                                  "description": "Generate a new prediction for the file system parameters which will deliver the best performance for the application.",
                                                                  "parameters": self.PredictionFormat.model_json_schema()}})
        self.tool_definitions = tool_definitions
        self.tool_functions = tool_functions

        self.experience_synthesis_format = ExperienceSynthesisFormat

        # Initialize a candidate with the starting config
        initial_candidate = Candidate(tuning_config=self.default_config, application=self.application, enable_runtime_analysis=enable_runtime_analysis)

        # Initialize the candidate database
        self.candidate_db = CandidateDB(initial_candidate, log_dir=self.run_config.config["Logging"]["log_dir"])

        # Initialize model
        self.model = self.run_config.config["Agent"]["model"]

        self.iterative_prediction_messages = None

        self.current_step_idx = 0
        self.current_tool_calls_total = 0

    def load_tuning_rule_set(self, rule_set_path: str):
        with open(rule_set_path, "r") as f:
            return json.load(f)

    def parse_completion(self, completion, extract_think: bool = False):
        if extract_think:
            thinking_tokens = re.findall(r'<think>.*?</think>', completion, flags=re.DOTALL)
            logger.info(f"Thinking tokens: {thinking_tokens}")
            completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL).strip()

        completion = json.loads(completion)
        if len(list(completion.keys())) == 1:
            completion = completion[list(completion.keys())[0]]
        for key in completion:
            if key != "stripe_settings":
                if not type(completion[key]) == dict:
                    completion[key] = {key: completion[key]}
            else:
                if not type(completion[key]) == list:
                    if key in completion[key]:
                        completion[key] = [completion[key][key]]
                    else:
                        completion[key] = [completion[key]]
        return completion
    


    def distill_config_from_prediction(self, prediction):
        distillation_prompt = ConfigDistillationPrompt(self.fs_config, prediction)
        distillation_messages = distillation_prompt.get_messages()
        distillation_completion = generate_completion("gpt-4o", distillation_messages, response_format=self.fs_config)
        distillation_messages.add_messages([{"role": "assistant", "content": distillation_completion}])
        distillation_completion = self.parse_completion(distillation_completion)
        new_config = self.fs_config(**distillation_completion)
        return new_config, distillation_messages
    
    
    def retry_distillation(self, distillation_messages: MessageThread, error: ValueError):
        verify_config_value_message = \
f"""
The configuration you suggested is not valid.

Here are the errors:
{error}

Please suggest a new configuration which adheres to the ranges specified in the following JSON schema:
{json.dumps(self.fs_config._model_json_schema(), indent=4)}
"""
        distillation_messages.add_messages([{"role": "user", "content": verify_config_value_message}])
        distillation_completion = generate_completion("gpt-4o", distillation_messages.dump(), response_format=self.fs_config)
        distillation_messages.add_messages([{"role": "assistant", "content": distillation_completion}])
        try:
            distillation_completion = self.parse_completion(distillation_completion)
            new_config = self.fs_config(**distillation_completion)
            if new_config.validate():
                return new_config
            else:
                raise ValueError("New config is not valid")
        except Exception as e:
            logger.error(f"Error in retry_distillation: {e}")
            raise e
    


    def generate_candidate_from_prediction(self, tool_call):
        prediction = json.loads(tool_call["function"]["arguments"])
        prediction = self.PredictionFormat(**prediction)

        new_config, distillation_messages = self.distill_config_from_prediction(prediction)
        try:
            new_config.validate()
        except ValueError as e:
            new_config = self.retry_distillation(distillation_messages, e)

        new_candidate = Candidate(tuning_config=new_config, application=self.application)
        self.candidate_db.add_candidate(new_candidate) 
        logger.info(f"New candidate config: {new_candidate.tuning_config.model_dump_json(indent=4)}")
        logger.info(f"New candidate score: {new_candidate.score}")

        return new_candidate



    def parse_tool_call(self, response):
        if "tool_calls" in response:
            self.iterative_prediction_messages.add_messages([{"role": "assistant", "content": response["response"], "tool_calls": [tool_call.to_dict() for tool_call in response["tool_calls"]]}])
            logger.log_message(self.iterative_prediction_messages[-1])
            self.current_tool_calls_total += 1
            for tool_call in response["tool_calls"]:
                function_to_call = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                if function_to_call == "generate_tuning_prediction":
                    function_output = self.generate_candidate_from_prediction(tool_call)
                    self.current_step_idx += 1
                else:
                    tool_call_args = tool_call["function"]["arguments"]
                    tool_call_args = json.loads(tool_call_args)
                    if function_to_call == "request_further_analysis":
                        tool_call_args["current_step"] = self.current_step_idx
                    function_output = self.tool_functions[function_to_call](**tool_call_args)

                result = {"type": function_to_call, "tool_call_id": tool_call_id, "content": function_output}
                return result
        else:
            raise ValueError("No tool calls found in response")



    def generate_first_step(self):
        kwargs = {
            "tuning_config_class": self.candidate_db.tuning_config_class,
            "application": self.application,
            "initial_candidate": self.candidate_db.candidates[0],
            "prediction_format": self.PredictionFormat,
            "use_runtime_description": True,
            "system_config": self.run_config.config["System"],
            "tools": self.tool_definitions,
            "experience_synthesis": self.previous_tuning_experience
        }
        generation_prompt = InitialGenerationPrompt(**kwargs)
        generation_messages = generation_prompt.get_messages()
        logger.log_messages_json(generation_messages)
        self.iterative_prediction_messages = generation_messages
        generation_completion = generate_completion(self.model, generation_messages, tools=self.tool_definitions)
        return generation_completion


    def generate_iterative_step(self, last_function_result: dict):
        if last_function_result["type"] == "request_further_analysis":
            kwargs = {
                "assistant_analysis": last_function_result["content"]
            }
            analysis_iteration_prompt = AnalysisIterationPrompt(**kwargs)
            analysis_iteration_user_message = analysis_iteration_prompt.get_user_message()
            
            analysis_iteration_user_message["tool_call_id"] = last_function_result["tool_call_id"]
            analysis_iteration_user_message["name"] = last_function_result["type"]
            analysis_iteration_user_message["role"] = "tool"
            self.iterative_prediction_messages.add_messages([analysis_iteration_user_message])
            logger.log_message(analysis_iteration_user_message)
            generation_completion = generate_completion(self.model, self.iterative_prediction_messages, tools=self.tool_definitions)
        
        elif last_function_result["type"] == "generate_tuning_prediction":
            kwargs = {
                "tuning_config_class": self.candidate_db.tuning_config_class,
                "previous_candidate": self.candidate_db.candidates[-1]
            }
            prediction_iteration_prompt = PredictionIterationPrompt(**kwargs)
            prediction_iteration_user_message = prediction_iteration_prompt.get_user_message()

            prediction_iteration_user_message["role"] = "tool"
            prediction_iteration_user_message["tool_call_id"] = last_function_result["tool_call_id"]
            prediction_iteration_user_message["name"] = last_function_result["type"]
            self.iterative_prediction_messages.add_messages([prediction_iteration_user_message])
            logger.log_message(prediction_iteration_user_message)
            generation_completion = generate_completion(self.model, self.iterative_prediction_messages, tools=self.tool_definitions)

        else:
            raise ValueError(f"Invalid function result type: {last_function_result['type']}")

        return generation_completion
        


    def step(self, last_function_result: dict = None):
        if not last_function_result:
            agent_completion = self.generate_first_step()
        else:
            agent_completion = self.generate_iterative_step(last_function_result)
        
        new_result = self.parse_tool_call(agent_completion)

        if self.application.runtime_description is not None:
            runtime_description_logger.log_runtime_description(self.application.runtime_description, self.current_step_idx)

        return new_result
        
    
    def synthesize_tuning_experience(self, function_result: dict):
        experience_synthesis_prompt = ExperienceSynthesisPrompt(self.experience_synthesis_format)
        experience_synthesis_user_message = experience_synthesis_prompt.get_user_message()
        experience_synthesis_user_message["role"] = "tool"
        experience_synthesis_user_message["tool_call_id"] = function_result["tool_call_id"]
        experience_synthesis_user_message["name"] = function_result["type"]
        self.iterative_prediction_messages.add_messages([experience_synthesis_user_message])
        logger.log_message(experience_synthesis_user_message)
        experience_synthesis_completion = generate_completion(self.model, self.iterative_prediction_messages, response_format=self.experience_synthesis_format)
        self.iterative_prediction_messages.add_messages([{"role": "assistant", "content": experience_synthesis_completion}])
        logger.log_message(self.iterative_prediction_messages[-1])
        if "no rules found" in experience_synthesis_completion.lower():
            return "No rules found"
        else:
            experience_synthesis = json.loads(experience_synthesis_completion)
            experience_synthesis = self.experience_synthesis_format(**experience_synthesis)
            return experience_synthesis
    
    
    def get_eval_alias(self):
        application_name = self.run_config.config["Application"]
        application_config = self.run_config.config["Application_Config"]
        if "amrex" in application_config:
            return "AMReX"
        elif "macsio" in application_config:
            if "512k" in application_config:
                return "MACSio_512k"
            elif "16m" in application_config:
                return "MACSio_16m"
        elif "IOR" in application_name:
            if "16m" in application_config:
                return "IOR_16m"
            elif "64k" in application_config:
                return "IOR_64k"
        elif "MDWorkbench" in application_name:
            if "8192" in application_config:
                return "MDWorkbench_8K"
            elif "2048" in application_config:
                return "MDWorkbench_2K"
        elif "IO500" in application_name:
            return "IO500"
        else:
            raise ValueError(f"Invalid application config: {application_config}")

    def save_results(self, include_candidates_after_idx: int = 0):
        application_alias = self.get_eval_alias()
        self.candidate_db.save_walltime(application_alias, 
                                        self.eval_result_type, 
                                        aggregate_results_file=self.aggregate_results_file, 
                                        result_source=self.eval_result_source, 
                                        speedup_results_file=self.speedup_results_file,
                                        include_candidates_after_idx=include_candidates_after_idx,
                                        using_rule_set=self.use_rule_set)
        self.candidate_db.plot_candidates()
        if self.application.name == "IO500":
            self.candidate_db.plot_complete_results_IO500()
        else:
            self.candidate_db.save_complete_results()

    def save_experience(self, experience_synthesis):
        path = os.path.join(self.run_config.config["Logging"]["log_dir"], "experience_synthesis.json")
        with open(path, "w") as f:
            f.write(experience_synthesis.model_dump_json(indent=4))

    def run_best_candidate(self):
        best_candidate = self.candidate_db.get_best_candidate()
        application = self.application_class(self.run_config.config["Application_Config"])
        if best_candidate:
            self.candidate_db.add_final_candidate(best_candidate, application)
        else:
            raise ValueError("No candidates found")

    def log_results(self):
        logger.log_benchmark_comparison_tables(self.candidate_db)
        logger.log_final_candidates(self.candidate_db)
        logger.log_final_scores(self.candidate_db)

    
    
    def run_no_change(self, max_steps: int = 5):
        try:
            for i in range(max_steps):
                new_candidate = Candidate(tuning_config=self.default_config, application=self.application)
                self.candidate_db.add_candidate(new_candidate)
                logger.info(f"New candidate: {new_candidate.score}")
                self.current_step_idx += 1
        except Exception as e:
            logger.error(f"Error in run_no_change: {e}")
            raise e
        finally:
            try:
                self.save_results()
                self.log_results()
            except Exception as e:
                logger.error(f"Error in saving results: {e}")
                logger.error(f"traceback: {traceback.format_exc()}")


    def run_with_best_config(self, iterations: int):
        best_config = self.candidate_db.get_best_candidate().tuning_config
        logger.info(f"BEST CONFIG FOUND BY TUNING: {best_config.model_dump_json(indent=4)}")
        if "Test_config" in self.run_config.config:
            application_config_name = self.run_config.config["Test_config"]
        else:
            application_config_name = self.run_config.config["Application_Config"]
        
        application = self.application_class(application_config_name)
        application.init_analysis_agent(self.fs_config)

        for i in range(iterations):
            new_candidate = Candidate(tuning_config=best_config, application=application, enable_runtime_analysis=False)
            self.candidate_db.add_candidate(new_candidate)
            logger.info(f"New candidate: {new_candidate.score}")
            self.current_step_idx += 1

    def run(self, max_steps: int = 7, test_config_iterations: int = 0):
        try:
            function_result = None
            total_steps = 0
            while (self.current_step_idx < max_steps) and (total_steps < 30):
                function_result = self.step(function_result)
                if function_result["type"] == "finalize_tuning_process":
                    experience_synthesis = self.synthesize_tuning_experience(function_result)
                    self.save_experience(experience_synthesis)
                    break
                total_steps += 1
            if self.eval_result_type == "aggregate":
                #save the speedup results for benchmarks so we don't have to rerun the benchmarks for AE
                application_alias = self.get_eval_alias()
                self.candidate_db.save_walltime(application_alias, 
                                                "speedup", 
                                                aggregate_results_file=self.aggregate_results_file, 
                                                speedup_results_file=self.speedup_results_file,
                                                result_source=self.eval_result_source,
                                                using_rule_set=self.use_rule_set) 

            walltime_candidates_idx = 0
            if test_config_iterations > 0:
                walltime_candidates_idx = len(self.candidate_db.candidates)
                self.run_with_best_config(test_config_iterations)
            
        except Exception as e:
            logger.error(f"Error in run: {e}")
            logger.error(f"traceback: {traceback.format_exc()}")
            raise e
        finally:
            try:
                self.save_results(walltime_candidates_idx)
                self.log_results()
            except Exception as e:
                logger.error(f"Error in saving results: {e}")
                logger.error(f"traceback: {traceback.format_exc()}")