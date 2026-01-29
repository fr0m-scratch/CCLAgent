from STELLAR.Candidates import Candidate
from STELLAR.Applications import Application
from STELLAR.FS import FSConfig
from STELLAR.Utils.logger import LOGGERS
from .messages import MessageThread
import json
from pydantic import BaseModel

logger = LOGGERS["completion"]

class Prompt():
    system_prompt: str = None
    user_prompt: str

    def __str__(self):
        return f"System Context: {self.system_context}\nPrompt: {self.user_prompt}"
    
    def get_user_message(self):
        return {"role": "user", "content": self.user_prompt}
    
    def get_messages(self):
        if self.system_prompt is None:
            logger.info(f"System prompt is None, returning user prompt: {self.user_prompt}")
            return MessageThread([{"role": "user", "content": self.user_prompt}])
        else:
            return MessageThread([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.user_prompt}])
        
    def add_system_prompt_section(self, prompt_section: str):
        self.system_prompt += f"\n{prompt_section}\n"
        
    def add_user_prompt_section(self, prompt_section: str):
        self.user_prompt += f"\n{prompt_section}\n"


class InitialGenerationPrompt(Prompt):
    system_prompt: str = \
"""
The assistant is named STELLAR, an expert in tuning file system parameters to maximize the performance of HPC applications.
STELLAR helps the user tune a set of file system parameters to maximize the performance of an application they are running.

The user provides STELLAR with the following information:
- High level information about the system the application is running on
- The application they are interested in maximizing the performance of
- A summarized description of the application's I/O behavior collected at runtime
- The application's performance when run with the system's default file system parameters
- A description of the file system parameters the user wants to tune


STELLAR tunes the file system by:
1) generating reasonable predictions for the optimal file system parameters to use given the context provided by the user
2) updating the predictions based on the results of re-running the application with the previous prediction
3) repeating the process until it is confident that the predictions are optimal.

STELLAR always follows these important rules when tuning the file system:
- STELLAR DOES NOT STOP its tuning process when there is still reasonable potential for further performance improvement.
- STELLAR DOES STOP the tuning process when it is confident that the predictions are optimal and further tuning would not result in any additional performance gains.
- When STELLAR makes a prediction and the application's performance declines as a result, STELLAR will always try to understand why this occurred and make a new prediction which corrects the issue.

"""
    user_prompt: str = "You are given the following information:\n"

    def __init__(self, tuning_config_class: FSConfig, system_config: dict = None, application: Application = None, 
                 initial_candidate: Candidate = None, prediction_format: BaseModel = None, 
                 use_runtime_description: bool = True, tools = None, experience_synthesis: dict = None):
        
        self.experience_synthesis = experience_synthesis
        if self.experience_synthesis is not None:
            self.add_system_prompt_section(self.format_experience_synthesis())
        
        if tools is not None:
            # add prediction_format to tools
            self.add_system_prompt_section(self.format_tools(tools))

        if system_config is not None:
            self.system_config = system_config
            self.add_user_prompt_section(self.format_system_info())

        self.tuning_config_class = tuning_config_class
        self.application = application
        self.initial_candidate = initial_candidate
        self.prediction_format = prediction_format

        self.add_user_prompt_section(self.format_application_info(use_runtime_description))
        self.add_user_prompt_section(self.format_tuning_config())
        self.add_user_prompt_section(self.format_task_description())


    def format_experience_synthesis(self):
        return \
f"""
From previous tuning attempts, STELLAR has learned a set of important lessons and uses them to guide future tuning attempts.
Each lesson contains a description of what was learned as well as a 'tuning_context' which describes the context (ie. the specific I/O behavior) where the lesson was learned.
Here are the lessons STELLAR has learned:
{json.dumps(self.experience_synthesis, indent=4)}
"""

    def format_tools(self, tools):
        analysis_str = ""
        if "request_further_analysis" in tools:
            analysis_str = "Whenever STELLAR uses the request_further_analysis tool, the user will ask an assistant to conduct a detailed analysis to answer STELLAR's request and then provide STELLAR with the results of the analysis."
        return \
f"""
At each step in the process of tuning the file system, STELLAR has access to the following tools to help it tune the file system:

{json.dumps(tools, indent=4)}

Whenever STELLAR uses the generate_tuning_prediction tool, the user will always implement the predicted best parameter settings, re-run the application, and then provide STELLAR with the measured results.
{analysis_str}
Whenever STELLAR uses the finalize_tuning_process tool, the user will consider the tuning process complete and ask STELLAR to synthesize the experience gained from the tuning process into a list of rules.

STELLAR always follows these important rules when using the provided tools:
- At each step, STELLAR only uses one of the tools.
- STELLAR always responds using the chosen tool's expected format.
"""


    def format_system_info(self):
        return \
f"""
### **System Information:**

The system where the application is being run is described by the following information:

 - There is a Lustre file system mounted at {self.system_config["mount_root"]}
 - There are {len(self.system_config["MDS"])} MDS servers in the Lustre file system.
 - There are {len(self.system_config["OSS"])} OSS servers in the Lustre file system. Each OSS has {self.system_config["OST_PER_OSS"]} OSTs meaning there are {len(self.system_config["OSS"]) * self.system_config["OST_PER_OSS"]} OSTs in total.
 - There are {len(self.system_config["Clients"])} client nodes in the Lustre file system.
 - Each client uses a {self.system_config["Client_processor"]["model"]} processor with {self.system_config["Client_processor"]["cores_per_socket"]} cores per socket, {self.system_config["Client_processor"]["threads_per_core"]} threads per core, {self.system_config["Client_processor"]["sockets"]} sockets, and {self.system_config["Client_processor"]["NUMA_nodes"]} NUMA nodes.
 - Each server uses a {self.system_config["Server_processor"]["model"]} processor with {self.system_config["Server_processor"]["cores_per_socket"]} cores per socket, {self.system_config["Server_processor"]["threads_per_core"]} threads per core, {self.system_config["Server_processor"]["sockets"]} sockets, and {self.system_config["Server_processor"]["NUMA_nodes"]} NUMA nodes.
 - The Lustre file system is connected via a {self.system_config["Network"]["type"]} network with a rated speed of {self.system_config["Network"]["speed"]}.
"""

    def format_application_info(self, use_runtime_description):
        content = \
f"""
### **Application Information:**

The application being optimized for is {self.application.name}

The general description of the application is as follows: {self.application.description}

The application was run using the system's default file system parameter settings and received the following score:
score: {self.initial_candidate.complete_results}
score_description: {self.application.score_metric_descriptions}

The system's default file system parameter settings are as follows:
{self.initial_candidate.tuning_config.model_dump_json(indent=4)}
"""
        if use_runtime_description:
            content += \
f"""
The application's I/O behavior was recorded using Darshan and the following summary was compiled by analyzing \
the recorded data which may help to tune the file system parameters to improve performance of the application:
{self.application.runtime_description}
"""
        return content
    

    def format_tuning_config(self):
        return \
f"""
### **Tuning Parameters:**

These are the parameters which may be tuned to improve application performance:
```json
{json.dumps(self.tuning_config_class._model_json_schema()["$defs"], indent=4)}
```
"""
    

    def format_task_description(self):
        return \
f"""
### **Task Description:**

Using the information above, help me tune the file system parameters to improve the performance of the application. Remember, you must use one of the provided tools.

"""
    


class PredictionIterationPrompt(Prompt):
    user_prompt: str = ""

    def __init__(self, tuning_config_class: FSConfig, previous_candidate: Candidate):
        
        self.tuning_config_class = tuning_config_class
        self.previous_candidate = previous_candidate

        self.add_user_prompt_section(self.format_previous_result())


    def format_previous_result(self):
        return \
f"""
Based on the predictions you made, I created the following configuration:
{self.previous_candidate.tuning_config.model_dump_json(indent=4)}

I ran the configuration and it received the following score: {self.previous_candidate.complete_results}. 

Remember you must use one of the provided tools. What tool do you want to use next?
"""
    

    
class AnalysisIterationPrompt(Prompt):
    user_prompt: str = ""

    def __init__(self, assistant_analysis: str):
        self.add_user_prompt_section(self.format_assistant_analysis(assistant_analysis))


    def format_assistant_analysis(self, assistant_analysis: str):
        return \
f"""
I gave my analysis assistant your request and it responded with the following information:
{assistant_analysis}

Remember you must use one of the provided tools. What tool do you want to use next?
"""

    

class ConfigDistillationPrompt(Prompt):
    system_prompt: str = \
"""
You are helpful assistant you always responds in proper JSON format.
The user is tuning a file system and needs to convert a description of desired tuning parameter values into a valid JSON object which can be used to set the file system parameters.
Your response must be a valid JSON object conforming to the schema provided by the user. You must not provide any commentary outside the JSON.
"""
    user_prompt: str = ""

    def __init__(self, tuning_config_class: FSConfig, parameter_predictions: BaseModel):
        self.tuning_config_class = tuning_config_class
        self.parameter_predictions = parameter_predictions
        self.add_user_prompt_section(self.format_parameter_predictions())
        self.add_user_prompt_section(self.add_task_description())

        logger.info("ConfigDistillationPrompt:")
        logger.log_messages_json(self.get_messages())


    def format_parameter_predictions(self):
        return \
f"""
### **Description of Desired Tuning Parameters:**

The user has provided the following description of the desired tuning parameters:
{self.parameter_predictions.model_dump_json(indent=4)}
"""
    
    def add_task_description(self):
        return \
f"""
### **Task Description:**

Your job is to convert the description of the desired tuning parameters into a valid JSON object which can be used to set the file system parameters. \
Your response must follow this JSON schema:
```json
{json.dumps(self.tuning_config_class._model_json_schema(), indent=4)}
```

Remember:
 - Your response must be a valid JSON object conforming to the schema provided by the user.
 - You must not provide any commentary outside the JSON.

"""




ANALYSIS_INSTRUCTION_NOTES = \
"""
Remember:
 - The application code will not be able to be changed, so you must only focus on information which can help to tune the file system parameters to improve performance of the application as it is currently written.
 - **DO NOT RUN COMMANDS TO CHANGE THE FILE SYSTEM PARAMETERS**, as this will be handled later by the user after reviewing your analysis.
 - **DO NOT SUGGEST ANY SPECIFIC COMMANDS TO RUN**, as the user is already an expert in implementing file system configuration changes.
 - **DO NOT CREATE ANY PLOTS OR GRAPHS**.
 - Keep these instructions as part of your plan so you do not forget them later in the analysis process.
"""





class DarshanAnalysisPrompt(Prompt):
    system_prompt: str = None
    user_prompt: str = "Here is some context before I give you the task:\n"

    def __init__(self, darshan_modules: list[str], setup_code: str = None, tuning_config_description: dict = None):
        self.tuning_config_description = tuning_config_description
        if self.tuning_config_description is not None:
            self.add_user_prompt_section(self.format_tuning_config_context(self.tuning_config_description))

        self.darshan_modules = darshan_modules
        self.add_user_prompt_section(self.format_darshan_modules_description(self.darshan_modules))

        self.setup_code = setup_code
        if self.setup_code is not None:
            self.add_user_prompt_section(self.format_darshan_analysis_setup_code(self.setup_code))
        else:
            self.add_user_prompt_section(self.format_darshan_analysis_no_setup_code())

        self.add_user_prompt_section(self.format_task_description())
        self.add_user_prompt_section(self.format_instruction_notes())

    
    def format_tuning_config_context(self, tuning_config_description: str):
        tuning_config_context = \
f"""
### **Tuning Configuration:**

I am trying to tune these file system parameters to achieve maximal performance on my HPC application:
```
{json.dumps(tuning_config_description, indent=4)}
```
"""
        return tuning_config_context


    def format_darshan_modules_description(self, darshan_modules: list[str]):
        darshan_modules_description = \
f"""
### **Darshan Modules:**

In order to decide which parameters to tune and how to tune them, I have run the application and traced its I/O behavior using Darshan. \
The application used these Darshan modules: {darshan_modules}.
"""
        return darshan_modules_description


    def format_darshan_analysis_setup_code(self, setup_code: str):
        setup_code_context = \
f"""
### **Environment Setup:**

I have processed the Darshan log by splitting each recorded Darshan module into one dataframe and one description string. \
Each module's description string contains information about the data columns in that module's corresponding dataframe as well as some important information about interpreting them, while the dataframe contains the actual data for the described columns.\
There is also a string called `header` which contains the information extracted from the start of the Darshan log which describes the application's total runtime, number of processes used, etc.\
This is the code I already ran in the environment to setup the data:

```
{setup_code}
```
"""
        return setup_code_context

    
    def format_darshan_analysis_no_setup_code(self):
        setup_context = \
"""
### **Environment Setup:**

I have created a directory of files which contain the data of a Darshan log, split into one data and one description file per Darshan module.\
The description file contains information about the counter columns in the data file as well as some important information about interpreting them. The data file contains the actual data collected by the Darshan module.\
There is also a file called 'header.txt' which represents the header of the Darshan log where some information such as application name and runtime is stored.\
My environment is already setup within the directory and you can access the files directly.
"""
        return setup_context



    def format_task_description(self):
        return \
f"""
### **Task Description:**

 1) Inspect the dataframes and description variables to understand the data columns and what they represent.
 2) Then, find which unique directories are accessed by the application.
 3) Then, you must analyze the data from the Darshan log to extract the most important information which may help guide me to tune file system parameters to improve performance of the application.

"""


    def format_instruction_notes(self):
        instruction_notes = ANALYSIS_INSTRUCTION_NOTES
        self.add_user_prompt_section(instruction_notes)





class DarshanSummaryPrompt(Prompt):
    system_prompt: str = \
"""
A user has asked an assistant to analyze a darshan log and extract any useful knowledge which may help to tune the file system parameters to improve the performance of the application which was traced using Darshan.
The analysis consists of an initial message from the user detailing the task for the assistant, followed by a series of messages between the assistant and the CLI console where the assistant describes it's analysis plan, uses the CLI to run the analysis code, and then interprets the results of the code that was run via the console's output.

"""
    user_prompt: str = ""

    def __init__(self, analysis: str):
        self.add_user_prompt_section(self.format_analysis_context(analysis))
        self.add_user_prompt_section(self.format_task_description())

        logger.info("DarshanSummaryPrompt:")
        logger.log_messages_json(self.get_messages())


    def format_analysis_context(self, analysis: str):
        analysis_context = \
f"""
Here is the full log of messages documenting the analysis conducted by the assistant:
{analysis}
"""
        return analysis_context


    def format_task_description(self):
        task_description = \
f"""


### **Task Description:**

You must review the analysis conducted by the assistant and generate a detailed summary of all of the information the assistant discovered through the analysis process to summarize the detailed I/O behavior of the application.
Your summary should include specific information discovered by the assistant about the application's I/O behavior which may be helpful to tune the file system parameters to improve performance of the application.
"""
        return task_description
    

class DarshanQAPrompt(Prompt):
    user_prompt: str = ""

    def __init__(self, question: str, new_environment: bool, setup_code: str = None):
        self.setup_code = setup_code
        self.question = question
        self.new_environment = new_environment
        if self.new_environment:
            self.add_user_prompt_section(self.format_new_env_description(self.setup_code))
        self.add_user_prompt_section(self.format_question(self.question))


    def format_new_env_description(self, setup_code: str):
        return \
f"""
I have rerun the application and replaced the CSV and description files in your current environment with the new ones.
"""
    
    def format_question(self, question: str):
        return \
f"""
Please reload the files and help me answer the following question:
{question}
"""





class ElbenchoAnalysisPrompt(Prompt):
    system_prompt: None
    user_prompt: str = "Here is some context before I give you the task:\n"

    def __init__(self, setup_code: str, fs_config_description: str):
        self.fs_config_description = fs_config_description
        if self.fs_config_description is not None:
            self.add_user_prompt_section(self.format_tuning_config_context(self.fs_config_description))


        self.setup_code = setup_code
        if self.setup_code is not None:
            self.add_user_prompt_section(self.format_elbencho_setup_code(self.setup_code))

        self.add_user_prompt_section(self.format_task_description())
        self.add_user_prompt_section(self.format_instruction_notes())

    
    def format_tuning_config_context(self, tuning_config_description: str):
        tuning_config_context = \
f"""
### **Tuning Configuration:**

I am trying to tune these file system parameters to achieve maximal performance on my HPC application:
```
{json.dumps(tuning_config_description, indent=4)}
```
"""
        return tuning_config_context


    
    def format_elbencho_setup_code(self, setup_code: str):
        setup_code_context = \
f"""
### **Environment Setup:**

I have loaded the Elbencho results csv file into a pandas dataframe and have also loaded a text-based description of the columns in the dataframe into a variable called `column_descriptions`. \
This is the code I already ran in the environment to setup the data:

```
{setup_code}
```
"""
        return setup_code_context


    def format_task_description(self):
        task_description = \
f"""
### **Task Description:**

 1) Inspect the CSV and description files to understand the data columns and what they represent.
 2) Then, find which unique directories are accessed by the application.
 3) Then, you must analyze the data from the Elbencho results to extract the most important information which may help guide me to tune file system parameters to improve performance of the application.

"""
        return task_description


    def format_instruction_notes(self):
        instruction_notes = ANALYSIS_INSTRUCTION_NOTES
        return instruction_notes




class ElbenchoSummaryPrompt(Prompt):
    system_prompt: str = \
"""
A user has asked an assistant to analyze a results csv file generated by the Elbencho benchmark and extract any useful knowledge which may help to tune the file system parameters to improve the performance of the application.
The analysis consists of an initial message from the user detailing the task for the assistant, followed by a series of messages between the assistant and the CLI console where the assistant describes it's analysis plan, uses the CLI to run the analysis code, and then interprets the results of the code that was run via the console's output.

"""
    user_prompt: str = ""

    def __init__(self, analysis: str):
        self.add_user_prompt_section(self.format_analysis_context(analysis))
        self.add_user_prompt_section(self.format_task_description())

        logger.info("ElbenchoSummaryPrompt:")
        logger.log_messages_json(self.get_messages())


    def format_analysis_context(self, analysis: str):
        analysis_context = \
f"""
Here is the full log of messages documenting the analysis conducted by the assistant:
{analysis}
"""
        return analysis_context


    def format_task_description(self):
        task_description = \
f"""


### **Task Description:**

You must review the analysis conducted by the assistant and generate a detailed summary of all of the information the assistant discovered through the analysis process.
Your summary should include specific information discovered by the assistant about the application's behavior which may be helpful to tune the file system parameters to improve performance of the application. This should be very detailed and include all of the information discovered by the assistant.
"""
        return task_description




class ExperienceSynthesisPrompt(Prompt):
    user_prompt: str = ""

    def __init__(self, experience_synthesis_format: BaseModel):
        self.experience_synthesis_format = experience_synthesis_format
        self.add_user_prompt_section(self.format_task_description())


    def format_task_description(self):
        task_description = \
f"""\
Thank you for helping me tune the file system parameters!

Since you have now made multiple attempts to tune the file system parameters, I need you to extract the general lessons which you have definitively learned about tuning the parameters for the given application and synthesize them into a list of rules.
I will use these rules in the future to further explore the parameter space at a finer granularity and I want to avoid trying values in ranges which do not make sense or have been proven to not work.
The rules should not focus on the specific values of the parameters, but rather the general trends which you have learned when tuning specific parameters unless the rule is very specific to a single parameter value.
The rules should also not specify specific application names in the description or tuning context as they should be generally applicable to any application which has similar I/O behavior.
It is extremely important that you only create rules which have been clearly verified by expected application performance changes based on your predictions. 
It is ok to not create any rules if you have not found any clear trends in the parameter space.

Here is an example of when you SHOULD NOT create a rule:

```
During tuning, you initially predicted that parameter P1 should be set to 1000 from a default value of 100 and P2 should be set to 2000 from a default value of 500.
You then tested those values and found that P1 = 1000 and P2 = 2000 resulted in significant performance regression.
Then, you updated your prediction to P1 = 500 and P2 = 1000 and found that the application's performance improved.


Why you should NOT create a rule:
- Since you initially increased both parameter values significantly, and then decreased them together as well, there is not sufficient evidence to definitively say that increasing P1 or P2 individually will result in performance gains or losses.
```

Here is an example of when you SHOULD create a rule:
```
During tuning, you initially predicted that parameter P1 should be set to 1000 from a default value of 100 and P2 should be set to 2000 from a default value of 500.
You then tested those values and found that P1 = 1000 and P2 = 2000 resulted in significant performance regression.
Then, you updated your prediction to reduce P1 to 500 but keep P2 at 2000 and found that the application's performance improved.
Finally, you made one more prediction to keep P1 at 500 and increase P2 to 3000 and found that the application's performance improved again.


Why you SHOULD create a rule:
- Since you initially increased P1 and P2 together, then reduced P1 but kept P2 the same, and then kept P1 the same but increased P2, there is sufficient evidence to definitively say that increasing P2 will result in performance gains for the observed I/O patterns of the application.
```

Now, please create the rules based on your experience tuning the file system parameters for the given application.
You must use the following JSON schema to format your response:
{json.dumps(self.experience_synthesis_format.model_json_schema(), indent=4)}
"""
        return task_description

