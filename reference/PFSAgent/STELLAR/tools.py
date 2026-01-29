from STELLAR.Applications.base import Application

def finalize_tuning_process(reason: str):
    return f"Tuning process finalized with reason: {reason}"


FINALIZE_TUNING_TOOL_DESCRIPTION = \
f"""
PURPOSE: End the tuning process.

USAGE DETAILS:
- This tool is used when STELLAR is confident that the optimal file system parameters have been found.
- When using this tool, STELLAR provides detailed reasoning for why the tuning process should be ended by summarizing what was done and what was learned.

"""

DARSHAN_ANALYSIS_TOOL_DESCRIPTION = \
f"""
PURPOSE: Make a request to a Darshan analysis assistant to analyze a specific aspect of the application's runtime I/O behavior

USAGE DETAILS:
- This tool is used when STELLAR needs to know more about specific aspects of the application's runtime I/O behavior to help generate a more informed set of parameter predictions.
- When using this tool, STELLAR provides a question that the Darshan analysis assistant can answer along with all the relevant context needed to conduct a thorough analysis.
- When using this tool, STELLAR only asks questions which are within the scope of the analysis assistant listed below.

ANALYSIS ASSISTANT SCOPE:
- The analysis assistant can only answer specific questions about the runtime I/O behavior of the application.
- The analysis assistant does not know more about the file system parameters or their effects than what is already known by STELLAR.
- The analysis assistant can only analyze the trace files from the application's most recent run so it cannot answer questions about comparing the log contents across different runs.

"""




TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "finalize_tuning_process",
            "description": FINALIZE_TUNING_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for ending the tuning process",
                    },
                },
                "required": ["reason"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_further_analysis",
            "description": DARSHAN_ANALYSIS_TOOL_DESCRIPTION,
            "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the Darshan analysis assistant",
                },
                },
                "required": ["question"],
            },
        }
    }
]

TOOL_FUNCTIONS = {
    "finalize_tuning_process": finalize_tuning_process
}

def get_tools_and_functions(application: Application):
    TOOL_FUNCTIONS["request_further_analysis"] = application.analysis_agent.answer_question
    return TOOL_DEFINITIONS, TOOL_FUNCTIONS
