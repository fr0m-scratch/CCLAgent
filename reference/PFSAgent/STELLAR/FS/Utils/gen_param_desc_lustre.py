from STELLAR.Utils.stellar_config import StellarConfig
from STELLAR.Utils.logger import LOGGERS
from STELLAR.RAG.rag_helper import init_rag_query_engine, retrieve_from_index
from litellm import completion
import json
import os
import re
import time
import litellm
from pydantic import BaseModel, Field
from typing import Union, Optional

litellm.enable_json_schema_validation=True

logger = LOGGERS["fs_config"]



class RangeMinMax(BaseModel):
    min: Union[int, str] = Field(description="The minimum value for the parameter")
    max: Union[int, str] = Field(description="The maximum value for the parameter")


class RangeStringValueList(BaseModel):
    string_values: Optional[list[str]] = Field(description="A list of valid string values for the parameter")

class RangeIntValueList(BaseModel):
    int_values: Optional[list[int]] = Field(description="A list of valid integer values for the parameter")



class OutputFormat(BaseModel):
    sufficient_info: bool = Field(description="True if the parameter has enough information to be described, False otherwise")
    definition: Optional[str] = Field(description="Brief technical definition")
    effect: Optional[str] = Field(description="Detailed description of performance/behavior effects")
    additional_info: Optional[str] = Field(description="Additional information or advice which may help optimize the parameter settings")
    range_description: Optional[str] = Field(description="Human readable description of valid values")
    range_value_type: Optional[str] = Field(description="The type of the value when setting the parameter (int or str)")
    range: Optional[Union[RangeMinMax, RangeStringValueList, RangeIntValueList]] = Field(description="The range of values for the parameter")


ORIGINAL_PARAM_DESC_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Docs", "lustre_params.json")

CONFIG = StellarConfig().get_instance().config
MODEL = CONFIG["Agent"]["model"]
RAG_CONFIG = CONFIG["RAG"]["Lustre_manual_index"]
RAG_CONFIG["source_data_dir"] = os.path.join(CONFIG["root_dir"], RAG_CONFIG["source_data_dir"])
RAG_CONFIG["index_dir"] = os.path.join(CONFIG["root_dir"], RAG_CONFIG["index_dir"])
SYSTEM_CONFIG = CONFIG["System"]


def query_rag(query_engine, query):
    query_result = retrieve_from_index(query_engine, query)
    return query_result


def update_param(param, param_path, query_engine):
    param_name = param

    rag_query_prompt = f"""
    How do I use the parameter {param_name}? It was found at {param_path}
    """
    rag_results = []
    rag_query_result = query_rag(query_engine, rag_query_prompt)
    parsed_rag_result = ""
    for node in rag_query_result:
        parsed_rag_result += f"{node.get_content()}\n"
        rag_results.append(node.get_content())
    

    system_prompt = """
    The assistant will always respond in proper JSON format.
    """

    llm_prompt = f"""
    I will provide you with a Lustre filesystem parameter and some information extracted from the Lustre documentation which may be related.
    Analyze the information from the documentation and provide a detailed description of the parameter including the definition, effect, type, range, and range_description if it is available in the documentation.

    Parameter name: {param_name}
    Path to tune via lctl: {param_path}
    Lustre version: 2.15.5
    Information extracted from the Lustre documentation: {parsed_rag_result}
    
    Provide the information in this format if it's a tunable parameter:
    {OutputFormat.model_json_schema()}

    You must follow these rules:
    - If the provided information from the documentation does not mention the parameter exactly as it is named in the "Parameter name" field, set "sufficient_info" to False.
    - If the parameter is not actually tunable, set "sufficient_info" to False.
    - If the documentation does mention the parameter but the information is not sufficient to describe the parameter, set "sufficient_info" to False.
    - If any range value must be set based the value of another parameter, specify it using this syntax: dependent(parameter_name). For example, if the parameter's max value is dependent on the value of the brw_size parameter, specify it as dependent(brw_size).
    - If any range value must be set based on a variable from the system, such as client memory or number of OSTs, use the appropriate variable from this list: [CLIENT_RAM, OST_COUNT, MDS_COUNT, CLIENT_COUNT, OST_PER_OSS].
    - If any range value needs to be calculated, specify it using this syntax: expression(calculation_formula). For example, if the parameter's max value should be set to half of the system's memory, specify it as expression(CLIENT_RAM / 2). Another example, if the stripecount parameter has a value range from -1 to number of OSTs-1, set the min value to -1 and the max value to expression(OST_COUNT-1).
    - If the parameter can only be enabled/disabled or is a boolean, specify its range as a list of integers [0, 1].
    - You should never set the range to a value that is vague such as "dependent on system requirements" or "dependent on workload"! Instead, describe exactly what it is dependent on.
    - If the parameter is dependent on a variable that is not easily defined, such as hardware performance, set the value to a number that makes sense for the context of the parameter.
    - If the documentation provides different range values for different Lustre versions, make sure that the range values are appropriate for Lustre version 2.15.5.

    """

    try:
        prompt_messages = [{
                "role": "system",
                "content": system_prompt
            },{
                "role": "user",
                "content": llm_prompt
            }]
        llm_response = completion(
            model=MODEL,
            messages=prompt_messages,
            response_format=OutputFormat,
            temperature=0.1
        )

        result = llm_response.choices[0].message.content
        logger.debug(f"Raw result: {result}")
        
        # First JSON parse
        parsed_response = json.loads(result)
        
        # Check if we got a string and try to parse it again
        if isinstance(parsed_response, str):
            logger.debug("First parse resulted in string, attempting second parse")
            try:
                parsed_response = json.loads(parsed_response)
            except json.JSONDecodeError:
                logger.error("Failed to parse double-encoded JSON")
                raise
        
        if not isinstance(parsed_response, dict):
            raise TypeError(f"Expected dict, got {type(parsed_response)}")
            
        return OutputFormat(**parsed_response), rag_results
    except Exception as e:
        for i in range(3):
            try:    
                time.sleep(20)
                logger.error(f"Error generating parameter description: {e}")
                llm_response = completion(
                    model=MODEL,
                    messages=[{
                    "role": "user",
                    "content": llm_prompt
                }],
                    response_format=OutputFormat,
                    temperature=0.3
                )
                result = llm_response.choices[0].message.content
                # Parse the JSON
                parsed_response = json.loads(result)
                logger.debug(f"Parsed response type: {type(parsed_response)}")
                logger.debug(f"Parsed response: {parsed_response}")
                
                if not isinstance(parsed_response, dict):
                    raise TypeError(f"Expected dict, got {type(parsed_response)}")
                    
                return OutputFormat(**parsed_response), rag_results
            except Exception as _e:
                logger.error(f"Error generating parameter description: {_e}")
                continue
        raise Exception(f"Failed to generate parameter description after 3 attempts: {e}")
        

def describe_params(params=None):
    query_engine = init_rag_query_engine(RAG_CONFIG)
    if not os.path.exists(ORIGINAL_PARAM_DESC_FILE):
        logger.error(f"Original parameter description file does not exist: {ORIGINAL_PARAM_DESC_FILE}")
        raise FileNotFoundError(f"Original parameter description file does not exist: {ORIGINAL_PARAM_DESC_FILE}")
    
    if params is None:
        with open(ORIGINAL_PARAM_DESC_FILE, "r") as f:
            params = json.load(f)
    else:
        params = params
    
    output_file = os.path.join(os.path.dirname(ORIGINAL_PARAM_DESC_FILE), f"{MODEL.split('/')[-1]}_described_lustre_params.json")
    described_params = {}
    insufficient_info_params = []
    all_rag_results = {}
    for param in params:
        param_path = params[param]["path"]
        param_desc, rag_results = update_param(param, param_path, query_engine)
        all_rag_results[param] = rag_results
        #time.sleep(1)
        if param_desc:
            if param_desc.sufficient_info:
                param_desc = param_desc.model_dump()
                param_desc.pop("sufficient_info")
                described_params[param] = {"description": param_desc, "path": params[param]["path"]}
            else:
                insufficient_info_params.append(param)
        with open(output_file, "w") as f:
            json.dump(described_params, f, indent=4)
    insufficient_info_params_file = os.path.join(os.path.dirname(output_file), f"{MODEL.split('/')[-1]}_insufficient_info_params.json")
    with open(insufficient_info_params_file, "w") as f:
        json.dump(insufficient_info_params, f, indent=4)
    rag_results_file = os.path.join(os.path.dirname(output_file), f"{MODEL.split('/')[-1]}_rag_results.json")
    with open(rag_results_file, "w") as f:
        json.dump(all_rag_results, f, indent=4)
    logger.info(f"Generated described parameter descriptions and saved to {output_file}")
    return described_params, insufficient_info_params






