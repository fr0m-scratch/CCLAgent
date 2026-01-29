from STELLAR.Utils.stellar_config import StellarConfig
from STELLAR.Utils.logger import LOGGERS
from litellm import completion
import json
import os
from pydantic import BaseModel


logger = LOGGERS["fs_config"]

CONFIG = StellarConfig.get_instance().config
MODEL = CONFIG["Agent"]["model"]

DESCRIBED_PARAM_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Docs", f"{MODEL.split('/')[-1]}_described_lustre_params.json")
PERFORMANCE_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Docs", f"{MODEL.split('/')[-1]}_performance_lustre_params.json")
TUNED_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Docs", f"{MODEL.split('/')[-1]}_tuned_lustre_params.json")
REASONING_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Docs", f"{MODEL.split('/')[-1]}_reasoning_lustre_params.json")

class ParamImportanceFormat(BaseModel):
    has_significant_impact: bool
    should_be_tuned_from_default: bool
    explanation: str

def analyze_parameter_impact(param_name, param_info):
    """
    Query LLM to determine if a parameter has significant performance impact.
    """
    # Skip parameters with insufficient information
    if param_info['description'] == "Insufficient information":
        return False

    system_prompt = """
    The assistant will always respond in proper JSON format.
    """

    prompt = f"""
    I need you to help me filter the tunable Lustre client side parameters to extract the ones that are most likely to have a significant impact on application performance and whose default values are not likely to deliver good performance.
    I will provide you with a parameter name, a description of the parameter, and the path I have found where I can tune the parameter via lctl (e.g. if the path is osc.*.max_rpcs_in_flight, then I can tune the parameter via lctl set_param osc.*.max_rpcs_in_flight=<value>).
    Your task is to analyze the parameter and determine two things:
    1. If it is likely to have a significant impact on application performance.
    2. If the parameter should be tuned from the default value or if the default value is likely to already deliver good performance.

    Use the following format to respond:
    {{
        "has_significant_impact": "true" if the parameter has a significant impact on application performance or "false" if it does not.
        "should_be_tuned_from_default": "true" if the parameter should be tuned from the default value or "false" if it is likely already near optimal at the default value.
        "explanation": "Explain why the parameter has a significant impact on application performance and why the value should be tuned."
    }}

    It is extremely important to be conservative in your assessment to minimize the number of parameters that are tuned.
    Also keep in mind that tuning actions will only be taken on the client side so consider the path to tune the parameter via lctl when determining if the parameter has a significant impact. \
    For example, ldlm.*.threads_min tuned from the client side will not affect the threads_min parameter on the MDS or OSS and so it is unlikely to have a significant impact on application performance.

    Parameter name: {param_name}
    Parameter description: {json.dumps(param_info['description'], indent=2)}
    Path to tune via lctl: {param_info['path']}
    """

    try:
        response = completion(
            model=MODEL,
            messages=[{
                "role": "system",
                "content": system_prompt
            },{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            response_format=ParamImportanceFormat
        )
        
        result = response.choices[0].message.content
        result = json.loads(result)
        result = ParamImportanceFormat(**result)
        logger.info(f"param_name: {param_name}")
        logger.info(f"Result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing parameter {param_name}: {str(e)}")
        return False

def filter_performance_parameters(described_params=None):
    """
    Filter the described parameters to identify those with significant performance impact.
    """
    if not os.path.exists(DESCRIBED_PARAM_FILE):
        logger.error("Described parameter file does not exist")
        raise FileNotFoundError("Described parameter file does not exist")

    if described_params is None:
        with open(DESCRIBED_PARAM_FILE, "r") as f:
            described_params = json.load(f)

    performance_params = {}
    should_be_tuned_params = {}
    param_reasoning = {}

    
    for param_name, param_info in described_params.items():
        logger.info(f"Analyzing parameter: {param_name}")
        
        result = analyze_parameter_impact(param_name, param_info)

        if result.has_significant_impact:
            performance_params[param_name] = param_info
            logger.info(f"Parameter {param_name} identified as performance-related")

            if result.should_be_tuned_from_default:
                should_be_tuned_params[param_name] = param_info
                logger.info(f"Parameter {param_name} identified as should be tuned from default")
        
        param_reasoning[param_name] = {
            "has_significant_impact": result.has_significant_impact,
            "should_be_tuned_from_default": result.should_be_tuned_from_default,
            "explanation": result.explanation
        }
        # Sleep to avoid rate limiting
        #time.sleep(1)

        # Save intermediate results
        with open(PERFORMANCE_OUTPUT_FILE, "w") as f:
            json.dump(performance_params, f, indent=4)
        with open(TUNED_OUTPUT_FILE, "w") as f:
            json.dump(should_be_tuned_params, f, indent=4)
        with open(REASONING_OUTPUT_FILE, "w") as f:
            json.dump(param_reasoning, f, indent=4)
        
        logger.info(f"Analysis complete. Found {len(performance_params)} performance-related parameters")
        logger.info(f"Results saved to {PERFORMANCE_OUTPUT_FILE}")
        logger.info(f"Results saved to {TUNED_OUTPUT_FILE}")

    
    return performance_params, should_be_tuned_params

if __name__ == "__main__":
    performance_params, should_be_tuned_params = filter_performance_parameters() 