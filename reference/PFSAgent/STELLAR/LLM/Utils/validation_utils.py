from litellm import get_supported_openai_params



def validate_output_support(model):
    supported_params = get_supported_openai_params(model)
    if "response_format" in supported_params:
        return True
    else:
        raise ValueError(f"Model {model} does not support structured output")


