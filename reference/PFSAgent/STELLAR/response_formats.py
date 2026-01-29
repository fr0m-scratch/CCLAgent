from pydantic import BaseModel, Field, create_model
from typing import Union


class ParameterPrediction(BaseModel):
    reasoning: str = Field(description="A detailed description of a prediction for what the best value for this parameter should be along with strong justifications.")
    predicted_value: str = Field(description="The predicted best value for this parameter.")

def CreatePredictionFormat(tuning_config_class: BaseModel):
    fields = {}
    for param in tuning_config_class.__annotations__.keys():
        if "stripe" in param:
            fields[param] = (ParameterPrediction, Field(description=f"Your prediction and reasoning for which directories should apply custom stripe settings and what the stripe settings should be. Remember that you can specify multiple directories, each with their own stripe settings."))
        else:
            fields[param] = (ParameterPrediction, Field(description=f"Your prediction and reasoning for what the best value for {param} should be."))

    return create_model(f"generate_tuning_prediction", **fields)


class ParameterExperienceFormat(BaseModel):
    parameter_name: str = Field(description="The name of the parameter that was tuned.")
    rule_description: str = Field(description="A generalized description of what was learned about tuning the parameter. 'generalized' means that the rule does not specify a particular value for the parameter, but rather a general strategy for tuning the parameter for the given workload.")
    tuning_context: str = Field(description="A detailed description of the I/O behavior context which for which the experience applies")


class ExperienceSynthesisFormat(BaseModel):
    tuning_rules: Union[list[ParameterExperienceFormat], str] = Field(description="A list of rules for tuning the file system parameters based on the experience you have gained from the tuning process. If you have not found any clear trends in the parameter space, you should just say 'No rules found'.")