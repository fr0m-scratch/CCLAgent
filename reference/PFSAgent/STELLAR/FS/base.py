from pydantic import BaseModel

class FSConfig(BaseModel):
    
    @staticmethod
    def describe():
        raise NotImplementedError("describe method not implemented")
        
    
    def validate(self):
        try:
            for param, field in self.model_fields.items():
                value = getattr(self, param)
                if hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is list:
                    if value is not None:  # Check if the list exists
                        for item in value:
                            if hasattr(item, 'validate'):
                                item.validate()
                else:
                    if value is not None and hasattr(value, 'validate'):
                        value.validate()
            return True
        except Exception as e:
            raise ValueError(f"Error validating {self.__class__.__name__}: {e}") from e
    
    def initialize(self):
        try:
            for param, field in self.model_fields.items():
                value = getattr(self, param)
                if hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is list:
                    if value is not None:  # Check if the list exists
                        for item in value:
                            if hasattr(item, 'initialize'):
                                item.initialize()
                else:
                    if value is not None and hasattr(value, 'initialize'):
                        value.initialize()
        except Exception as e:
            raise ValueError(f"Error initializing {self.__class__.__name__}: {e}") from e
        
    @staticmethod
    def read():
        raise NotImplementedError("read method not implemented")


class DependentRangeValue():
    """Represents a range that depends on other parameters"""
    def __init__(self, dependent_param: str, expression: str):
        self.dependent_param = dependent_param
        self.expression = expression

    def __json__(self):
        return {
            "dependent_param": self.dependent_param,
            "expression": self.expression
        }
    

class PlanDescription(BaseModel):
    plan_description: str

    @staticmethod
    def describe():
        return {
            "plan_description": "A high-level description of the plan to generate the configuration based on the provided information"
        }
    
    def validate(self):
        return True
    
    def initialize(self):
        return True
    
    @staticmethod
    def read():
        return "This is the default configuration"
