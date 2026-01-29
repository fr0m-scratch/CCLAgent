
from STELLAR.Utils.stellar_config import StellarConfig
from STELLAR.Utils.logger import LOGGERS
from STELLAR.FS.Utils import validate_path, validate_size, validate_integer, run_command
from .base import FSConfig, DependentRangeValue, PlanDescription
import re
import os
import json
from typing import Dict, Any, Type, Optional, Union
from pydantic import BaseModel, create_model, Field


logger = LOGGERS["fs_config"]


def get_system_config():
    return StellarConfig.get_instance().config['System']

SYSTEM_CONFIG = get_system_config()

param_definitions_path = os.path.join(os.path.dirname(__file__), "Docs", StellarConfig.get_instance().config["param_definitions"] + ".json")
with open(param_definitions_path, "r") as f:
    PARAM_DEFINITIONS = json.load(f)


######################################################################
# Base Lustre Parameter Class
######################################################################
class LustreParam(BaseModel):


    def validate(self):
        raise NotImplementedError("validate method not implemented")


    @staticmethod
    def initialize(name, value):
        path = PARAM_DEFINITIONS[name]["path"]
        command = ['lctl', 'set_param', f'{path}={value}']
        try:
            clients = get_system_config()['Clients']
            for client in clients:
                command = ['ssh', client, 'lctl', 'set_param', f'{path}={value}']
                run_command(command)
                logger.info(f"Initialized {name} to {value} on {client}")
            for client in clients:
                check_command = ['ssh', client, 'lctl', 'get_param', f'{path}']
                output = run_command(check_command)
                logger.info(f"Output: {output}")

        except Exception as e:
            print(f"Error setting {name}: {e}")
    
    @staticmethod
    def describe(param_name):
        return {"definition": PARAM_DEFINITIONS[param_name]["description"]["definition"],
                "effect": PARAM_DEFINITIONS[param_name]["description"]["effect"],
                "additional_info": PARAM_DEFINITIONS[param_name]["description"]["additional_info"],
                "range_description": PARAM_DEFINITIONS[param_name]["description"]["range_description"],
                "range_value_type": PARAM_DEFINITIONS[param_name]["description"]["range_value_type"],
                "range": PARAM_DEFINITIONS[param_name]["description"]["range"]
                }
    
    @staticmethod
    def read(param_name):
        path = PARAM_DEFINITIONS[param_name]["path"]
        command = ['lctl', 'get_param', path]
        try:
            logger.info(f"Running command: {command}")
            output = run_command(command)
            logger.info(f"Output: {output}")
            return output
        except Exception as e:
            print(f"Error reading {param_name}: {e}")

######################################################################
# Stripe
######################################################################


class StripeParams(BaseModel):
    path: str = Field(
        ...,
        description=""
    )
    stripe_count: int = Field(
        ..., 
        description=""
    )
    stripe_size: str = Field(
        ...,
        description=""
    )

    @staticmethod
    def describe():
        stellar_config = StellarConfig.get_instance().config
        mount_root = stellar_config['System']['mount_root']
        application = stellar_config['Application']
        if application in stellar_config['Application_Configs']:
            if 'data_dir' in stellar_config['Application_Configs'][application]:
                data_dir = stellar_config['Application_Configs'][application]['data_dir']
        else:
            data_dir = mount_root
        #data_dir = mount_root
        osts = len(stellar_config['System']['OSS']) * stellar_config['System']['OST_PER_OSS']
        return {
            "path": {
                "definition": "Path in the Lustre file system where stripe settings apply.",
                "effect": "The path determines the location of the data in the Lustre file system. Stripe settings apply to all files and directories within this path.",
                "additional_info": "Stripe settings are applied recursively to all subdirectories of the specified path.\n- The path must be a valid directory in the Lustre file system.",
                "type": "path",
                "range_description": "Valid path in mounted Lustre file system (range values indicate the root of the mounted Lustre file system)",
                "range": {
                "values": f"Any path in '{mount_root}/*'"
                }
            },
            "stripe_count": {
                "definition": "The stripe count parameter in Lustre specifies the number of Object Storage Targets (OSTs) across which a file will be striped.",
                "effect": "Setting the stripe count determines how data is distributed across OSTs, affecting parallel I/O performance. A higher stripe count can improve performance for large files by allowing concurrent access to multiple OSTs, but may increase overhead and risk of partial data loss if an OST fails. A stripe count of -1 stripes the file across all available OSTs, while a value of 0 uses the system default of 1.",
                "additional_info": "",
                "type": "integer",
                "range_description": "The stripe count can be set to a specific number of OSTs, -1 for all available OSTs, or 0 to use the system default.",
                "range": {
                "min": -1,
                "max": osts
                }
            },
            "stripe_size": {
                "definition": "The stripe size parameter in Lustre specifies the size of each stripe in a file that is distributed across multiple Object Storage Targets (OSTs).",
                "effect": "The stripe size determines the amount of data written to each OST before moving to the next. A larger stripe size can improve performance for large, sequential I/O operations by reducing the frequency of switching between OSTs. However, it may also increase lock hold times and contention during shared file access. A smaller stripe size may lead to inefficient I/O and reduced performance due to increased overhead from more frequent OST switching.",
                "additional_info": "The stripe size must be an even multiple of the system page size, as shown by 'getpagesize()'. The default Lustre stripe size is 4MB. Choosing a stripe size is a balancing act; it should match the I/O patterns of your application. For high-speed networks, a stripe size between 1 MB and 4 MB is recommended. The maximum stripe size is 4 GB.",
                "type": "size",
                "range_description": "The stripe size must be a multiple of the page size, with a minimum recommended size of 512 K and a maximum of 4 G.",
                "range": {
                "min": "64K",
                "max": "4G"
                }
            }
        }
    
    def validate(self):
        # validate the values of the parameters
        # check if the path is valid
        system_config = get_system_config()
        if not validate_path(system_config['mount_root'], self.path):
            raise ValueError(f"Invalid path: Expected path in mounted Lustre file system but got {self.path}")
        # check if the stripe count is valid
        if not validate_integer(-1, len(system_config['OSS']) * system_config['OST_PER_OSS'], self.stripe_count):
            raise ValueError(f"Invalid stripe_count: Expected integer between -1 and {len(system_config['OSS']) * system_config['OST_PER_OSS']} but got {self.stripe_count}")
        # check if the stripe size is valid
        if not validate_size("64KB", "4GB", self.stripe_size):
            raise ValueError(f"Invalid stripe_size: Expected size between 64KB and 4GB but got {self.stripe_size}")
        return True
    
    def initialize(self):
        logger.info(f"Initializing Lustre parameters: {self.path}, {self.stripe_count}, {self.stripe_size}")
        # check if the the directory exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        command = ['lfs', 'setstripe', '-c', str(self.stripe_count), '-S', str(self.stripe_size), self.path]
        try:
            logger.info(f"Running command: {command}")
            run_command(command)
            command = ['lfs', 'getstripe', self.path]
            logger.info(f"Running command: {command}")
            output = run_command(command)
            logger.info(f"Output: {output}")
            logger.info("Lustre parameters initialized successfully")
        except Exception as e:
            print(f"Error setting stripe parameters: {e}")

    @classmethod
    def read(cls, path: str=None):
        if not path:
            mount_root = get_system_config()['mount_root']
            command = ['lfs', 'getstripe', mount_root]
        else:
            command = ['lfs', 'getstripe', path]
        try:
            logger.info(f"Running command: {command}")
            output = run_command(command)
            logger.info(f"Output: {output}")
            
            stripe_params = []
            lines = output.splitlines()
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Check if line is a path
                if line.startswith('/'):
                    current_path = line
                    
                    # Move to next line which should contain stripe info
                    i += 1
                    if i < len(lines):
                        stripe_info = lines[i].strip()
                        # Combine with next line if it exists and doesn't start with '/'
                        if i + 1 < len(lines) and not lines[i + 1].startswith('/'):
                            stripe_info += ' ' + lines[i + 1].strip()
                            i += 1
                        
                        # Extract stripe parameters
                        stripe_count_match = re.search(r'stripe_count:\s*(-?\d+)', stripe_info)
                        stripe_size_match = re.search(r'stripe_size:\s*(\d+)', stripe_info)
                        
                        if stripe_count_match and stripe_size_match:
                            stripe_count = int(stripe_count_match.group(1))
                            stripe_size = int(stripe_size_match.group(1))

                            stripe_params.append(
                                cls(
                                    path=current_path,
                                    stripe_count=stripe_count,
                                    stripe_size=str(stripe_size),
                                    stripe_index=0  # Default to 0 since it's not in output
                                )
                            )
                        else:
                            logger.warning(f"Could not parse stripe info from: {stripe_info}")
                i += 1
            
            return stripe_params
        except Exception as e:
            logger.error(f"Error reading stripe parameters: {e}")
            return []


######################################################################
# Config Templates
######################################################################

def parse_expression(value: str) -> Union[int, str, DependentRangeValue]:
    if "expression" in value:
        expression = re.search(r'expression\((.*)\)', value).group(1)
        #evaluate the expression
        res = eval(expression)
        if type(res) == float:
            res = int(res)
        return res
    return value


def evaluate_system_vars()-> dict:
    config = get_system_config()
    variables = {}
    variables["CLIENT_COUNT"] = len(config["Clients"])
    variables["OST_COUNT"] = len(config["OSS"]) * config["OST_PER_OSS"]
    variables["MDS_COUNT"] = len(config["MDS"])
    variables["CLIENT_RAM"] = config["CLIENT_RAM"]
    variables["OST_PER_OSS"] = config["OST_PER_OSS"]
    return variables


def parse_range_to_dependents(value: Union[int, str, None]) -> Union[int, str, DependentRangeValue]:
    """Parse a range value to determine if it's dependent on other parameters"""
    if isinstance(value, (int, type(None))):
        return value
    
    
    variables = evaluate_system_vars()
    for variable in variables:
        if variable in value:
            #replace the variable with the value
            value = value.replace(variable, str(variables[variable]))

    # re to find content of dependent(...)
    if "dependent" in value:
        dependent_param = re.search(r'dependent\((.*)\)', value)
        if dependent_param:
            dependent_param = dependent_param.group(1)
            return DependentRangeValue(dependent_param=dependent_param, expression=value)
    
    value = parse_expression(value)
    
    return value

def get_dependent_param_value(param_name: str, template_params_list: list[str] = None) -> Optional[Union[int, str]]:
    """Get the value of a dependent parameter"""
    try:
        # Try to read the parameter value using the appropriate class
        if param_name in PARAM_CLASSES and not param_name in template_params_list:
            return PARAM_CLASSES[param_name].read()
        elif param_name in PARAM_CLASSES and param_name in template_params_list:
            return param_name
        return None
    except Exception as e:
        logger.warning(f"Failed to get dependent parameter {param_name}: {e}")
        return None



def parse_range_value(value: Union[int, str, None], dependent_value: Union[int, str, None] = None, template_params_list: list[str] = None) -> Union[int, str]:
    if isinstance(value, (int, type(None))):
        return value
    
    config = get_system_config()
    variables = config.keys()
    for variable in variables:
        if variable in value:
            value = value.replace(variable, str(config[variable]))
    
    if "dependent" in value:
        dependent_param = re.search(r'dependent\((.*)\)', value)
        if dependent_param:
            dependent_param = dependent_param.group(1)
            value = value.replace(f"dependent({dependent_param})", dependent_param)
            dependent_value = get_dependent_param_value(dependent_param, template_params_list)
            if dependent_value is None:
                raise ValueError(f"Unable to determine value: dependent parameter {dependent_param} not available")
        return dependent_value
    
    value = parse_expression(value)
    return value

def resolve_dependent_values(param_description, template_params_list: list[str] = None):
    range_info = param_description.get('range', {})
    for key, value in range_info.items():
        if isinstance(value, DependentRangeValue):
            dep_value = get_dependent_param_value(value.dependent_param, template_params_list)
            expression = value.expression
            if dep_value is None:
                raise ValueError(f"Unable to determine value: dependent parameter {value.dependent_param} not available")
            range_info[key] = parse_range_value(expression, dep_value, template_params_list)
    param_description['range'] = range_info
    return param_description

def resolve_dependent_single_value(value) -> Union[int, str]:
    if isinstance(value, DependentRangeValue):
        dep_value = get_dependent_param_value(value.dependent_param)
        expression = value.expression
        if dep_value is None:
            raise ValueError(f"Unable to determine value: dependent parameter {value.dependent_param} not available")
        return parse_range_value(expression, dep_value)
    return value



def create_lustre_param_class(param_name: str, param_description: Dict[str, Any], mock: bool = False) -> Type[LustreParam]:
    """
    Dynamically creates a LustreParam class based on the parameter configuration.
    """
    
    param_type = param_description['description']['range_value_type']
    param_type = param_type.lower()
    if param_type == "integer":
        param_type = "int"
    elif param_type == "string":
        param_type = "str"

    type_annotation = {
        'int': int,
        'str': str
    }.get(param_type, str)

    if not type_annotation:
        raise ValueError(f"Invalid parameter type: {param_type}")
    
    range_info = param_description['description'].get('range', {})
    # Parse range values
    if range_info and "min" in range_info and "max" in range_info:
        if mock:
            range_min = range_info.get('min')
            range_max = range_info.get('max')
        else:
            range_min = parse_range_to_dependents(range_info.get('min'))
            range_max = parse_range_to_dependents(range_info.get('max'))
        param_description['description']['range'] = {
            "min": range_min,
            "max": range_max
        }
    elif range_info and "string_values" in range_info:
        string_values = range_info.get('string_values', [])
        if string_values:
            for i, value in enumerate(string_values):
                if mock:
                    string_values[i] = value
                else:
                    string_values[i] = parse_range_to_dependents(value)
        else:
            return None
        param_description['description']['range'] = {
            "string_values": string_values
        }
    elif range_info and "int_values" in range_info:
        int_values = range_info.get('int_values', [])
        if int_values:
            for i, value in enumerate(int_values):
                if mock:
                    int_values[i] = value
                else:
                    int_values[i] = parse_range_to_dependents(value)
            param_description['description']['range'] = {
                "int_values": int_values
            }
        else:
            return None
    else:
        raise ValueError(f"Invalid range: {range_info}")
    
    
    # Create Field with description
    field_description = param_description['description']['definition']
    field_range_desc = param_description['description']['range_description']
    field_range_values = param_description['description']['range']
    field_description = f"{field_description}\n\nRange Description: {field_range_desc}\n\nRange Values: {field_range_values}"
    
    field = Field(
        ...,  # This means the field is required
        description=field_description,
        title=param_name.replace('_', ' ').title()
    )

    def create_validate_method(param_name):
        def validate(self):
            value = getattr(self, param_name)
            param_desc = self._description
            range_info = param_desc['description']['range']
            logger.info(f"range_info: {range_info}")
            logger.info(f"value: {value}")
            if "min" in range_info and "max" in range_info:
                range_min = resolve_dependent_single_value(range_info["min"])
                range_max = resolve_dependent_single_value(range_info["max"])
                logger.info(f"range_min: {range_min}")
                logger.info(f"range_max: {range_max}")
                logger.info(f"value: {value}")
                #if not validate_integer(range_min, range_max, value):
                #    raise ValueError(f"Invalid {param_name}: Expected integer between {range_min} and {range_max} but got {value}")
            elif param_type in ('str'):
                if value not in string_values:
                    raise ValueError(f"Invalid {param_name}: Expected one of {string_values} but got {value}")
                    
            return True
        
        return validate
    
    def create_describe_method():
        @staticmethod
        def describe(template_params_list: list[str] = None):
            unparsed_description = LustreParam.describe(param_name)
            if mock:
                return unparsed_description
            else:
                return resolve_dependent_values(unparsed_description, template_params_list)
        return describe
    
    def create_initialize_method():
        def initialize(self):
            return LustreParam.initialize(param_name, getattr(self, param_name))
        return initialize
    
    def create_read_method():
        @staticmethod
        def read():
            output = LustreParam.read(param_name)
            if output:
                try:
                    if type_annotation == int:
                        return int(re.search(fr'{param_name}=\s*(\d+)', output).group(1))
                    elif type_annotation == str:
                        return re.search(fr'{param_name}=\s*(\S+)', output).group(1)
                    else:
                        raise ValueError(f"Invalid parameter type: {type_annotation}")
                except ValueError as e:
                    raise ValueError(f"Failed to read {param_name}: {e}") from e
        return read
    

    attributes = {
        '__module__': __name__,
        '__annotations__': {param_name: type_annotation},
        param_name: field,  # Add the Field to attributes
        '_description': param_description,
        'validate': create_validate_method(param_name),
        'describe': create_describe_method(),
        'initialize': create_initialize_method(),
        'read': create_read_method()
    }

    # Change this line to inherit from LustreParam instead of BaseModel
    return type(
        param_name.title().replace('_', '') + 'Param',
        (LustreParam,),  # Changed from BaseModel to LustreParam
        attributes
    )


PARAM_CLASSES = {"plan_description": PlanDescription, "min_stripe": list[StripeParams]}
for param_name, param_config in PARAM_DEFINITIONS.items():
    param_class = create_lustre_param_class(param_name, param_config)
    if param_class:
        PARAM_CLASSES[param_name] = param_class



CONFIG_TEMPLATES = {
    "min": ["min_stripe", "max_rpcs_in_flight", "max_dirty_mb", "max_read_ahead_mb"],
    "o1-selected": ["min_stripe", "max_rpcs_in_flight", "max_mod_rpcs_in_flight", "threads_min", "threads_max", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "max_read_ahead_async_active", "checksum_pages", "lru_size", "statahead_running_max", "statahead_max"],
    "o1-selected-no-stripe": ["max_rpcs_in_flight", "max_mod_rpcs_in_flight", "threads_min", "threads_max", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "max_read_ahead_async_active", "checksum_pages", "lru_size", "statahead_running_max", "statahead_max"],
    "selected": ["max_rpcs_in_flight", "max_mod_rpcs_in_flight", "threads_min", "threads_max", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "max_read_ahead_async_active", "checksum_pages", "lru_size", "statahead_running_max", "statahead_max", "max_pages_per_rpc"],
    "selected-ws": ["min_stripe", "max_rpcs_in_flight", "max_mod_rpcs_in_flight", "threads_min", "threads_max", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "max_read_ahead_async_active", "checksum_pages", "lru_size", "statahead_running_max", "statahead_max", "max_pages_per_rpc"],
    "filtered": ["min_stripe", "max_dirty_mb", "timeout", "max_rpcs_in_flight", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "statahead_agl", "statahead_max","max_mod_rpcs_in_flight", "max_pages_per_rpc"],
    "filtered-wc": ["min_stripe", "max_dirty_mb", "timeout", "max_rpcs_in_flight", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "statahead_agl", "statahead_max","max_mod_rpcs_in_flight", "max_pages_per_rpc", "checksum_pages"],
    "filtered-nb": ["min_stripe", "max_read_ahead_whole_mb", "max_read_ahead_per_file_mb", "max_read_ahead_mb", "statahead_max", "mdc-max_rpcs_in_flight", "osc-max_rpcs_in_flight", "max_mod_rpcs_in_flight", "mdc-max_dirty_mb", "osc-max_dirty_mb", "osc-max_pages_per_rpc", "mdc-max_pages_per_rpc"]
}



def create_config_from_template(template: Union[str, list[str]]):  
    global CONFIG_TEMPLATES
    if isinstance(template, str):
        if template not in CONFIG_TEMPLATES:
            raise ValueError(f"Invalid template: {template}")
        params = CONFIG_TEMPLATES[template]
        template_name = template
    elif isinstance(template, list):
        for param in template:
            if param not in PARAM_CLASSES and param not in PARAM_DEFINITIONS:
                raise ValueError(f"Invalid parameter: {param}")
            elif param in ["stripetype", "stripesize", "stripecount"]:
                raise ValueError(f"Invalid parameter: {param}")
        params = template
        template_name = "custom"
    

    @classmethod
    def _model_json_schema(cls, *args, **kwargs):
        # Get the original schema
        schema = cls.model_json_schema(*args, **kwargs)
        param_descriptions = cls.describe()
        
        # replace $defs with the param_descriptions
        schema["$defs"] = param_descriptions
        
        return schema

    @staticmethod
    def describe():
        desc = {}
        for param in params:
            #logger.info(f"param: {param}")
            param_class = PARAM_CLASSES[param]
            #logger.info(f"param_class: {param_class}")
            # Check if field type is a list using get_origin
            if hasattr(param_class, "__origin__") and param_class.__origin__ is list:
                #logger.info(f"param_class is a list")
                # Get the type of elements in the list using __args__[0]
                element_type = param_class.__args__[0]
                #logger.info(f"element_type: {element_type}")
                if "stripe" in param:
                    desc[f"stripe_settings (you can specify multiple of these for different paths)"] = [element_type.describe()]
                else:
                    desc[f"{param} (you can specify multiple of these)"] = [element_type.describe()]
            else:
                desc[param] = param_class.describe(params)
                #logger.info(f"desc: {desc}")
        return desc
    
    @staticmethod
    def read():
        # Create a dictionary to store the read values
        values = {}
        for param in params:
            param_class = PARAM_CLASSES[param]
            if "stripe" in param:
                param = "stripe_settings"              

            #logger.info(f"param_class: {param_class}")
            # Check if field type is a list
            if hasattr(param_class, "__origin__") and param_class.__origin__ is list:
                #logger.info(f"param_class is a list")
                element_type = param_class.__args__[0]
                #logger.info(f"element_type: {element_type}")
                raw_values = element_type.read()
                #logger.info(f"raw_values: {raw_values}")
                values[param] = raw_values
            else:
                raw_value = param_class.read()
                #logger.info(f"raw_value: {raw_value}")
                values[param] = param_class(**{param: raw_value})
        
        try:
            # Create an instance of ConfigModel with the read values
            return ConfigModel(**values)
    
        except Exception as e:
            logger.error(f"Failed to create config instance from read values: {e}")
            logger.info(f"Attempted values: {values}")
            raise ValueError(f"Failed to create config from read values: {e}") from e
    
    try:
        # Create the model with all arguments as keyword arguments
        kwargs = {param: (PARAM_CLASSES[param], ...) for param in params}
        #logger.info(f"kwargs: {kwargs}")
        for param in params:
            if "stripe" in param:
                value = kwargs[param]
                # remove the param from the kwargs
                del kwargs[param]
                # add "stripe_settings" to the kwargs
                kwargs["stripe_settings"] = value
                
        #logger.info(f"kwargs after: {kwargs}")
        ConfigModel = create_model(
            f"{template_name}LustreParams",
            __base__=FSConfig,
            describe=describe,
            read=read,
            _model_json_schema=_model_json_schema,  # Add the new method
            **kwargs
        )
        return ConfigModel
    except Exception as e:
        raise ValueError(f"Failed to create config model: {e}") from e



def get_lustre_config():
    # Get the complexity level from the stellar configuration
    stellar_config = StellarConfig.get_instance()
    config_template = stellar_config.config.get('template')
    return create_config_from_template(config_template)




