import pandas as pd
from tqdm import tqdm
import re
from STELLAR.Utils.logger import LOGGERS
import os
import subprocess
from datetime import datetime
from typing import Literal

logger = LOGGERS["darshan_utils"]


def parse_agg_settings(files_list: list[str], path_to_log_dir: str, aggregation_settings: None | Literal["all"] | datetime) -> list[str]:
    logger.info(f"Parsing aggregation settings: {aggregation_settings}")
    if not aggregation_settings:
        most_recent_log = max(files_list, key=lambda x: os.path.getmtime(os.path.join(path_to_log_dir, x)))
        return [os.path.join(path_to_log_dir, most_recent_log)]
    elif aggregation_settings == "all":
        return [os.path.join(path_to_log_dir, file) for file in files_list]
    elif isinstance(aggregation_settings, datetime):
        # Convert datetime to timestamp for comparison
        timestamp_cutoff = aggregation_settings.timestamp()
        # get all logs since the given date
        logger.info(f"Timestamp cutoff: {timestamp_cutoff}")
        logger.info(f"Path to log dir: {path_to_log_dir}")
        for file in files_list:
            logger.info(f"File: {file} - Timestamp: {os.path.getmtime(os.path.join(path_to_log_dir, file))}")
        return [os.path.join(path_to_log_dir, file) for file in files_list if os.path.getmtime(os.path.join(path_to_log_dir, file)) >= timestamp_cutoff]
    else:
        raise ValueError(f"Invalid aggregation settings: {aggregation_settings}")
    

def parse_darshan_log(log_dir, date_formatted_dir=False, aggregation_settings: None | Literal["all"] | datetime = None):
    # logs will be in the format of <log_dir>/year/month/day
    if date_formatted_dir:
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        path_to_log_dir = os.path.join(log_dir, str(year), str(month), str(day))
    else:
        path_to_log_dir = log_dir
    logs = os.listdir(path_to_log_dir)
    log_files = parse_agg_settings(logs, path_to_log_dir, aggregation_settings)
    
    log_contents = []
    new_file_names = []
    for log_file in log_files:
        if log_file.endswith(".darshan"):
            new_file_name = log_file.replace(".darshan", ".txt")
            logger.info(f"Parsing darshan log file: {log_file}")
            # create txt file with log contents
            command = f"darshan-parser --show-incomplete {log_file} > {new_file_name}"
            subprocess.run(command, shell=True)
        else:
            new_file_name = log_file
        # read txt file
        with open(new_file_name, "r") as file:
            log_contents.append(file.read())
        new_file_names.append(new_file_name)
        
    return log_contents, new_file_names

skip_lines = [
    "# *WARNING*: The POSIX module contains incomplete data!",
    "#            This happens when a module runs out of",
    "#            memory to store new record data.",
    "# To avoid this error, consult the darshan-runtime",
    "# documentation and consider setting the",
    "# DARSHAN_EXCLUDE_DIRS environment variable to prevent",
    "# Darshan from instrumenting unecessary files.",
]

def extract_modules(log_data):
    modules = {}
    module = None
    in_module = False
    current_description = []
    
    for line in tqdm(log_data.splitlines()):
        if line in skip_lines:
            continue
        # If we find a module header, start collecting description
        if "module data" in line:
            current_description = []
            continue
            
        # If we hit the column definitions, we're done with description
        elif line.startswith("#<module>"):
            column_names = re.findall(r'<(.*?)>', line)
            # if column names have a space, remove it
            column_names = [name.replace(" ", "_") for name in column_names]
            in_module = True
            continue
        # Collect all comment lines as description
        elif line.startswith("#") and not in_module:
            current_description.append(line)
            
        elif in_module:
            if not line.strip():  # Empty line signifies end of module data
                in_module = False
                module = None
            else:
                fields = line.split()
                if not module:
                    module = fields[0]  # First field is the module name
                    modules[module] = {
                        'columns': column_names,
                        'data': [],
                        'description': '\n'.join(current_description)
                    }
                    modules[module]['data'].append(fields)
                else:
                    modules[module]['data'].append(fields)
    
    return modules


def extract_header(log_data):
    # header is every line that is before "# description of columns:"
    # save the lines as text
    header_text = ""
    for line in tqdm(log_data.splitlines()):
        if "log file regions" in line:
            break
        else:
            header_text += line + "\n"
    return header_text


def merge_headers(headers: list[str]) -> str:
    if len(headers) == 1:
        return headers[0]
    
    else:
        headers_str = ""
        headers_idx = 1
        for header in headers:
            headers_str += f"Header for Darshan Log File {headers_idx}:\n{header}\n\n"
            headers_idx += 1
        return headers_str
    

def merge_module_dfs(module_dfs: dict) -> dict:
    new_module_dfs = {}
    for module in module_dfs:
        new_module_dfs[module] = {"description": module_dfs[module]["description"], "dataframes": []}
        if len(module_dfs[module]["dataframes"]) > 1:
            new_module_dfs[module]["dataframe"] = pd.concat(module_dfs[module]["dataframes"])
        else:
            new_module_dfs[module]["dataframe"] = module_dfs[module]["dataframes"][0]
    return new_module_dfs

def parse_darshan_to_csv(darshan_content: list[str], output_dir: str, save_to_file: bool = True):

    # Load the darshan log file
    logger.info("Loading darshan log file")
    if not darshan_content or len(darshan_content) == 0:
        logger.error("No data found in the log file")
        return
    else:
        logger.info("Data loaded successfully")
    
    # Extracting the modules
    logger.info("Parsing darshan log file")
    modules_per_log = []
    headers = []
    try:
        for content in darshan_content:
            modules_per_log.append(extract_modules(content))
            headers.append(extract_header(content))
        header = merge_headers(headers)
    except Exception as e:
        logger.error(f"Error parsing log file: {e}")
        return
    logger.info("Parsing complete")
    if save_to_file:
        with open(f"{output_dir}/header.txt", "w") as f:
            f.write(header)
    # Create counter table for each module
    module_dfs = {}

    for log_modules in modules_per_log:
        for module in log_modules:
            logger.info(f"Found module: {module}")
            description = log_modules[module]["description"]
            columns = log_modules[module]["columns"]
            rows = log_modules[module]["data"]
            df = pd.DataFrame(rows, columns=columns)
            index_columns = [column for column in columns if column not in ['counter', 'value']]

            df = df.pivot_table(index=index_columns, columns='counter', values='value', aggfunc='first').reset_index()
            # save the dataframe to a csv file named after the module
            if module not in module_dfs:
                module_dfs[module] = {
                    "description": description,
                    "dataframes": [df]
                }
            else:
                module_dfs[module]["dataframes"].append(df)
    module_dfs = merge_module_dfs(module_dfs)

    if not save_to_file:
        return module_dfs, header
    else:
        for module in module_dfs:
            description = module_dfs[module]["description"]
            df = module_dfs[module]["dataframe"]
            logger.info("Saving data to csv file")
            output_data_file_path = f"{output_dir}/{module}.csv"
            output_description_file_path = f"{output_dir}/{module}_description.txt"
            with open(output_description_file_path, "w") as f:
                f.write(description)
            df.to_csv(output_data_file_path, index=False)
            logger.info(f"{module} data saved to {output_data_file_path}")
    
    


    




    
