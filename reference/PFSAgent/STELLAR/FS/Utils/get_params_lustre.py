from STELLAR.Utils.logger import LOGGERS
import os
import json
import stat


logger = LOGGERS["fs_config"]

LUSTRE_PARAMS_BASE_PATH = ["/sys/fs/lustre", "/proc/fs/lustre", "/sys/kernel/debug/lustre"]
LUSTRE_PARAMS_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Docs", "new_lustre_params.json")



def get_tunable_params():
    tunable_params = {}
    unique_file_paths = set()
    for param_source_dir in LUSTRE_PARAMS_BASE_PATH:
        for root, dirs, files in os.walk(param_source_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                unique_file_paths.add(filepath)

                
                # Get file mode
                try:
                    file_mode = os.stat(filepath).st_mode
                except OSError:
                    # If we can't stat the file, skip it
                    continue
                
                # Check readability and writability by owner.
                # Typically, if you run this as root, owner permissions apply.
                # If the file allows reading and writing at the user/owner level, we consider it tunable.
                is_readable = bool(file_mode & stat.S_IRUSR)
                is_writable = bool(file_mode & stat.S_IWUSR)
                
                if is_readable and is_writable:
                    # Build the relative path parts to determine module/instance/param
                    rel_path = os.path.relpath(filepath, param_source_dir)
                    parts = rel_path.split(os.sep)
                    
                    # The parameter is always the last part
                    parameter = parts[-1]
                    
                    if len(parts) == 1:
                        # Only parameter: directly under /sys/fs/lustre
                        # Just use the parameter name
                        param_path = parameter
                    elif len(parts) == 2:
                        # Two parts: module/parameter with no instance directories
                        module, parameter = parts
                        param_path = f"{module}.{parameter}"
                    else:
                        # More than two parts: module, instances, and parameter
                        # Replace all instances with '*'
                        module = parts[0]
                        parameter = parts[-1]
                        num_instances = len(parts) - 2
                        param_path = f"{module}.{'*.' * num_instances}{parameter}"
                        path_depth = len(param_path.split("."))
                    
                    if parameter not in tunable_params:
                        tunable_params[parameter] = {}
                        tunable_params[parameter]["path"] = param_path
                    else:
                        existing_path = tunable_params[parameter]["path"]
                        existing_path_depth = len(existing_path.split("."))
                        if param_path == existing_path:
                            continue
                        elif path_depth > existing_path_depth:
                            tunable_params[parameter]["path"] = param_path
                        elif path_depth == existing_path_depth:
                            existing_root_type = existing_path.split(".")[0]
                            new_root_type = param_path.split(".")[0]
                            # remove the parameter from tunable_params
                            del tunable_params[parameter]
                            tunable_params[f"{new_root_type}-{parameter}"] = {}
                            tunable_params[f"{new_root_type}-{parameter}"]["path"] = param_path
                            tunable_params[f"{existing_root_type}-{parameter}"] = {}
                            tunable_params[f"{existing_root_type}-{parameter}"]["path"] = existing_path


    logger.info(f"Found {len(tunable_params)} tunable parameters")
    logger.info(f"Saving to {LUSTRE_PARAMS_OUTPUT_FILE}")
    logger.info(f"Found {len(unique_file_paths)} unique file paths")
    with open(LUSTRE_PARAMS_OUTPUT_FILE, 'w') as f:
        json.dump(tunable_params, f, indent=4)

    return tunable_params

