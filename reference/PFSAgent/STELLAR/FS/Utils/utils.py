import subprocess

def run_command(command):
    try:
        command_str = " ".join(command)
        result = subprocess.run(command_str, capture_output=True, text=True, shell=True)
        return result.stdout.strip()
    except Exception as e:
        raise e



def validate_path(allowed_path_root, path):
    # check if the path is valid
    if not path.startswith(allowed_path_root):
        return False
    return True


def convert_to_bytes(size_str):
    # convert the size to bytes
    units = {'KB': 1024, 'K': 1024, 'MB': 1024**2, 'M': 1024**2, 'GB': 1024**3, 'G': 1024**3}
    for unit in units:
        if unit in size_str:
            return int(size_str.replace(unit, '')) * units[unit]
    if "B" in size_str:
        return int(size_str.replace("B", '')) 
    return int(size_str)


def validate_size(min_size, max_size, size):
    # validate the size
    # convert the size to bytes and compare
    size_in_bytes = convert_to_bytes(size)
    min_size_in_bytes = convert_to_bytes(min_size)
    max_size_in_bytes = convert_to_bytes(max_size)
    if size_in_bytes < min_size_in_bytes or size_in_bytes > max_size_in_bytes:
        return False
    return True


def validate_integer(min_value, max_value, value):
    # validate the integer
    if value < min_value or value > max_value:
        return False
    return True


def validate_boolean(value):
    # validate the boolean
    if value not in [0, 1]:
        return False
    return True


def parse_expression(expression):
    # parse the expression
    pass

def parse_dependency(dependency):
    # parse the dependency
    pass

def parse_range(range_dict):
    pass
