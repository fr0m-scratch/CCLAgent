import logging
import os
from .stellar_config import StellarConfig
import json
import copy


def parse_level(level):
    if level == "INFO":
        return logging.INFO
    elif level == "DEBUG":
        return logging.DEBUG
    elif level == "WARNING":
        return logging.WARNING
    elif level == "ERROR":
        return logging.ERROR
    else:
        raise ValueError(f"Invalid log level: {level}")


def setup_logger(name):
    agent_config = StellarConfig.get_instance()
    log_dir = agent_config.config['Logging']['log_dir']
    log_level = agent_config.config['Logging']['log_level']
    log_level = parse_level(log_level)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatters
    #file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    file_formatter = logging.Formatter('%(module)s - %(funcName)s - %(message)s')
    if log_dir:
        # File Handler setup
        log_file_path = os.path.join(log_dir, f"Stellar-{name}.log")
        if os.path.exists(log_file_path):
            # clear the file
            with open(log_file_path, 'w') as f:
                f.truncate(0)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # File Handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        # Stream Handler (stdout)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    return logger

class StellarLogger:
    def __init__(self, name):
        self.logger = setup_logger(name)
        self.message_idx = 0

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        print(message)
        self.logger.error(message)

    def parse_message(self, message):
        # Don't modify the original message - create a deep copy to work with
        message_copy = copy.deepcopy(message)
        self.message_idx += 1
        formatted_output = [
            f"=== MESSAGE #{self.message_idx} ===",
            f"Role: {message_copy.get('role', 'unknown')}",
            f"Content: {message_copy.get('content', '')}",
            ""
        ]
        for key, value in message_copy.items():
            if key != "role" and key != "content":
                # For complex types (dicts, lists), use JSON formatting with indentation
                if isinstance(value, (dict, list)):
                    # Special handling for tool_calls to parse the arguments string as JSON
                    if key == "tool_calls" and isinstance(value, list):
                        formatted_output.append(f"{key}:")
                        processed_value = []
                        for tool_call in value:
                            # Create a deep copy of the tool call that we can modify
                            processed_tool_call = dict(tool_call)
                            # Check for function.arguments and parse it if it's a JSON string
                            if "function" in processed_tool_call and isinstance(processed_tool_call["function"], dict):
                                if "arguments" in processed_tool_call["function"] and isinstance(processed_tool_call["function"]["arguments"], str):
                                    try:
                                        # Try to parse arguments as JSON for LOGGING ONLY
                                        # Don't modify the structure that's passed back
                                        args_json = json.loads(processed_tool_call["function"]["arguments"])
                                        # Replace the string with the parsed JSON object - for DISPLAY only
                                        processed_tool_call["function"]["arguments"] = args_json
                                    except json.JSONDecodeError:
                                        # If it's not valid JSON, keep it as is
                                        pass
                            processed_value.append(processed_tool_call)
                        # Format the processed tool_calls
                        formatted_json = json.dumps(processed_value, indent=2)
                        for line in formatted_json.split('\n'):
                            formatted_output.append(f"  {line}")
                    else:
                        # Normal JSON formatting for other complex structures
                        formatted_output.append(f"{key}:")
                        formatted_json = json.dumps(value, indent=2)
                        for line in formatted_json.split('\n'):
                            formatted_output.append(f"  {line}")
                else:
                    formatted_output.append(f"{key}: {value}")
        return formatted_output
    
    def log_message(self, message):
        formatted_output = self.parse_message(message)
        self.logger.info("\n".join(formatted_output))

    def log_messages_json(self, messages):
        """
        Logs each message in the messages JSON array using the provided logger.
        Each message contains a 'role' and 'content' field.
        """
        formatted_output = ["=== MESSAGES ==="]
        
        for message in messages:
            formatted_output.extend(self.parse_message(message))
            formatted_output.append("")  # blank line after each message
        
        # Join all lines with newlines and log as a single message
        self.logger.info("\n".join(formatted_output))

    def log_tuning_knowledge(self, tuning_knowledge, step_idx):
        """
        Logs tuning knowledge as a single formatted message.
        """
        formatted_output = [
            f"=== TUNING KNOWLEDGE at step {step_idx} ===",
            tuning_knowledge.model_dump_json(indent=4),
            ""  # blank line at end
        ]
        self.logger.info("\n".join(formatted_output))

    def log_runtime_description(self, runtime_description, step_idx):
        """
        Logs a neatly formatted runtime description with the sections:
          - Summary
          - Code
          - Problems

        :param runtime_description: A dictionary containing keys: "summary", "code", "problems".
        """
        # Extract sections
        if type(runtime_description) == str:
            summary = runtime_description
            code = ""
            problems = ""
        else:
            summary = runtime_description.summary
            code = runtime_description.code
            problems = runtime_description.problems

        # Format each section's content
        summary_lines = [line for line in summary.splitlines()]
        code_lines = [line for line in code.splitlines()]
        problem_lines = [line for line in problems.splitlines()]

        formatted_output = [
            f"=== RUNTIME DESCRIPTION at step {step_idx} ===",
            "",
            "=== SUMMARY ===",
            *summary_lines,  # Unpack the lines
            "",  # Blank line
            "=== CODE ===",
            *code_lines,  # Unpack the lines
            "",  # Blank line
            "=== PROBLEMS ===",
            *problem_lines,  # Unpack the lines
            ""  # Extra blank line at end
        ]

        self.logger.info("\n".join(formatted_output))

    def log_benchmark_comparison_tables(self, candidate_db):
        """
        Creates two tables for all benchmark_results:
        1) A table comparing the 'value' across each benchmark result (columns).
        2) A table comparing the 'time' across each benchmark result (columns).
        """
        candidates = candidate_db.sort_candidates_by_id()
        benchmark_results = [c.complete_results for c in candidates]
        
        formatted_output = []
        
        if not benchmark_results:
            formatted_output.append("No benchmark results to display.")
            self.logger.info("\n".join(formatted_output))
            return

        # 1) Collect all unique phase names
        all_phases = set()
        for result in benchmark_results:
            phases = result.get("phases", {})
            for phase_name in phases.keys():
                all_phases.add(phase_name)

        all_phases = sorted(all_phases)
        num_benchmarks = len(benchmark_results)
        
        # Prepare headers
        value_headers = [f"#Val{i+1}" for i in range(num_benchmarks)]
        time_headers = [f"#Time{i+1}" for i in range(num_benchmarks)]

        # Value table
        formatted_output.append("=== VALUE COMPARISON TABLE ===")
        header_value_line = f"{'PHASE':<25}"
        for header in value_headers:
            header_value_line += f" {header:>12}"
        formatted_output.append(header_value_line)
        formatted_output.append("-" * len(header_value_line))

        for phase_name in all_phases:
            row_str = f"{phase_name:<25}"
            for result in benchmark_results:
                phase_data = result.get("phases", {}).get(phase_name, {})
                if not phase_data:
                    row_str += f" {'N/A':>12}"
                else:
                    val = phase_data.get("value", "N/A")
                    if isinstance(val, (int, float)):
                        row_str += f" {val:>12.4f}"
                    else:
                        row_str += f" {'N/A':>12}"
            formatted_output.append(row_str)

        formatted_output.append("")  # blank line between tables

        # Time table
        formatted_output.append("=== TIME COMPARISON TABLE ===")
        header_time_line = f"{'PHASE':<25}"
        for header in time_headers:
            header_time_line += f" {header:>12}"
        formatted_output.append(header_time_line)
        formatted_output.append("-" * len(header_time_line))

        for phase_name in all_phases:
            row_str = f"{phase_name:<25}"
            for result in benchmark_results:
                phase_data = result.get("phases", {}).get(phase_name, {})
                if not phase_data:
                    row_str += f" {'N/A':>12}"
                else:
                    t = phase_data.get("time", "N/A")
                    if isinstance(t, (int, float)):
                        row_str += f" {t:>12.4f}"
                    else:
                        row_str += f" {'N/A':>12}"
            formatted_output.append(row_str)

        formatted_output.append("")  # extra blank line at end
        
        self.logger.info("\n".join(formatted_output))

    def log_final_candidates(self, candidate_db):
        candidates = candidate_db.dump_candidates()
        self.logger.info("=== FINAL CANDIDATES ===")
        for idx, candidate in enumerate(candidates):
            self.logger.info(f"Candidate {idx}: {candidate.tuning_config.model_dump_json(indent=4)}")
            self.logger.info(f"Score: {candidate.score}")
        self.logger.info("")

    def log_final_scores(self, candidate_db):
        scores = [candidate.score for candidate in candidate_db.candidates]
        self.logger.info("=== FINAL SCORES ===")
        for idx, score in enumerate(scores):
            self.logger.info(f"Candidate {idx}: {score}")
        self.logger.info("")


    def log_candidate_evolution(self, candidate_db):
        candidates = candidate_db.dump_candidates()
        self.logger.info("=== CANDIDATE EVOLUTION TABLE===")


LOGGERS = {
    "main": StellarLogger("main"),
    "application": StellarLogger("application"),
    "analysis_agent": StellarLogger("analysis_agent"),
    "candidate": StellarLogger("candidate"),
    "candidate_queue": StellarLogger("candidate_queue"),
    "fs_config": StellarLogger("fs_config"),
    "completion": StellarLogger("completion"),
    "tuning_knowledge": StellarLogger("tuning_knowledge"),
    "runtime_description": StellarLogger("runtime_description"),
    "darshan_utils": StellarLogger("darshan_utils"),
    "sync_change_monitor": StellarLogger("sync_change_monitor")
}

