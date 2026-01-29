from .candidate import Candidate
from .queue import get_candidate_queue
from STELLAR.Utils.logger import LOGGERS
from STELLAR.Applications import Application
import matplotlib.pyplot as plt
import time
import json
import statistics
from typing import Literal, Union
import os

logger = LOGGERS["candidate"]

class CandidateDB:
    def __init__(self, initial_candidate: Candidate=None, checkpoint_dir: str=None, log_dir: str=None):
        self.candidates = []
        self.failed_candidates = []
        self.candidate_queue = get_candidate_queue()
        self.log_dir = log_dir
        if checkpoint_dir:
            self.load(checkpoint_dir)
        else:
            self.tuning_config_class = type(initial_candidate.tuning_config)
            self.application = initial_candidate.application
            # add initial candidate to the database
            self.add_candidate(initial_candidate)


    def wait_for_candidate(self, candidate: Candidate):
        while not candidate.status:
            time.sleep(1)

    def add_candidate(self, candidate: Candidate):
        #existing_candidate = self.search_candidates(candidate)
        #if not existing_candidate:
        self.candidates.append(candidate)
        self.candidate_queue.add_candidate(candidate)
        self.wait_for_candidate(candidate)
        if candidate.status == "failed":
            self.failed_candidates.append(candidate)
        else:
            self.sort_candidates()
        return True
    
    def add_final_candidate(self, candidate: Candidate, application: Application, repititions: int = 1):
        final_candidate = Candidate(tuning_config=candidate.tuning_config, application=application, force_disable_darshan=True)
        for i in range(repititions):
            self.candidate_queue.add_candidate(final_candidate)
            self.wait_for_candidate(final_candidate)
            if final_candidate.status == "failed":
                raise ValueError("Candidate failed")
        
        
        
    def sort_candidates(self):
        # remove failed candidates from the list
        self.candidates = [c for c in self.candidates if c.status == "success"]
        # None values will be sorted to the end when reverse=True
        self.candidates.sort(key=lambda x: float(x.id.split('|')[0]))   

    def sort_candidates_by_score(self):
        candidates_copy = self.candidates.copy()
        if type(self.application.score_metric) == str:
            candidates_copy.sort(key=lambda x: float('-inf') if x.score is None else x.score, reverse=True)
        elif type(self.application.score_metric) == list:
            score_metric = ""
            for metric in self.application.score_metric:
                if "total" in metric:
                    score_metric = metric
                    break
                if "tp" in metric:
                    score_metric = metric
                    break
            candidates_copy.sort(key=lambda x: float('-inf') if x.score is None else x.score[score_metric], reverse=True)
        return candidates_copy

    def get_top_k_candidates(self, k: int):
        sorted_candidates = self.sort_candidates_by_score()
        return sorted_candidates[:min(k, len(self.candidates))]
    
    def search_candidates(self, candidate: Candidate):
        for c in self.candidates:
            if c.tuning_config == candidate.tuning_config:
                return c
        return None
    
    def get_best_candidate(self):
        sorted_candidates = self.sort_candidates_by_score()
        if not sorted_candidates:
            return None
        return sorted_candidates[0]
    
    def get_candidate_config_class(self):
        return self.tuning_config_class
    
    def get_candidate_application(self):
        return self.application
    
    def get_last_k_candidates(self, k: int):
        candidates_by_id = self.sort_candidates_by_id()
        return candidates_by_id[-k:]
    
    def sort_candidates_by_id(self):
        # id is timestamp|uuid, so we can sort by timestamp from oldest to newest
        candidates_copy = self.candidates.copy()
        candidates_copy.sort(key=lambda x: float(x.id.split('|')[0]))
        return candidates_copy
    
    def dump_candidates(self, k: int=None):
        if k:
            return [c for c in self.candidates[:k]]
        else:
            return self.candidates
    
    def to_json(self):
        return {
            "candidates": [c.to_json() for c in self.candidates],
            "failed_candidates": [c.to_json() for c in self.failed_candidates]
        }
    
    
    def summarize_scores(self, scores):
        summary_stats = {}
        metrics = list(scores[0].keys())
        for metric in metrics:
            summary_stats[metric] = {
                "mean": sum(s[metric] for s in scores) / len(scores),
                "min": min(s[metric] for s in scores),
                "max": max(s[metric] for s in scores)
            }
            if len(scores) > 2:
                summary_stats[metric]["stddev"] = statistics.stdev(s[metric] for s in scores)

        return summary_stats
                        

    def plot_candidates(self):
        scores = [c.score for c in self.candidates if c.score is not None]
        if len(scores) == 0:
            logger.log_message("No scores to plot")
            return
        summary_stats = self.summarize_scores(scores)
        # Save summary stats to JSON
        with open(f"{self.log_dir}/score_summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=4)

        # Save scores data to JSON
        json_data = {
            "scores": scores,
            "score_description": self.candidates[0].score_description
        }
        with open(f"{self.log_dir}/score_data.json", "w") as f:
            json.dump(json_data, f, indent=4)

        if type(scores[0]) == float:
            # Single metric case (float)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(scores)
            ax.set_xlabel("Candidate idx")
            ax.set_ylabel("Score")
            ax.set_title(f"{self.candidates[0].score_description} with each generation")
            ax.grid(True)
            fig.savefig(f"{self.log_dir}/score_plot.png")
        else:
            # Multiple metrics case (dictionary)
            metrics = list(scores[0].keys())
            n_metrics = len(metrics)
            
            # Create a figure with subplots - one for each metric
            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics), sharex=True)
            
            # Handle the case where there's only one metric (axes would not be an array)
            if n_metrics == 1:
                axes = [axes]
                
            # Plot each metric in its own subplot
            for i, metric in enumerate(metrics):
                ax = axes[i]
                metric_values = [s[metric] for s in scores]
                ax.plot(metric_values, marker='o', linewidth=2)
                ax.set_ylabel(metric)
                if type(self.candidates[0].score_description) == dict:
                    ax.set_title(f"{metric} - {self.candidates[0].score_description.get(metric, '')}")
                else:
                    ax.set_title(f"{metric} - {self.candidates[0].score_description}")
                ax.grid(True)
            
            # Set the x-label only for the bottom subplot
            axes[-1].set_xlabel("Candidate idx")
            
            # Add overall title and adjust layout
            plt.suptitle("Score metrics for each candidate generation", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
            
            # Save the figure
            fig.savefig(f"{self.log_dir}/score_plot.png")
            
            # Additionally, create individual plots for each metric
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                metric_values = [s[metric] for s in scores]
                ax.plot(metric_values, marker='o', linewidth=2)
                ax.set_xlabel("Candidate idx")
                ax.set_ylabel(metric)
                if type(self.candidates[0].score_description) == dict:
                    ax.set_title(f"{metric} - {self.candidates[0].score_description.get(metric, '')}")
                else:
                    ax.set_title(f"{metric} - {self.candidates[0].score_description}")
                ax.grid(True)
                safe_metric = metric.replace("/", "_")
                safe_metric = safe_metric.replace(" ", "_")
                fig.savefig(f"{self.log_dir}/{safe_metric}_score_plot.png")
                plt.close(fig)
            
        plt.close(fig)
    

    def convert_dict_object_to_list(self, dict_object, existing_data=None):
        if existing_data is None:
            # Handle the first insertion as before
            if isinstance(dict_object, dict):
                for k, v in dict_object.items():
                    if isinstance(v, (dict, list)):
                        dict_object[k] = self.convert_dict_object_to_list(v)
                    else:
                        dict_object[k] = [v]
                return dict_object
            elif isinstance(dict_object, list):
                return [self.convert_dict_object_to_list(i) for i in dict_object]
            else:
                return [dict_object]
        else:
            # Handle merging with existing data
            if isinstance(dict_object, dict):
                for k, v in dict_object.items():
                    if k in existing_data:
                        if isinstance(v, (dict, list)):
                            existing_data[k] = self.convert_dict_object_to_list(v, existing_data[k])
                        else:
                            existing_data[k].append(v)
                return existing_data
            elif isinstance(dict_object, list):
                return [self.convert_dict_object_to_list(new_item, existing_item) 
                       for new_item, existing_item in zip(dict_object, existing_data)]
            else:
                if isinstance(existing_data, list):
                    existing_data.append(dict_object)
                return existing_data
        

    def summarize_complete_results(self, complete_results):
        # calculate mean, min, max, stddev of any list in complete_results
        # complete_results may have multiple keys and they may be nested
        summary = {}
        for key, value in complete_results.items():
            if isinstance(value, list):
                summary[key] = {
                    "mean": sum(value) / len(value),
                    "min": min(value),
                    "max": max(value),
                    "stddev": statistics.stdev(value)
                }
            elif isinstance(value, dict):
                summary[key] = self.summarize_complete_results(value)
        return summary
    


    def get_IO500_walltime(self, candidate: Candidate):
        if candidate.complete_results is None:
            return None
        else:
            phases = candidate.complete_results["phases"]
            total_walltime = 0
            for phase in phases:
                total_walltime += float(phases[phase]["time"])
            return total_walltime
            
    def get_IOR_walltime(self, candidate: Candidate):
        if candidate.complete_results is None:
            return None
        else:
            write_results = float(candidate.complete_results["write"]["mean_time"])
            read_results = float(candidate.complete_results["read"]["mean_time"])
            return write_results + read_results

    def get_MDWorkbench_walltime(self, candidate: Candidate):
        if candidate.complete_results is None:
            return None
        else:
            return float(candidate.complete_results["total_runtime"])


    def get_H5Bench_walltime(self, candidate: Candidate):
        if candidate.complete_results is None:
            return None
        else:
            return float(candidate.complete_results["walltime"])

    def get_walltime_from_candidate(self, candidate: Candidate):
        if "IO500" in candidate.application.name:
            return self.get_IO500_walltime(candidate)
        if "IOR" in candidate.application.name:
            return self.get_IOR_walltime(candidate)
        if "MDWorkbench" in candidate.application.name:
            return self.get_MDWorkbench_walltime(candidate)
        if "amrex" in candidate.application.name.lower() or "macsio" in candidate.application.name.lower():
            return self.get_H5Bench_walltime(candidate)



    def save_walltime(self, application_alias: str, 
                      calculation_type: Literal["speedup", "aggregate"], 
                      aggregate_results_file: Union[str, None] = None, 
                      speedup_results_file: Union[str, None] = None, 
                      result_source: Union[str, None] = None, 
                      include_candidates_after_idx: int = 0,
                      using_rule_set: bool = False):
        logger.info(f"Saving walltime for {application_alias} with calculation type {calculation_type}")
        logger.info(f"Aggregate results file: {aggregate_results_file}")
        logger.info(f"Speedup results file: {speedup_results_file}")
        logger.info(f"Result source: {result_source}")
        logger.info(f"Include candidates after idx: {include_candidates_after_idx}")
        logger.info(f"Using rule set: {using_rule_set}")
        candidates_to_include = self.candidates[include_candidates_after_idx:]
        if calculation_type == "aggregate":
            if result_source is None:
                raise ValueError("result_source is required for aggregate calculation")
            #calculate the mean and stddev of the walltime of the candidates
            all_results = []
            for candidate in candidates_to_include:
                walltime = self.get_walltime_from_candidate(candidate)
                if walltime is not None:
                    all_results.append(walltime)
            if len(all_results) == 0:
                raise ValueError("No results to calculate aggregate")
            if len(all_results) == 1:
                mean = all_results[0]
                stddev = 0
            else:
                mean = statistics.mean(all_results)
                stddev = statistics.stdev(all_results)
            if os.path.exists(aggregate_results_file):
                with open(aggregate_results_file, "r") as f:
                    application_alias_data = json.load(f)
                if application_alias not in application_alias_data:
                    application_alias_data[application_alias] = {}
                application_alias_data[application_alias][result_source] = {
                    "mean": mean,
                    "stddev": stddev
                }
            else:
                # create the dirs if they don't exist
                os.makedirs(os.path.dirname(aggregate_results_file), exist_ok=True)
                application_alias_data = {application_alias: {result_source: {
                    "mean": mean,
                    "stddev": stddev
                }}}
            with open(aggregate_results_file, "w") as f:
                json.dump(application_alias_data, f, indent=4)
            
        elif calculation_type == "speedup":
            #calculate the speedup of the candidates
            if aggregate_results_file is None:
                raise ValueError("aggregate_results_file is required for speedup calculation")
            with open(aggregate_results_file, "r") as f:
                reference_data = json.load(f)
            if application_alias not in reference_data:
                raise ValueError(f"Application alias {application_alias} not found in reference file. Speedup calculation requires a reference file.")
            if "default" not in reference_data[application_alias]:
                raise ValueError(f"Default configuration not found in reference file for {application_alias}. Speedup calculation requires a reference file with default configuration.")
            reference_mean = reference_data[application_alias]["default"]["mean"]
            if os.path.exists(speedup_results_file):
                with open(speedup_results_file, "r") as f:
                    application_alias_data = json.load(f)
            else:
                os.makedirs(os.path.dirname(speedup_results_file), exist_ok=True)
                application_alias_data = {application_alias: None}
            if using_rule_set:
                rule_set_string = "With Rules"
            else:
                rule_set_string = "No Rules"
            if application_alias not in application_alias_data:
                application_alias_data[application_alias] = {"default_mean_performance": reference_mean, rule_set_string:{"tuning_iterations": [], "calculated_speedups": []}}
            else:
                if rule_set_string not in application_alias_data[application_alias]:
                    application_alias_data[application_alias][rule_set_string] = {"tuning_iterations": [], "calculated_speedups": []}
            for candidate_idx, candidate in enumerate(candidates_to_include):
                if candidate_idx == 0:
                    application_alias_data[application_alias][rule_set_string]["tuning_iterations"].append(reference_mean)
                    application_alias_data[application_alias][rule_set_string]["calculated_speedups"].append(1)
                    continue
                walltime = self.get_walltime_from_candidate(candidate)
                if walltime is not None:
                    application_alias_data[application_alias][rule_set_string]["tuning_iterations"].append(walltime)
                    application_alias_data[application_alias][rule_set_string]["calculated_speedups"].append(reference_mean / walltime)
            with open(speedup_results_file, "w") as f:
                json.dump(application_alias_data, f, indent=4)
        

        

    def save_complete_results(self):
        merged_complete_results = self.convert_dict_object_to_list(self.candidates[0].complete_results)
        if len(self.candidates) > 1:
            for candidate in self.candidates[1:]:
                merged_complete_results = self.convert_dict_object_to_list(candidate.complete_results, merged_complete_results)


        complete_results_summary = self.summarize_complete_results(merged_complete_results)

        # Save data to file
        with open(f"{self.log_dir}/complete_results_data.json", "w") as f:
            json.dump(merged_complete_results, f, indent=4)

        with open(f"{self.log_dir}/complete_results_summary.json", "w") as f:
            json.dump(complete_results_summary, f, indent=4)



    def plot_complete_results_IO500(self):
        phases = {"IOPS_phases": {"phases": {}, "units": "kIOPS"}, "BW_phases": {"phases": {}, "units": "GiB/s"}, "wall_time": {"phases": {}, "units": "seconds"}}
        
        # Collect data from candidates
        for candidate in self.candidates:
            if candidate.complete_results is not None:
                if "phases" in candidate.complete_results.keys():
                    for phase in candidate.complete_results["phases"]:
                        unit = candidate.complete_results["phases"][phase]["unit"]
                        value = candidate.complete_results["phases"][phase]["value"]
                        if "IOPS" in unit:
                            if phase not in phases["IOPS_phases"]["phases"]:
                                phases["IOPS_phases"]["phases"][phase] = [value]
                            else:
                                phases["IOPS_phases"]["phases"][phase].append(value)
                        elif "GiB/s" in unit:
                            if phase not in phases["BW_phases"]["phases"]:
                                phases["BW_phases"]["phases"][phase] = [value]
                            else:
                                phases["BW_phases"]["phases"][phase].append(value)
                    
                        

        # Save data to files
        # JSON format
        json_data = {
            "IOPS_phases": {
                "units": phases["IOPS_phases"]["units"],
                "phases": {phase: values for phase, values in phases["IOPS_phases"]["phases"].items()}
            },
            "BW_phases": {
                "units": phases["BW_phases"]["units"],
                "phases": {phase: values for phase, values in phases["BW_phases"]["phases"].items()}
            }
        }
        with open(f"{self.log_dir}/complete_results_data.json", "w") as f:
            json.dump(json_data, f, indent=4)


        # Create the plot
        for phase_type in phases.keys():
            if phase_type == "IOPS_phases":
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()  # Create second y-axis
                
                # Plot IOPS on left y-axis
                for phase_name, values in phases["IOPS_phases"]["phases"].items():
                    ax1.plot(values, label=phase_name, color='blue')
                ax1.set_xlabel("Candidate idx")
                ax1.set_ylabel(f"IOPS ({phases['IOPS_phases']['units']})", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Plot BW on right y-axis
                for phase_name, values in phases["BW_phases"]["phases"].items():
                    ax2.plot(values, label=phase_name, color='red', linestyle='--')
                ax2.set_ylabel(f"Bandwidth ({phases['BW_phases']['units']})", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.title("Complete results of candidates with each phase")
                fig.tight_layout()
                fig.savefig(f"{self.log_dir}/complete_results_plot_combined.png")
                plt.close(fig)
                break




        

    