from STELLAR.Applications import Application
import os
from STELLAR.Utils.logger import LOGGERS
import json

logger = LOGGERS["application"]

KERNEL_APPLICATIONS = {
    "amrex": {
        "name": "AMReX",
        "description": "AMReX is a software framework for massively parallel, block-structured adaptive mesh refinement (AMR) applications"
    },
    "macsio": {
        "name": "MACSio",
        "description": "MACSio is a Multi-purpose, Application-Centric, Scalable I/O Proxy Application"
    }
}

class H5Bench(Application):
    name = "H5Bench"
    description = "H5Bench is a benchmark for HDF5."
    runtime_description = None
    score_metric = ["walltime"]
    score_metric_descriptions = {
        "walltime": "Time in seconds to complete the application run from start to finish"
    }


    def __init__(self, config_name: str = None):
        super().__init__(config_name)
        self.replace_app_name_from_config()
        self.update_config_vars("mpi", "command", "mpirun")
        self.update_config_vars("mpi", "ranks", self.stellar_config.config["procs"])
        self.update_config_vars("directory", value=self.data_dir)
        if self.stellar_config.config["mpi_hostfile"] != "":
            hostfile_string = f"--hostfile {self.stellar_config.config['mpi_hostfile']}"
        else:
            hostfile_string = ""
        self.update_config_vars("mpi", "configuration", f"{hostfile_string} -np {self.stellar_config.config['procs']} -env LD_PRELOAD {self.stellar_config.config['Darshan']['darshan_preload_path']}")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


    def build(self):
        pass


    def update_config_vars(self, config_section: str, key: str=None, value: str=None):
        # load the config json file
        config_path = self.get_config_path()
        with open(config_path, "r") as f:
            config = json.load(f)
        # update the config
        if key:
            config[config_section][key] = value
        else:
            config[config_section] = value
        # save the config
        with open(config_path, "w") as f:
            json.dump(config, f)


    def replace_app_name_from_config(self):
        for kernel_app in KERNEL_APPLICATIONS:
            if kernel_app in self.config_name:
                self.name = KERNEL_APPLICATIONS[kernel_app]["name"]
                self.description = KERNEL_APPLICATIONS[kernel_app]["description"]
                break

    def get_config_path(self):
        selected_config = f"{self.config_name}.json"
        configs_folder = os.path.join(os.path.dirname(__file__), "configs")
        config_files = os.listdir(configs_folder)
        if selected_config not in config_files:
            raise ValueError(f"Config file {selected_config} not found in {configs_folder}")
        return os.path.join(configs_folder, selected_config)
    

    def parse_config(self):
        config_path = self.get_config_path()
        with open(config_path, "r") as f:
            config_content = json.load(f)
        
        return {
            "config_path": config_path,
            "config_content": config_content
        }
    
    def get_run_command(self, application_config_dict: dict) -> list:
        return [
            os.path.join(os.path.dirname(__file__), "run.sh"),
            application_config_dict["config_path"],
            self.data_dir,
            self.results_dir
        ]
    

    def parse_results(self):
        walltime_results_file = os.path.join(self.results_dir, "total_walltime.txt")
        with open(walltime_results_file, "r") as f:
            walltime = float(f.read().strip().split(":")[1].strip())
        
        results = {
            "walltime": walltime
        }
        logger.info(f"H5Bench results: {results}")
        return results
    
    def get_score(self, results: dict) -> dict:
        return {
            "value": results,
            "description": self.score_metric_descriptions["walltime"]
        }
    
    
    