from STELLAR.Utils.stellar_config import StellarConfig
from STELLAR.Utils.logger import LOGGERS
from STELLAR.FS import FSConfig
from typing import Union
import subprocess
import os
import re
import time
from datetime import datetime
from typing import Literal
import shutil


logger = LOGGERS["application"]
sync_change_monitor_logger = LOGGERS["sync_change_monitor"]


class Application:
    name: str
    description: str
    runtime_description: Union[str, None]
    score_metric: str
    config_name: str
    debug: bool = False
    setup_complete: bool = False
    has_pre_execution_steps: bool = False
    use_mpi: bool = True
    analysis_agent_type: str = "Darshan"
    log_aggregation_settings: None | Literal["all"] | datetime = None

    def __init__(self, config_name: str = None):
        self.stellar_config = StellarConfig.get_instance()
        if config_name:
            self.config_name = config_name
        if self.stellar_config.config["Debug"]["use_debug_settings"]:
            self.config_name = "debug"
            self.debug = True
        if not self.config_name:
            raise ValueError("config_name is required")
        
        self.analysis_messages = None
        self.analysis_summary = None
        self.analysis_agent = None
        if self.name in self.stellar_config.config["Application_Configs"]:
            if "root_dir" in self.stellar_config.config["Application_Configs"][self.name]:
                self.application_root_dir = self.stellar_config.config["Application_Configs"][self.name]["root_dir"]
            if "results_dir" in self.stellar_config.config["Application_Configs"][self.name]:
                self.results_dir = self.stellar_config.config["Application_Configs"][self.name]["results_dir"]
            if "data_dir" in self.stellar_config.config["Application_Configs"][self.name]:
                self.data_dir = self.stellar_config.config["Application_Configs"][self.name]["data_dir"]


    def init_analysis_agent(self, fs_config_class: FSConfig):
        from STELLAR.RuntimeAnalysis import ANALYSIS_AGENTS
        kwargs = {}
        if self.log_aggregation_settings:
            kwargs["log_aggregation_settings"] = self.log_aggregation_settings
        self.analysis_agent = ANALYSIS_AGENTS[self.analysis_agent_type](fs_config_class, **kwargs)

    def build(self):
        raise NotImplementedError
    
    def set_log_aggregation_settings(self, log_aggregation_settings: None | Literal["all"] | datetime):
        self.log_aggregation_settings = log_aggregation_settings
    
    

    def get_run_command(self, application_config_dict: dict) -> list:
        # get the run command from the application config
        # the command is a list of strings
        raise NotImplementedError
    

    def parse_config(self) -> dict:
        # parse the application config
        # create a dictionary with "config_path" and "config_content" keys
        # config_path is the path to the application config file
        # config_content is the content of the application config file
        raise NotImplementedError

    def parse_results(self) -> dict:
        # parse the results from the application
        raise NotImplementedError


    def get_score(self, application_results: dict) -> dict[str, Union[float, dict[str, float]]]:
        # get the score from the application results
        # score is a dictionary with "value" and "description" keys
        # value is either a float or a dictionary with the score values per metric
        # description is either a string or a dictionary with the score descriptions per metric
        raise NotImplementedError
    

    def clear_cache_with_ior(self):
        mount_point = self.stellar_config.config["System"]["mount_root"]
        commands = [f"/custom-install/benchmarks/io500/bin/ior -w -b 1G -t 1M -F -o {mount_point}/ior_test", f"rm {mount_point}/ior_test"]
        for command in commands:
            subprocess.run(command, shell=True)


    def run_and_score(self, runtime_analysis: bool = False):
        logger.info(f"Running {self.name}")
        application_config_dict = self.parse_config()

        if "config_path" in application_config_dict:
            logger.info(f"Running {self.name} with config: {application_config_dict['config_path']}")
            with open(application_config_dict["config_path"], "r") as f:
                logger.info(f"config content: {f.read()}")
        elif "config_content" in application_config_dict:
            logger.info(f"Running {self.name} with config: {application_config_dict['config_content']}")
        else:
            raise ValueError(f"config_path or config_content not found in application_config_dict for {self.name}")
        

        run_command = self.get_run_command(application_config_dict)
        preload_path=self.stellar_config.config["Darshan"]["darshan_preload_path"]
        if self.use_mpi:
            env_var_name = f"{self.name.upper()}_MPIARGS"
            mpi_env_flags = self.stellar_config.config["mpi_flags"]
            env_flags_string = ""
            hostfile_string = ""
            for flag in mpi_env_flags:
                env_flags_string += f"-env {flag} "
            if self.stellar_config.config["mpi_hostfile"] != "":
                hostfile_string = f"--hostfile {self.stellar_config.config['mpi_hostfile']}"
            os.environ[env_var_name] = f"{hostfile_string} -np {self.stellar_config.config['procs']} {env_flags_string} -env LD_PRELOAD {preload_path}"
            logger.info(f"ENV ARGS: {os.environ[env_var_name]}")
        else:
            env_var_name = f"{self.name.upper()}_RUNARGS"
            os.environ[f"{self.name.upper()}_RUNARGS"] = f"env LD_PRELOAD={preload_path} env DARSHAN_ENABLE_NONMPI=1"
            logger.info(f"ENV ARGS: {os.environ[env_var_name]}")


        run_environment = os.environ.copy()
        
        if self.has_pre_execution_steps:
            pre_execution_steps = self.get_pre_execution_steps(application_config_dict)
            for step in pre_execution_steps:
                logger.info(f"Running pre-execution step: {step}")
                if "command" in step:
                    command_str = " ".join(step["command"])
                    logger.info(f"Running pre-execution command: {command_str}")
                    subprocess.run(command_str, shell=True)
                elif "method" in step:
                    if "args" in step:
                        step["method"](**step["args"])
                    else:
                        step["method"]()


        command_str = " ".join(run_command)
        logger.info(f"Running {self.name} with command: {command_str}")
        run_output = subprocess.run(command_str,
                                    env=run_environment,
                                    shell=True)
        logger.info(f"run_output: {run_output}")
        
        logger.info(f"{self.name} run output: {run_output.stdout}")
        logger.info(f"{self.name} run error: {run_output.stderr}")

        if run_output.returncode == 0:
            logger.info(f"{self.name} run completed successfully")

        else:
            logger.error(f"{self.name} run failed with return code {run_output.returncode}")
            logger.error(f"{self.name} run output: {run_output.stdout}")
            logger.error(f"{self.name} run error: {run_output.stderr}")
            raise Exception(f"{self.name} run failed with return code {run_output.returncode}")
        
        try:
            application_results = self.parse_results()
        except Exception as e:
            logger.error(f"Error parsing {self.name} results: {e}")
            raise e

        try:
            score = self.get_score(application_results)
        except Exception as e:
            logger.error(f"Error getting {self.name} score: {e}")
            raise e
        
        if runtime_analysis:
            try:
                darshan_results = self.run_analysis_agent()
                self.update_runtime_description(darshan_results)
                logger.info(f"Darshan results: {darshan_results}")
            except Exception as e:
                logger.error(f"Error running analysis agent: {e}")

        # move the results to the logging directory
        if self.results_dir:
            log_dir = self.stellar_config.config["Logging"]["log_dir"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_log_dir = os.path.join(log_dir, "application_results", timestamp)
            if not os.path.exists(results_log_dir):
                os.makedirs(results_log_dir)
            shutil.copytree(self.results_dir, results_log_dir, dirs_exist_ok=True)
            results_entries = os.listdir(self.results_dir)
            # remove the files and subdirectories of the results directory
            for entry in results_entries:
                path = os.path.join(self.results_dir, entry)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

        
        
        return score, application_results
            

    @classmethod
    def update_runtime_description(cls, runtime_description: str):
        cls.runtime_description = runtime_description

    def run_analysis_agent(self):
        if self.analysis_agent is None:
            raise ValueError("Analysis agent not initialized")
        analysis_messages, analysis_summary = self.analysis_agent.run()
        self.analysis_messages = analysis_messages
        return analysis_summary
    

    def clear_remote_cache(self, hostname: str):
        # ssh to hostname
        subprocess.run(f"ssh {hostname} 'echo 3 > /proc/sys/vm/drop_caches'", shell=True)
        subprocess.run(f"ssh {hostname} 'sync'", shell=True)
        # clear lustre cache
        subprocess.run(f"ssh {hostname} 'lctl set_param ldlm.*.*.lru_size=clear'", shell=True)

    def clear_local_cache(self):
        subprocess.run(f"echo 3 > /proc/sys/vm/drop_caches", shell=True)
        subprocess.run(f"sync", shell=True)
        # clear lustre cache
        subprocess.run(f"lctl set_param ldlm.*.*.lru_size=clear", shell=True)


    def clear_all_caches(self):
        # get current client by hostname
        current_client = subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout.strip().split(".")[0]
        clients = self.stellar_config.config["System"]["Clients"]
        for client in clients:
            if current_client in client or current_client == client:
                continue
            # clear lustre cache for client
            self.clear_remote_cache(client)
        
        # clear local cache
        self.clear_local_cache()

        servers = self.stellar_config.config["System"]["Servers"]
        for server in servers:
            self.clear_remote_cache(server)


    def clear_and_remount(self):
        lustre_mount_point = self.stellar_config.config["System"]["mount_root"]
        clear_fs_command = f"rm -rf {self.data_dir}"
        try:
            res = subprocess.run(clear_fs_command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            logger.info(f"Command '{clear_fs_command}' completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clearing {self.data_dir}: {e}")
            raise e
        
        current_client = subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout.strip().split(".")[0]
        clients = self.stellar_config.config["System"]["Clients"]
        for client in clients:
            if current_client in client or current_client == client:
                # Local remount
                umount_command = f"umount {lustre_mount_point}"
                mount_command = f"mount -t lustre {self.stellar_config.config['System']['MGS_HOST']}@{self.stellar_config.config['System']['MGS_CONN_TYPE']}:/{self.stellar_config.config['System']['FS_NAME']} {lustre_mount_point}"
            else:
                # Remote remount - both commands need to run on the remote host
                umount_command = f"ssh {client} 'umount {lustre_mount_point}'"
                mount_command = f"ssh {client} 'mount -t lustre {self.stellar_config.config['System']['MGS_HOST']}@{self.stellar_config.config['System']['MGS_CONN_TYPE']}:/{self.stellar_config.config['System']['FS_NAME']} {lustre_mount_point}'"
            try:
                res = subprocess.run(umount_command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                logger.info(f"Command '{umount_command}' completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command '{umount_command}' failed with return code {e.returncode}")
                logger.error(f"Error output: {e.stderr.decode()}")
            try:
                res = subprocess.run(mount_command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                logger.info(f"Command '{mount_command}' completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command '{mount_command}' failed with return code {e.returncode}")
                logger.error(f"Error output: {e.stderr.decode()}")
                raise


    def wait_for_sync_changes(self, mds_list, check_interval=2, verbose=True):
        """
        1) For each MDS node, read the current 'osc.*.max_rpcs_in_progress' values
        and store them.
        2) Set all to 16384 to speed up processing.
        3) Wait until all MDS nodes have osc.*.sync_changes == 0.
        4) Restore the original 'max_rpcs_in_progress' values on each MDS node.

        :param mds_list: A list of MDS node hostnames (or IPs).
        :param check_interval: number of seconds to sleep between checks of sync_changes
        :param verbose: if True, print progress messages
        """

        # Regex to parse lines like:
        #   osc.myfs-OST0000-osc-MDT0000.max_rpcs_in_progress=512
        rpc_pattern = re.compile(r'^(osc\.[^=]+)\.max_rpcs_in_progress=(\d+)$')
        # Regex for lines like:
        #   osc.hasanfs-OST0000-osc-MDT0000.sync_changes=1747330
        sync_pattern = re.compile(r'^.*sync_changes=(\d+)$')

        # Step 1: Gather the current max_rpcs_in_progress on each MDS
        # We'll store them in a dict of the form:
        #   saved_params[MDS][osc_name] = old_value
        saved_params = {}

        for mds in mds_list:
            saved_params[mds] = {}
            # 1A) read current param values:
            cmd = ["ssh", mds, "lctl", "get_param", "osc.*.max_rpcs_in_progress"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"[{mds}] Failed to get max_rpcs_in_progress:\n{result.stderr}")

            lines = result.stdout.strip().splitlines()
            for line in lines:
                match = rpc_pattern.match(line.strip())
                if match:
                    osc_name, old_val_str = match.groups()
                    old_val = int(old_val_str)
                    # e.g. osc_name='osc.hasanfs-OST0000-osc-MDT0000'
                    saved_params[mds][osc_name] = old_val

            if verbose:
                sync_change_monitor_logger.info(f"[{mds}] Found {len(saved_params[mds])} OSTs with max_rpcs_in_progress")

        NEW_VALUE = 16384
        try:
            # Step 2: set all to 16384
            if verbose:
                sync_change_monitor_logger.info(f"Setting all MDS max_rpcs_in_progress to {NEW_VALUE}")

            for mds in mds_list:
                for osc_name in saved_params[mds]:
                    cmd = ["ssh", mds, "lctl", "set_param",
                        f"{osc_name}.max_rpcs_in_progress={NEW_VALUE}"]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(f"[{mds}] Failed setting {osc_name}.max_rpcs_in_progress:\n{result.stderr}")

            # Step 3: Wait for sync_changes to hit 0 on all MDSes
            if verbose:
                sync_change_monitor_logger.info("Now waiting for osc.*.sync_changes to reach 0 on all MDS nodes...")

            while True:
                all_done = True
                for mds in mds_list:
                    # read sync_changes from each MDS
                    cmd = ["ssh", mds, "lctl", "get_param", "osc.*.sync_changes"]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(f"[{mds}] get_param sync_changes failed:\n{result.stderr}")

                    lines = result.stdout.strip().splitlines()
                    total_changes = 0
                    for line in lines:
                        m = sync_pattern.match(line.strip())
                        if m:
                            total_changes += int(m.group(1))

                    if verbose:
                        sync_change_monitor_logger.info(f"   [{mds}] sync_changes total={total_changes}")

                    if total_changes > 0:
                        all_done = False

                if all_done:
                    if verbose:
                        sync_change_monitor_logger.info("All MDS nodes have zero sync_changes. Done.")
                    break

                time.sleep(check_interval)

        finally:
            # Step 4: restore the old max_rpcs_in_progress on each MDS
            if verbose:
                sync_change_monitor_logger.info("Restoring original max_rpcs_in_progress values...")

            for mds in mds_list:
                for osc_name, old_val in saved_params[mds].items():
                    cmd = ["ssh", mds, "lctl", "set_param",
                        f"{osc_name}.max_rpcs_in_progress={old_val}"]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(f"[{mds}] Failed restoring {osc_name}.max_rpcs_in_progress:\n{result.stderr}")

            if verbose:
                sync_change_monitor_logger.info("All original values restored.")

    def reset_FS(self):
        mds_list = self.stellar_config.config["System"]["MDS"]
        self.clear_and_remount()
        self.clear_all_caches()
        self.wait_for_sync_changes(mds_list)


    


        
            
        