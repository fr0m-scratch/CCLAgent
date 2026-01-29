# STELLAR Reproduction Instructions

This README walks you through provisioning a CloudLab cluster, installing all software dependencies, running the automated evaluations, and interpreting the resulting plots that reproduce Figures 5–7 of the STELLAR paper.

## Prerequisites
 - CloudLab account with permission to allocate bare-metal nodes.
 - Familiarity with basic Linux shell commands and ssh.
 - Anthropic and OpenAI API keys.


## Steps
### Set Up the CloudLab Cluster
1. Load the profile 
     - Launch an experiment from the CloudLab profile:
https://www.cloudlab.us/show-profile.php?uuid=2b8aa435-245a-11f0-828b-e4434b2381fc

2.	Pick nodes:
     - Prefer Wisconsin site nodes c220g[1,2,5]. Group c220g5 was used in the original artifact evaluation — choose it if available.
     - The profile defaults to 5 client nodes and 6 server nodes. Keep this topology unless you have a specific reason to change it.

3.	Wait for the cluster to be provisioned.


### Configure Software Dependencies & Framework Code

##### IMPORTANT: Use a tmux session or another terminal multiplexer while following the rest of the instructions so long-running setup steps can survive disconnects.

1. Log in to node0 and initiate the pre-configured env:

```shell
ssh <your-cloudlab-username>@<node0_address>
sudo su -
```


2.	Download the source code:

```shell
cd /custom-install
wget <zenodo_link>
unzip STELLAR_SC.zip
```


3.	ONLY If your cluster is not 5 clients / 6 servers follow these steps:
	 - Create a `clients.txt` file in the  `/custom-install` directory
         - The file should contain one client hostname per line (i.e. node0, node1)
         - The client on the first line should be `node0` which you are connected to.
	 - Create a `servers.txt` file in the  `/custom-install` directory
         - The file should contain one server hostname per line 
         - The first line should be `server0` which will become the shared MGS/MDS
	 - Export paths so the setup scripts can find them:
        ```shell
        export SERVERS_FILE="/custom-install/servers.txt"
        export CLIENTS_FILE="/custom-install/clients.txt"
        ```

4.	Enter setup script directory and source host files:
```shell
cd /custom-install/PFSAgent/setup_scripts
source setup_cluster_files.sh
```

5.	Configure known-hosts files on all nodes:

```shell
./setup_ssh.sh
# Enter yes to all first-time authenticity prompts
```


6.	Run the rest of the automated setup:

```shell
./run_all_setup_script.sh
```


7.	Verify key installation points (see "Verification Procedures for the Setup Process" below).

8.	Set API keys for experiments:

```shell
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

---

### Verification Procedures for the Setup Process

Each step of the setup process carried out in the primary setup script (`run_all_setup_script.sh`) will create an output log `/custom-install/setup_logs/`. If an error occurs while running the setup script, the logs in these directories should be consulted first.

If no errors are reported while running the setup script the following commands can be run to manually check that the setup was successful (expand each element to view an example of the expected output):

<details>
<summary>Command: <code>lfs df</code></summary>
    
    UUID                   1K-blocks        Used   Available Use% Mounted on
    lustrefs-MDT0000_UUID   679719732      162084   622107156   1% /mnt/lustrefs[MDT:0]
    lustrefs-OST0001_UUID  1136649852    79534728   999646704   8% /mnt/lustrefs[OST:1]
    lustrefs-OST0002_UUID  1136649852    60996568  1018184864   6% /mnt/lustrefs[OST:2]
    lustrefs-OST0003_UUID  1136649852    57616876  1021564556   6% /mnt/lustrefs[OST:3]
    lustrefs-OST0004_UUID  1136649852    51321080  1027860352   5% /mnt/lustrefs[OST:4]

    filesystem_summary:   4546599408   249469252  4067256476   6% /mnt/lustrefs
</details>

<details>
<summary>Command: <code>cat /etc/mpi/hostfile</code></summary>
    
    node0
    node1
    node2
    node3
    node4
</details>

<details>
<summary>Command: <code>darshan-config --log-path</code></summary>
    
    /mnt/lustrefs/darshan-logs
</details>

<details>
<summary>Command: <code>which darshan-parser</code></summary>
    
    /usr/local/bin/darshan-parser
</details>



---
### Execute the Evaluations
1.	Run the automated script:
    ```shell
    cd /custom-install/PFSAgent/run_scripts
    ./run_all_evals.sh
    ```

The wrapper script performs three evaluations detailed below:

#### **Eval\_1 – Matching Human Baselines**

This sub-task reproduces the results of figure 5, highlighting STELLAR's ability to achieve similar tuning performance to human experts within 5 attempts. Note that in this sub-task, the wall time results plotted for each type of configuration (Default, Expert and STELLAR) is based on the average of 3 runs rather than 8 as reported in the paper. This reduction is made to conserve time during reproduction of the results. The results are replicated using the following procedure:

- Collect wall time results for all 5 benchmark workloads (3 runs each) using the default file system configuration settings  
- Collect wall time results for all 5 benchmark workloads (3 runs each) using the file system configuration settings generated by the system expert (these can be found in the `starter_configs` directory where labels include ‘expert’)  
- Allow STELLAR to tune each benchmark workload (with up to 5 iterations) and collect wall time results by re-running the best generated configuration 3 times for each benchmark  
- Plot the recorded results using the `plot_evals.py` script, saving one plot per workload in the `/eval_results/plots/Eval_1` directory  

Each sub-workflow invokes `plot_evals.py` to save publication-ready figures.


#### **Eval\_2 – Leveraging Previous Experience**

This sub-task recreates the evaluation results reported in figure 6. Figure 6 compares the per-iteration tuning results of STELLAR for each benchmark workload with no prior knowledge of the applications to those with a set of rules aggregated through a previous experience of tuning each workload. Since Eval\_1 already tunes each workload without any prior knowledge of the applications, these results are reused to conserve time during reproduction of the results. The remaining results required to reproduce Figure 6 are collected and visualized with the following procedure:

- Allow STELLAR to tune each benchmark workload (with up to 5 iterations) while using the set of tuning rules aggregated from running each of the benchmark workloads one time  
- Plot the recorded results and the STELLAR results from the first evaluation using the `plot_evals.py` script, saving one plot per workload in the `/eval_results/plots/Eval_2` directory  


#### **Eval\_3 – Generalizing to real-applications**

This sub-task recreates the evaluation results reported in figure 7. The procedure to collect and visualize the results is as follows:

- Collect wall time results for each of the 3 real-application workloads (3 runs each) using default file system configuration settings. The average of these results will be used to calculate the speedup achieved by each tuning iteration  
- Allow STELLAR to tune each of the 3 real-application workloads without any prior tuning knowledge set  
- Allow STELLAR to tune each of the 3 real-application workloads with the addition of the same rule set used in the previous evaluation which includes only knowledge aggregated from tuning the benchmark workloads  
- Plot the recorded results using the `plot_evals.py` script, saving one plot per workload in the `/eval_results/plots/Eval_3` directory 

---

### Analyze the Results

The resulting plots are found in /custom-install/eval_results/plots:


#### **Eval\_1 results**  
The plots in the Eval\_1 subdirectory correspond to the results in figure 5 of the paper, where each plot file corresponds to a single benchmark workload (5 in total). Each plot shows the comparison of wall time among the default file system configuration (labeled ‘default’), the settings suggested for the workload and system by a system expert (labeled ‘expert’), and the settings found by the STELLAR framework (labeled ‘STELLAR').



#### **Eval\_2 results**  
The plots in the Eval\_2 subdirectory correspond to the results shown in figure 6 of the paper, where each plot file corresponds to a single benchmark workload (5 in total). Each plot shows the speedup achieved by each tuning iteration generated by the STELLAR framework. Each plot includes the results of tuning the application without any initial knowledge of the benchmark workloads (in blue) and the tuning results with a set of tuning rules generated from previous experiences tuning the benchmark workloads (in red).


#### **Eval\_3 results**  
The plots in the Eval\_3 subdirectory correspond to the results shown in figure 7 of the paper, where each plot file corresponds to a single real-application workload (3 in total). Each plot shows the speedup achieved by each tuning iteration generated by the STELLAR framework. Each plot includes the results of tuning the application without any initial set of tuning rules (in blue) and the tuning results with the same set of tuning rule applied in Eval\_2 which only includes knowledge of the 5 benchmark workloads (in red).
"""
