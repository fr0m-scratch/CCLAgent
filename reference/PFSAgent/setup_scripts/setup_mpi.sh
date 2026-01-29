#!/bin/bash

# read clients file from CLIENTS_FILE
clients_file=$CLIENTS_FILE
# check if clients_file is set
if [ -z "$clients_file" ]; then
    echo "Error: CLIENTS_FILE is not set. Please set it in the environment."
    exit 1
fi

# read client nodes from clients_file
mapfile -t client_nodes < "$clients_file"

cwd=$(pwd)
# go to home directory
cd ~

# Create hostfile on each node
for machine in "${client_nodes[@]}"; do
    # Generate hostfile content dynamically
    hostfile_content=""
    for node in "${client_nodes[@]}"; do
        # Skip adding the current node to its own hostfile
        hostfile_content+="${node}\n"
    done
        ssh -T -p 22 $machine << EOF  
        # Create MPI hostfile
        mkdir -p /etc/mpi
        echo -e "${hostfile_content}" > /etc/mpi/hostfile
        exit
EOF
done

# Test MPI connectivity
echo "Testing MPI connectivity..."
mpirun --hostfile /etc/mpi/hostfile \
       -bind-to numa \
       -np 16 \
       hostname

# go back to original directory
cd $cwd
echo "MPI setup complete!"



