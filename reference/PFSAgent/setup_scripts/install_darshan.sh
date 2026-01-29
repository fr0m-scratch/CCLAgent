#!/bin/bash

fs_name=$FS_NAME
# check if fs_name is set
if [ -z "$fs_name" ]; then
    echo "Error: FS_NAME is not set. Please set it in the environment."
    exit 1
fi

# read clients file from CLIENTS_FILE
clients_file=$CLIENTS_FILE
# check if clients_file is set
if [ -z "$clients_file" ]; then
    echo "Error: CLIENTS_FILE is not set. Please set it in the environment."
    exit 1
fi

cwd=$(pwd)

# read client nodes from clients_file
mapfile -t client_nodes < "$clients_file"
echo "Client nodes: ${client_nodes[@]}"

# write the darshan_config file at /custom-install/io-profilers/darshan_config to the client machines
# content of the file:
# MAX_RECORDS 100000000 POSIX
# MODMEM 1000
# NAMEMEM 100
# also export DARSHAN_CONFIG_PATH env var on all client machines'

for machine in "${client_nodes[@]}"; do
    ssh -tt -p 22 $machine << EOF
        source /opt/rh/gcc-toolset-13/enable
        cd /custom-install/io-profilers/darshan-3.4.5
        ./prepare.sh
        cd darshan-runtime
        ./configure --with-log-path=/mnt/$fs_name/darshan-logs --with-jobid-env=NONE CC=mpicc
        make
        make install
        exit
EOF
done

# install darshan-util on current machine
cd /custom-install/io-profilers/darshan-3.4.5/darshan-util
./configure
make
make install

cd /custom-install/io-profilers/darshan-3.4.5/darshan-runtime
chmod 777 darshan-mk-log-dirs.pl
mkdir -p /mnt/$fs_name/darshan-logs
./darshan-mk-log-dirs.pl

#check that /mnt/$fs_name/darshan-logs is not empty
if [ -z "$(ls -A /mnt/$fs_name/darshan-logs)" ]; then
    echo "Error: /mnt/$fs_name/darshan-logs is empty. The darshan-mk-log-dirs.pl script failed to configure the log directory."
    exit 1
fi

cd $cwd
echo "Darshan setup complete!"

