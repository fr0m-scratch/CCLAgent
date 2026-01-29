#!/bin/bash

fs_name=$FS_NAME
# check if fs_name is set
if [ -z "$fs_name" ]; then
    echo "Error: FS_NAME is not set. Please set it in the environment."
    exit 1
fi


# read servers file from SERVERS_FILE
servers_file=$SERVERS_FILE
# check if servers_file is set
if [ -z "$servers_file" ]; then
    echo "Error: SERVERS_FILE is not set. Please set it in the environment."
    exit 1
fi

# read clients file from CLIENTS_FILE
clients_file=$CLIENTS_FILE
# check if clients_file is set
if [ -z "$clients_file" ]; then
    echo "Error: CLIENTS_FILE is not set. Please set it in the environment."
    exit 1
fi

# mgs node is the first line of the servers file
mgs_node=$(head -n 1 $servers_file)

mgs_ip=$(ssh $mgs_node << 'ENDSSH'
device=$(awk "{print \$1}" /var/emulab/boot/ifmap)
ip addr show $device | grep -oP "inet \K[\d.]+"
ENDSSH
)

# Verify the IP was retrieved
if [ -z "$mgs_ip" ]; then
    echo "Error: Failed to get MGS IP"
    exit 1
else
    echo "MGS IP is: $mgs_ip"
fi

ssh -tt -p 22 $mgs_node  << EOF
device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')

ip addr add ${mgs_ip}/24 dev \$device
ip link set \$device up


rmmod -f lustre
rmmod -f lov
rmmod -f mdc
rmmod -f lmv
rmmod -f ptlrpc
rmmod -f obdclass
rmmod -f ksocklnd
rmmod -f lnet
rmmod -f libcfs

echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf

modprobe libcfs
modprobe lnet
lctl network configure
lctl network up
lctl list_nids

mkfs.lustre --mgs --reformat /dev/sda3

mkfs.lustre --fsname=$fs_name --mgsnode=${mgs_ip}@tcp --mdt --index=0 --reformat /dev/sda4
mkdir /mnt/mgt /mnt/mdt
mount -t lustre /dev/sda3 /mnt/mgt
mount -t lustre /dev/sda4 /mnt/mdt
exit
EOF


# List of servers from SERVERS_FILE
mapfile -t server_machines < "$servers_file"

# Starting index for --index and OST directories
start_index=0
ost_per_node=1

# Loop through each machine
for machine in "${server_machines[@]}"; do
echo "Running setup for $machine"

# Assign the current index values
idx1=$((start_index))

# Assign the directory names
ost1="ost${idx1}"

# Run the SSH commands
ssh -T -p 22 $machine << EOF
device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')
ip addr add \$existing_ip/24 dev \$device
ip link set \$device up
echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf
mkfs.lustre --fsname=$fs_name --ost --mgsnode=${mgs_ip}@tcp --index=$idx1 --reformat /dev/sda4
mkdir -p /mnt/$ost1
mount -t lustre /dev/sda4 /mnt/$ost1
lctl set_param ost.OSS.ost_io.nrs_policies="tbf"
lctl set_param *.*.job_cleanup_interval=2
exit
EOF


start_index=$((start_index+ost_per_node))
done



# Array of client machine names from CLIENTS_FILE (excluding the first one)
mapfile -t client_machines < "$clients_file"
client_machines=("${client_machines[@]:1}")

# Loop through each machine
for machine in "${client_machines[@]}"; do
echo "Running setup for $machine"

# Run the SSH commands
ssh -T -p 22 $machine << EOF
device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')
ip addr add \$existing_ip/24 dev \$device
ip link set \$device up

rmmod -f lustre
rmmod -f lov
rmmod -f mdc
rmmod -f lmv
rmmod -f ptlrpc
rmmod -f obdclass
rmmod -f ksocklnd
rmmod -f lnet
rmmod -f libcfs

echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf

modprobe libcfs
modprobe lnet
lctl network configure
lctl network up
lctl list_nids

 
if [ -d "/mnt/$fs_name" ]; then
    rm -rf /mnt/$fs_name
fi
mkdir /mnt/$fs_name/
mount -t lustre ${mgs_ip}@tcp:/$fs_name /mnt/$fs_name/
exit
EOF

# run for the client currently connected to
device=$(awk '{print $1}' /var/emulab/boot/ifmap)
existing_ip=$(ip addr show $device | grep -oP 'inet \K[\d.]+')
ip addr add $existing_ip/24 dev $device
ip link set $device up
rmmod -f lustre
rmmod -f lov
rmmod -f mdc
rmmod -f lmv
rmmod -f ptlrpc
rmmod -f obdclass
rmmod -f ksocklnd
rmmod -f lnet
rmmod -f libcfs
echo "options lnet networks=tcp($device)" > /etc/modprobe.d/lustre.conf
modprobe libcfs
modprobe lnet
lctl network configure
lctl network up
lctl list_nids

if [ -d "/mnt/$fs_name" ]; then
    rm -rf /mnt/$fs_name
fi
mkdir /mnt/$fs_name/
mount -t lustre ${mgs_ip}@tcp:/$fs_name /mnt/$fs_name/




done
