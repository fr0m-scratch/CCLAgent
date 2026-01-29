#!/bin/bash

# check if CLIENTS_FILE and SERVERS_FILE are set. 
# If they are, we will use them instead of creating our own.
if [ -z "$CLIENTS_FILE" ]; then
    echo "CLIENTS_FILE is not set. Creating our own."

    client_file_path="/custom-install/clients.txt"
    cat > $client_file_path << EOF
client0
client1
client2
client3
client4
EOF
    echo "New clients file was created at $client_file_path."
    echo "setting CLIENTS_FILE to $client_file_path"
    export CLIENTS_FILE=$client_file_path
else
    echo "CLIENTS_FILE was already set in the environment to $CLIENTS_FILE"
fi

if [ -z "$SERVERS_FILE" ]; then
    echo "SERVERS_FILE is not set. Creating our own."

    server_file_path="/custom-install/servers.txt"
    cat > $server_file_path << EOF
server0
server1
server2
server3
server4
server5
EOF
    echo "New servers file was created at $server_file_path."
    echo "setting SERVERS_FILE to $server_file_path"
    export SERVERS_FILE=$server_file_path
else
    echo "SERVERS_FILE was already set in the environment to $SERVERS_FILE"
fi