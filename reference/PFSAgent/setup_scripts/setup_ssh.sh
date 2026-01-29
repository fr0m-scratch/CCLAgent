#!/bin/bash

# Exit on any error
set -e

# Check if environment variables are set
if [[ -z "$SERVERS_FILE" || -z "$CLIENTS_FILE" ]]; then
    echo "Error: SERVERS_FILE and CLIENTS_FILE environment variables must be set"
    exit 1
fi

# Read all hosts into arrays
mapfile -t servers < "$SERVERS_FILE"
mapfile -t clients < "$CLIENTS_FILE"

# Combine all hosts
all_hosts=("${servers[@]}" "${clients[@]}")

echo "Setting up known hosts for ${#all_hosts[@]} nodes..."

# For each host, add all other hosts to its known_hosts
for host in "${all_hosts[@]}"; do
    echo "Configuring known hosts on $host..."
    
    # Use ssh-keyscan to get host keys and add to known_hosts
    ssh "$host" "
        mkdir -p ~/.ssh
        chmod 700 ~/.ssh
        touch ~/.ssh/known_hosts
        
        # Clear existing entries for these hosts to avoid duplicates
        $(printf 'ssh-keygen -R %s 2>/dev/null || true; ' "${all_hosts[@]}")
        
        # Add all hosts to known_hosts
        ssh-keyscan -H $(printf '%s ' "${all_hosts[@]}") 2>/dev/null >> ~/.ssh/known_hosts
        
        chmod 600 ~/.ssh/known_hosts
    "
done

echo "Known hosts setup completed successfully!"
