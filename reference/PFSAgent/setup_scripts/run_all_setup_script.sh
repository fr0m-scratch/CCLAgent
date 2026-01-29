#!/bin/bash

# Exit on any error
set -e

# add lustrefs to environment
export FS_NAME="lustrefs"

echo "=========================================="
echo "Starting Cluster Setup Process"
echo "=========================================="

echo "Creating setup logs directory..."
mkdir -p /custom-install/setup_logs
echo "✓ Setup logs directory created"

########################################################
# mount Lustre
########################################################

echo ""
echo "🗄️  Mounting Lustre filesystem..."
./mount_lustre.sh > /custom-install/setup_logs/mount_lustre.log 2>&1
echo "✓ Lustre filesystem mounted successfully"

########################################################
# Setup mpi
########################################################

echo ""
echo "🔧 Setting up MPI..."
./setup_mpi.sh > /custom-install/setup_logs/setup_mpi.log 2>&1
echo "✓ MPI setup completed successfully"

########################################################
# Setup Darshan
########################################################

echo ""
echo "📊 Installing Darshan..."
./install_darshan.sh > /custom-install/setup_logs/install_darshan.log 2>&1
echo "✓ Darshan installation completed successfully"

########################################################
# Setup applications
########################################################

echo ""
echo "🚀 Setting up applications..."
./setup_applications.sh > /custom-install/setup_logs/setup_applications.log 2>&1
echo "✓ Applications setup completed successfully"

########################################################
# install requirements
########################################################

echo ""
echo "📦 Installing Python requirements..."
cwd=$(pwd)
cd /custom-install/PFSagent
pip install -r requirements.txt > /custom-install/setup_logs/pip_install.log 2>&1
cd $cwd
echo "✓ Python requirements installed successfully"

echo ""
echo "=========================================="
echo "🎉 Cluster setup completed successfully!"
echo "=========================================="

