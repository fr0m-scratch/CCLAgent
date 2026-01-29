#!/bin/bash



# Setup IO500
cwd=$(pwd)
# install io500
cd /custom-install/benchmarks/io500
source /opt/rh/gcc-toolset-13/enable
./prepare.sh
make
cd $cwd
echo "IO500 setup complete! This include io500, ior, and mdworkbench."

