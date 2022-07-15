#!/usr/bin/env bash

source ~/.bashrc

find /app -type f -name '*.pyc' -delete

# HPC SDK entrypoint
if [ "x$MPIVER" = "xHPCX" ]; then \
   echo "Loading HPCX"
   source $HPCSDK_HOME/comm_libs/hpcx/latest/hpcx-init.sh
   hpcx_load
fi;

exec "$@"
