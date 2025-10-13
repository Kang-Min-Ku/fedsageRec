#!/usr/bin/env bash

echo "It is going on"
WORKER_NUM=$1
CONFIG_FILE=${2}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 main_test.py \
  --client_num_per_round $WORKER_NUM \
  --config $CONFIG_FILE
