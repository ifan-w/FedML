#!/usr/bin/env bash

CLIENT_NUM=100
WORKER_NUM=10
MODEL="resnet56"
DISTRIBUTION="hetero"
ROUND=100
EPOCH=20
BATCH_SIZE=64
LR=0.001
DATASET="cifar10"
DATA_DIR="./../../../data/cifar10"
DEFENSE_TYPE="norm_diff_clipping"
NORM_BOUND=5.0
STDDEV=0.025
POISON_TYPE="southwest"
ATTACK_FREQ=10

# CLIENT_NUM=$1
# WORKER_NUM=$2
# SERVER_NUM=$3
# GPU_NUM_PER_SERVER=$4
# MODEL=$5
# DISTRIBUTION=$6
# ROUND=$7
# EPOCH=$8
# BATCH_SIZE=$9
# LR=$10
# DATASET=$11
# DATA_DIR=$12
# DEFENSE_TYPE=$13
# NORM_BOUND=$14
# STDDEV=$15
# POISON_TYPE=$16
# ATTACK_FREQ=$17


PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg_robust.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_gpu02_alone_10" \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --defense_type $DEFENSE_TYPE \
  --norm_bound $NORM_BOUND \
  --stddev $STDDEV \
  --poison_type $POISON_TYPE \
  --attack_freq $ATTACK_FREQ

  # --gpu_server_num $SERVER_NUM \
  # --gpu_num_per_server $GPU_NUM_PER_SERVER \