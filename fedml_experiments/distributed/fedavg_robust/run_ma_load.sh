#!/usr/bin/env bash

CLIENT_NUM=100
WORKER_NUM=10
MODEL="resnet18"
DISTRIBUTION="hetero"
DATASET="cifar10"
DATA_DIR="./../../../data/cifar10"
GPU_MAPPING_FILE="gpu_mapping.yaml"
GPU_MAPPING_KEY="mapping_gpu02_alone_10"
ROUND=10000
TEST_FREQ=10
LOAD_MODEL_PATH="models/cifar10-resnet18/sgd-0.1-10000-2-Sat-Nov-13-15-49-45-2021"

# server defence
DEFENSE_TYPE="none"
# DEFENSE_TYPE="norm_diff_clipping"
NORM_BOUND=5.0
STDDEV=0.025

# attacker client
POISON_TYPE="plus"
# POISON_TYPE="southwest"
POISON_FRAC=0.5
ATTACK_NUM=1
ATTACK_FREQ=1
# ATTACK_CASE=
ATTACKER_EPOCH=2
ATTACKER_LR=0.1
ATTACKER_OPTIM="sgd"
ATTACKER_TYPE="multi_shot"
ROUNDS_AFTER_ATTACK=200

# normal client
BATCH_SIZE=64
EPOCH=2
LR=0.1
OPTIM="sgd"

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

rm -f output.log

DEFENSE_TYPE="none"
NORM_BOUND=5.0
STDDEV=0.025
ATTACK_NUM=1
ATTACKER_EPOCH=2
ATTACKER_LR=0.1
POISON_FRAC=0.5
mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg_robust.py \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --model $MODEL \
  --partition_method $DISTRIBUTION  \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --gpu_mapping_file $GPU_MAPPING_FILE \
  --gpu_mapping_key $GPU_MAPPING_KEY \
  --comm_round $ROUND \
  --frequency_of_the_test $TEST_FREQ \
  --load_model_path $LOAD_MODEL_PATH \
  --defense_type $DEFENSE_TYPE \
  --norm_bound $NORM_BOUND \
  --stddev $STDDEV \
  --poison_type $POISON_TYPE \
  --poison_frac $POISON_FRAC \
  --attack_num $ATTACK_NUM
  --attack_freq $ATTACK_FREQ \
  --attack_epochs $ATTACKER_EPOCH \
  --attack_lr $ATTACKER_LR \
  --attack_optimizer $ATTACKER_OPTIM \
  --attack_type $ATTACKER_TYPE \
  --attack_afterward $ROUNDS_AFTER_ATTACK \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCH \
  --lr $LR \
  --client_optimizer $OPTIM \
  --note "Multi-shot attack" \
  --title "MA, $ATTACK_FREQ| $POISON_TYPE-$POISON_FRAC-$ATTACK_NUM <-> $DEFENSE_TYPE-$NORM_BOUND-$STDDEV | $ATTACKER_LR-$ATTACKER_EPOCH-$ATTACKER_OPTIM, $ROUND, $MODEL, $DATASET"
