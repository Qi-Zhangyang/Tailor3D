#!/usr/bin/env bash

set -x

PARTITION=$1
QUOTA=$2
JOB_NAME=$3
GPUS=$4

ACCELERATE_ARGS=${ACCELERATE_ARGS:-""}
CFG_PATH=${CFG_PATH:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --quotatype=${QUOTA} \
    --gres=gpu:${GPUS} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    accelerate launch --config_file ${ACCELERATE_ARGS} -m openlrm.launch train.lrm --config ${CFG_PATH}