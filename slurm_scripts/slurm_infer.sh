#!/usr/bin/env bash

set -x

PARTITION=$1
QUOTA=$2
JOB_NAME=$3
GPUS=$4

CFG_PATH=${CFG_PATH:-""}
INFER_ARGS=${INFER_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

srun_command="srun -p ${PARTITION} --job-name=${JOB_NAME} --quotatype=${QUOTA} --gres=gpu:${GPUS} --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1"

$srun_command python -m openlrm.launch infer.lrm --infer ${CFG_PATH} ${INFER_ARGS}
