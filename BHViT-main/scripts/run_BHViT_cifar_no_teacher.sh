#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-./dataset}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-25641}
OUTPUT_DIR=${OUTPUT_DIR:-./output_cifar_no_teacher}

torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} main_new.py \
    --num-workers=8 \
    --batch-size=32 \
    --epochs=4\
    --dropout=0.0 \
    --drop-path=0.0 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.00 \
    --lr=5e-4 \
    --warmup-epochs=0 \
    --color-jitter=0.0 \
    --aa=noaug \
    --reprob=0.0 \
    --mixup=0.0 \
    --cutmix=0.0 \
    --data-set=CIFAR \
    --data-path=${DATA_DIR} \
    --output-dir=${OUTPUT_DIR} \
    --model=configs/BHViT \
    --model-type=BHViT \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp
