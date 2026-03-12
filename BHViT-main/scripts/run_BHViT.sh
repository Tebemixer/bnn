DATA_DIR=${DATA_DIR:-./dataset}

# Single-node launch. Set NPROC_PER_NODE to number of GPUs.
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Use main_new.py (main.py does not exist in this fork).
torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=25641 main_new.py \
    --num-workers=2 \
    --batch-size=128 \
    --epochs=100 \
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
    --output-dir= \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-small-patch16-224 \
    --teacher-model-file= \
    --model=configs/BHViT \
    --model-type=BHViT \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp \
    --resume=
