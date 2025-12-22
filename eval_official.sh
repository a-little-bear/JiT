#!/bin/bash

# 你的官方权重路径 (请修改)
CKPT_PATH="./checkpoints/jit_b_16.pth"
# 哪怕是评估，代码可能也会检查 data_path，指向真实路径最稳妥
DATA_PATH="/mnt/d/Dataset/ILSVRC2012_img_train"
OUTPUT_DIR="./eval_output"

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--img_size 256 --noise_scale 1.0 \
--batch_size 64 \
--gen_bsz 64 \
--num_images 50000 \
--cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} \
--resume ${CKPT_PATH} \
--data_path ${DATA_PATH} \
--evaluate_gen