#!/bin/bash

# 模型权重路径 (例如你的 600 epoch 模型)
CKPT_PATH="./output_jit_b16"

# 你的验证集路径 (重要：请修改这里)
# 通常 ImageNet 的结构是 /path/to/val，里面包含大量图片或子文件夹
# 如果你的验证集图片在 /mnt/d/Dataset/ILSVRC2012_img_val
VAL_PATH="/mnt/d/Dataset/ILSVRC2012_img_val" 
TRAIN_DIR="/mnt/d/Dataset/ILSVRC2012_img_train"

# 输出目录
OUTPUT_DIR="./eval_output_epoch600"

mkdir -p ${OUTPUT_DIR}

# 启动评估
# 注意：--num_images 设置为 5000 (1000类别 * 5张/类别)
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--img_size 256 --noise_scale 1.0 \
--batch_size 130 \
--global_batch_size 1300 \
--gen_bsz 1024 \
--num_images 5000 \
--cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} \
--resume ${CKPT_PATH} \
--val_path ${VAL_PATH} \
--data_path ${TRAIN_DIR} \
--evaluate_gen