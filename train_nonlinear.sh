#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

# 设置你的数据路径
TRAIN_DIR="/mnt/d/Dataset/ILSVRC2012_img_train"
OUTPUT_DIR="./output_jit_b16"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 启动训练
# nproc_per_node=1 (单卡)
# batch_size=64 (物理显存限制，根据你的显存可调整，5090 24G 跑64-96应该没问题)
# global_batch_size=1024 (论文复现目标)
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--use_nonlinear \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 130 \
--num_sampling_steps 25 \
--sampling_method euler \
--subset_ratio 0.01 \
--global_batch_size 1300 \
--blr 5e-5 \
--epochs 1000 --warmup_epochs 5 \
--gen_bsz 1024 \
--num_images 2000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} \
--data_path ${TRAIN_DIR} \
--online_eval \
--eval_freq 20 \
--ema_decay1 0.999 \
--ema_decay2 0.9999 \
--env local