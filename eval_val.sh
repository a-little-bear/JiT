#!/bin/bash

# 设置对比实验的名称: original / nonlinear / nonlinear_ss_rotation / nonlinear_ss_time
# 你可以手动修改这个变量来跑不同的实验
name="original"

# 模型权重根目录
CKPT_ROOT="./results/${name}/output_jit_b16"

# 验证集路径 (请确保路径正确)
VAL_PATH="/mnt/d/Dataset/ILSVRC2012_img_val" 
TRAIN_DIR="/mnt/d/Dataset/ILSVRC2012_img_train"

# 指标输出目录
SCORE_DIR="./results/${name}/score_jit_b16"
mkdir -p ${SCORE_DIR}

# 根据实验名称自动设置参数
EXTRA_ARGS=""
if [[ $name == *"nonlinear"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_nonlinear"
fi

if [[ $name == *"ss_rotation"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_self_supervised --ss_method rotation"
elif [[ $name == *"ss_time"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_self_supervised --ss_method time_pred"
fi

echo "Running evaluation for: $name"
echo "Extra arguments: $EXTRA_ARGS"

# 遍历所有 checkpoint 文件
# 包括 checkpoint-100.pth, checkpoint-200.pth ... 以及 checkpoint-last.pth
for ckpt in $(ls ${CKPT_ROOT}/checkpoint-*.pth | sort -V); do
    echo "=================================================="
    echo "Processing checkpoint: ${ckpt}"
    
    # 创建临时目录用于 resume (main_jit.py 默认读取目录下的 checkpoint-last.pth)
    TMP_RESUME_DIR="./tmp_resume_${name}"
    mkdir -p ${TMP_RESUME_DIR}
    ln -sf $(readlink -f $ckpt) ${TMP_RESUME_DIR}/checkpoint-last.pth

    # 启动评估
    # --num_images 建议设为 50000 以获得准确的 FID，但为了快速测试可以先设小
    torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    main_jit.py \
    --model JiT-B/16 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 130 \
    --global_batch_size 1300 \
    --gen_bsz 1024 \
    --num_images 5000 \
    --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
    --output_dir ${SCORE_DIR} \
    --resume ${TMP_RESUME_DIR} \
    --val_path ${VAL_PATH} \
    --data_path ${TRAIN_DIR} \
    --evaluate_gen \
    ${EXTRA_ARGS}

    # 清理临时软链接
    rm -rf ${TMP_RESUME_DIR}
done

echo "--------------------------------------------------"
echo "All checkpoints for $name processed."
echo "Results saved in ${SCORE_DIR}/metrics.csv"
echo "You can now use this CSV to plot your comparison curves."