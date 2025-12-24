Changes to the Original JiT Code:
- Parameters matched single 5090 GPU (32G)
- added eval_official.sh, train.sh
- supported gradiant accumulation 
- supported 非线性瓶颈设计
- supported 是否开启自监督信号
    - 自监督方法选择: 时间步预测 或 旋转预测
    - 自监督损失权重
- Changed FID calculation to be based on the validation set of 2012 ImageNet

# Google Colab

## 数据下载

下载 ImageNet

```py
from google.colab import drive
drive.mount('/content/drive')
```

```py
# 进入 Drive 目录
%cd /content/drive/MyDrive/Dataset/ImageNet2012

# 下载训练集
!wget -c "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" -O ILSVRC2012_img_train.tar

# 下载验证集
!wget -c "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" -O ILSVRC2012_img_val.tar

!ls -lh /content/drive/MyDrive/Dataset/ImageNet2012

drive.flush_and_unmount()
```

## 环境配置

安装Repo

```
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/a-little-bear/JiT.git
!pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity

# 重启notebook应用torch fidelity
```

```
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import timm
import einops
import torch_fidelity
print("所有依赖已成功加载！")
```

## 数据处理

```
# 定义路径
TRAIN_TAR = '/content/drive/MyDrive/Dataset/ImageNet2012/ILSVRC2012_img_train.tar'
VAL_TAR = '/content/drive/MyDrive/Dataset/ImageNet2012/ILSVRC2012_img_val.tar'
GT_FILE = '/content/JiT/ILSVRC2012_validation_ground_truth.txt'

# 创建本地存储目录（Colab 根目录下，不会改动 Drive 原文件）
!mkdir -p /content/imagenet/train /content/imagenet/val /content/output
```

```
%%bash
# 1. 解压验证集到本地 val 文件夹
# tar -xf /content/drive/MyDrive/Dataset/ImageNet2012/ILSVRC2012_img_val.tar -C /content/imagenet/val

# 2. 下载处理脚本（ImageNet 官方常用的整理脚本，或使用下面这段命令直接处理）
# 这里我们直接用 bash 读取 ground truth 并移动文件
cd /content/imagenet/val

# 这里的处理逻辑：读取每一行 label，创建目录并移动对应图片
# 注意：ILSVRC2012_img_val_00000001.JPEG 对应 ground truth 第一行
i=1
while read label; do
    prefix="n$(printf "%08d" $label)" # 这里需要注意：ground truth 通常是 class ID 序号，需对应 Synset ID
    # 如果你的 txt 已经是 n 开头的文件夹名，直接用 $label 即可
    mkdir -p $label
    filename=$(printf "ILSVRC2012_val_%08d.JPEG" $i)
    if [ -f "$filename" ]; then
        mv "$filename" $label/
    fi
    i=$((i+1))
done < /content/JiT/ILSVRC2012_validation_ground_truth.txt
```

```
%%bash
# 1. 解压主压缩包到本地 train 文件夹
tar -xf /content/drive/MyDrive/Dataset/ImageNet2012/ILSVRC2012_img_train.tar -C /content/imagenet/train

# 2. 循环解压内部的 1000 个子压缩包
cd /content/imagenet/train
for f in *.tar; do
    d="${f%.tar}"
    mkdir -p "$d"
    tar -xf "$f" -C "$d"
    rm "$f" # 删除本地解压出来的子 tar 包以节省 Colab 空间，不影响 Drive 里的原文件
done
```

## 运行

./sh -env 改为 colab。

```py
%cd /content/JiT
!chmod +x train_nonlinear_ss_rotation.sh
!./train_nonlinear_ss_rotation.sh
```

## 最终保存输出

```
drive.flush_and_unmount()
```





# AutoDL 

## 安装 ImageNet

AutoPanel切换成清华源。启用加速：`source /etc/network_turbo`

```py
# 1. 创建目标文件夹
mkdir -p /root/autodl-tmp/imagenet/train && mkdir -p /root/autodl-tmp/imagenet/val

# 2. 解压验证集 (较快，建议先试这个)
tar -xf /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar -C /root/autodl-tmp/imagenet/val

# 3. 解压训练集 (耗时较长，约 20-40 分钟)
tar -xf /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar -C /root/autodl-tmp/imagenet/train
```

## 依赖

```py
git clone --depth=1 https://github.com/a-little-bear/JiT.git

conda env create -f environment.yaml
conda activate jit
```

```py
# 先卸载清理（必做）
pip uninstall torch torchvision torchaudio -y

# 最新 nightly：https://download.pytorch.org/whl/nightly/torch/

pip install torch==2.11.0.dev20251222+cu128 torchvision==0.25.0.dev20251222+cu128 torchaudio==2.10.0.dev20251222+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 重新拉取

```
git fetch --all
git reset --hard new/main
git pull new main
```

然后 .sh 修改 env 为 autodl，给 .sh 加上 chmod +x xxx.sh

## 保存输出

```
# 假设输出在 /root/autodl-tmp/output
cd /root/autodl-tmp
tar -czvf output_data.tar.gz output/
```

然后在AutoPanel上传压缩包到阿里云盘
