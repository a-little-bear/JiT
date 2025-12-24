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
```

安装Repo

```
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Repo
!git clone https://github.com/a-little-bear/JiT.git

%cd /content/drive/MyDrive/Repo/JiT
```

Notebook：

复制Repo

```py
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/drive/MyDrive/Repo/JiT /content/JiT
%cd /cont
```



安装依赖

```py
# 1. 安装 YAML 中指定的第三方库
# 注意：我们跳过了 pytorch 和 cuda，因为 Colab 已经内置了完全相同的版本
!pip install numpy==1.22 \
             opencv-python==4.11.0.86 \
             timm==0.9.12 \
             tensorboard==2.10.0 \
             scipy==1.9.1 \
             einops==0.8.1 \
             gdown==5.2.0

# 2. 安装来自 GitHub 的特定依赖
!pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity

# 3. 验证关键版本（可选）
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import timm
import einops
import torch_fidelity
print("所有依赖已成功加载！")
```





# AutoDL 安装 ImageNet

```py
# 1. 创建目标文件夹
mkdir -p /root/autodl-tmp/imagenet/train && mkdir -p /root/autodl-tmp/imagenet/val

# 2. 解压验证集 (较快，建议先试这个)
tar -xf /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar -C /root/autodl-tmp/imagenet/val

# 3. 解压训练集 (耗时较长，约 20-40 分钟)
tar -xf /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar -C /root/autodl-tmp/imagenet/train
```

clone and install

```py
rm -f ~/.condarc
cat <<EOF > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-nightly: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda clean -i



git clone --depth=1 https://gitclone.com/github.com/a-little-bear/JiT.git

conda env create -f environment.yaml
conda activate jit
```

```py
# 先卸载清理（必做）
pip uninstall torch torchvision torchaudio -y

# 强制安装 20251203 的 cu128 版本
# 注意：这里把 cu126 换成了 cu128
pip install torch==2.10.0.dev20251203+cu128 torchvision==0.25.0.dev20251203+cu128 torchaudio==2.10.0.dev20251203+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```

