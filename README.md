Changes to the Original JiT Code:
- Parameters matched single 5090 GPU (32G)
- added eval_official.sh, train.sh
- supported gradiant accumulation 
- supported 非线性瓶颈设计
- supported 是否开启自监督信号
    - 自监督方法选择: 时间步预测 或 旋转预测
    - 自监督损失权重
- Changed FID calculation to be based on the validation set of 2012 ImageNet



AutoDL 安装 ImageNet

```py
# 1. 创建目标文件夹
mkdir -p /root/autodl-tmp/imagenet/train && mkdir -p /root/autodl-tmp/imagenet/val

# 2. 解压验证集 (较快，建议先试这个)
tar -xf /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar -C /root/autodl-tmp/imagenet/val

# 3. 解压训练集 (耗时较长，约 20-40 分钟)
tar -xf /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar -C /root/autodl-tmp/imagenet/train
```





```py
# 先卸载清理（必做）
pip uninstall torch torchvision torchaudio -y

# 强制安装 20251203 的 cu128 版本
# 注意：这里把 cu126 换成了 cu128
pip install torch==2.10.0.dev20251203+cu128 torchvision==0.25.0.dev20251203+cu128 torchaudio==2.10.0.dev20251203+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```

