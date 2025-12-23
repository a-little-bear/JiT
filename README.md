Changes to the Original JiT Code:
- Parameters matched single 5090 GPU (32G)
- added eval_official.sh, train.sh
- supported gradiant accumulation 
- supported 非线性瓶颈设计
- supported 是否开启自监督信号
    - 自监督方法选择: 时间步预测 或 旋转预测
    - 自监督损失权重
- Changed FID calculation to be based on the validation set of 2012 ImageNet