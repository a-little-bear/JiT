import torch
import torchvision

print(f"Torch Version: {torch.__version__}")
try:
    # 尝试在 5090 上分配显存并运行一个算子
    a = torch.randn(1, 3, 224, 224).cuda()
    b = torch.nn.functional.conv2d(a, torch.randn(3, 3, 3, 3).cuda())
    print("✅ 5090 (sm_120) 算子运行成功！环境已修复。")
except Exception as e:
    print(f"❌ 运行失败: {e}")