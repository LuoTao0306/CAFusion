import torch
from thop import profile
from torchsummary import summary
import twoDomainModel as  Model # 请替换为你的模型文件路径

# 初始化模型
model = Model.Model()

# 定义输入张量 (batch_size, channels, height, width)
# 根据实际情况调整输入尺寸
input_tensor = torch.randn(1, 2, 256, 256)  # 假设输入是256x256的图像

# 1. 使用torchsummary查看模型结构和参数数量
# print("模型结构和参数信息：")
# summary(model, input_size=(2, 256, 256), device='cpu')  # 输入尺寸为 (channels, height, width)

# 2. 使用thop计算FLOPs和参数数量
print("\n计算FLOPs和参数数量：")
flops, params = profile(model, inputs=(input_tensor,))

# 格式化输出
print(f"FLOPs: {flops / 1e9:.2f} G")  # 转换为GigaFLOPs
print(f"参数数量: {params / 1e6} M")  # 转换为Million Parameters