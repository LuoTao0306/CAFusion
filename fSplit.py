"该部分代码的功能是分离源图像的高频和低频信息"
#1、计算梯度；2、计算密度；3、自适应分离高低频信息
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SobelOperator(nn.Module):
    """Sobel算子模块，用于计算图像梯度"""

    def __init__(self):
        super(SobelOperator, self).__init__()
        # 定义Sobel卷积核
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 注册为不可训练的参数
        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x):
        # 如果输入是RGB图像，转换为灰度图
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # 计算x和y方向的梯度
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # 计算梯度幅值
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # 归一化到0-1范围
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        return x, grad_mag  # x是灰度图，grad_mag是梯度幅值


class EdgeDensityCalculator(nn.Module):
    """边缘密度计算模块"""

    def __init__(self, threshold=0.2):
        super(EdgeDensityCalculator, self).__init__()
        self.threshold = threshold

    def forward(self, grad_mag):
        # 二值化梯度图像
        edge_mask = (grad_mag > self.threshold).float()

        # 计算边缘密度
        edge_pixels = torch.sum(edge_mask, dim=[1, 2, 3])
        total_pixels = torch.numel(edge_mask[0])
        edge_density = edge_pixels / total_pixels

        return edge_density, edge_mask


class AdaptiveCutoffFrequency(nn.Module):
    """自适应截止频率计算模块"""

    def __init__(self, alpha=0.5, f0=0.05, min_freq=0.01, max_freq=0.5):
        super(AdaptiveCutoffFrequency, self).__init__()
        self.alpha = alpha
        self.f0 = f0
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, edge_density):
        # 计算截止频率
        cutoff_freq = self.alpha * edge_density + self.f0
        # 限制范围
        cutoff_freq = torch.clamp(cutoff_freq, self.min_freq, self.max_freq)
        return cutoff_freq


class FourierHighLowSplit(nn.Module):
    """傅里叶变换高低频分离模块"""

    def __init__(self):
        super(FourierHighLowSplit, self).__init__()

    def forward(self, image, cutoff_freq):
        # 确保输入是单通道灰度图
        if image.size(1) > 1:
            image = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]

        batch_size, _, height, width = image.size()

        # 傅里叶变换
        fft = torch.fft.fft2(image)
        fft_shifted = torch.fft.fftshift(fft)

        # 创建高斯低通滤波器
        x = torch.linspace(-0.5, 0.5, width, device=image.device)
        y = torch.linspace(-0.5, 0.5, height, device=image.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        distance = torch.sqrt(xx ** 2 + yy ** 2)

        # 为批次中的每个样本创建滤波器
        low_pass = torch.exp(-(distance ** 2) / (2 * (cutoff_freq[:, None, None] ** 2)))
        low_pass = low_pass.unsqueeze(1)  # 增加通道维度
        high_pass = 1 - low_pass

        # 应用滤波器
        fft_low = fft_shifted * low_pass
        fft_high = fft_shifted * high_pass

        # 逆傅里叶变换
        low_freq = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_low)))
        high_freq = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_high)))

        # 归一化到0-1范围
        low_freq = (low_freq - low_freq.min()) / (low_freq.max() - low_freq.min() + 1e-8)
        high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)

        return low_freq, high_freq



class AutoFFTModule(nn.Module):
    """自适应傅里叶变换模块，整合所有子模块"""

    def __init__(self, threshold=0.2, alpha=0.5, f0=0.05):
        super(AutoFFTModule, self).__init__()
        self.sobel = SobelOperator()
        self.edge_density = EdgeDensityCalculator(threshold)
        self.adaptive_cutoff = AdaptiveCutoffFrequency(alpha, f0)
        self.fourier_split = FourierHighLowSplit()

    def forward(self, x):
        # 1. 计算梯度
        gray, gradient = self.sobel(x)

        # 2. 计算边缘密度
        edge_density, edge_mask = self.edge_density(gradient)

        # 3. 计算自适应截止频率
        cutoff_freq = self.adaptive_cutoff(edge_density)

        # 4. 傅里叶变换高低频分离
        low_freq, high_freq = self.fourier_split(gray, cutoff_freq)

        # 返回结果以及中间变量（方便调试和可视化）
        return low_freq,high_freq
