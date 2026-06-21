import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .adaptive_conv_cuda.adaptive_conv import AdaptiveConv
except ImportError:
    AdaptiveConv = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import time

# 傅里叶神经算子层
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 傅里叶模态数量
        self.modes2 = modes2
        
        # 傅里叶空间的权重矩阵
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,2,  dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,2, dtype=torch.float32)
        )

    def compl_mul2d(self, input, weights):
        # 复数乘法
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 计算傅里叶变换，直接得到复数张量
        x_ft = torch.fft.rfft2(x, norm="ortho")
        
        # 将权重转换为复数张量
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)
        
        # 在傅里叶域相乘（只使用低频模态）
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], weights2
        )
        
        # 逆傅里叶变换，直接传入复数张量
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x

# FNO块
class FNOBlock(nn.Module):
    def __init__(self, channels, modes1, modes2):
        super(FNOBlock, self).__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        return self.activation(x1 + x2)

# 分辨率不变的FNO模型
class ResolutionInvariantFNO(nn.Module):
    def __init__(self, input_channels=1024, output_channels=3, width=64, modes=16):
        super(ResolutionInvariantFNO, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.width = width
        self.modes = modes
        
        # 输入提升：将1024通道映射到更小的特征空间
        self.input_lift = nn.Sequential(
            nn.Conv2d(input_channels, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1)
        )
        
        # FNO块序列
        self.fno_blocks = nn.Sequential(
            FNOBlock(width, modes, modes),
            FNOBlock(width, modes, modes),
            FNOBlock(width, modes, modes),
            FNOBlock(width, modes, modes)
        )
        
        # 输出投影：映射到RGB图像
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width//2, 1),
            nn.GELU(),
            nn.Conv2d(width//2, output_channels, 1),
            nn.Tanh()  # 输出在[-1, 1]范围
        )
        
    def forward(self, x):
        # x形状: [batch, channels, height, width]
        # 输入提升
        x = self.input_lift(x)
        
        # FNO块
        x = self.fno_blocks(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x

# 生成模拟数据
def generate_simulation_data(batch_size, resolution, feature_channels=1024):
    """
    生成模拟的训练数据
    在实际应用中，这里应该替换为真实的数据加载逻辑
    """
    # 生成随机特征
    features = torch.randn(batch_size, feature_channels, resolution, resolution)
    
    # 模拟从特征生成图像的过程
    # 这里使用一个简单的转换：特征的平均值作为基础，加上一些噪声和模式
    images = torch.zeros(batch_size, 3, resolution, resolution)
    
    for i in range(batch_size):
        # 使用特征的前几个通道生成简单的RGB模式
        r_channel = torch.mean(features[i, :64, :, :], dim=0).unsqueeze(0)
        g_channel = torch.mean(features[i, 64:128, :, :], dim=0).unsqueeze(0)
        b_channel = torch.mean(features[i, 128:192, :, :], dim=0).unsqueeze(0)
        
        # 归一化到[-1, 1]
        rgb = torch.cat([r_channel, g_channel, b_channel], dim=0)
        rgb = (rgb - rgb.mean()) / (rgb.std() + 1e-8)
        rgb = torch.tanh(rgb)  # 确保在[-1, 1]范围内
        
        images[i] = rgb
    
    return features, images

# 训练函数
def train_model():
    # 参数设置
    batch_size = 8
    epochs = 100
    learning_rate = 1e-4
    train_resolution = 16  # 训练时分辨率
    test_resolution = 256  # 测试时分辨率
    
    # 创建模型
    model = ResolutionInvariantFNO(
        input_channels=1024,
        output_channels=3,
        width=64,
        modes=8  # 对于16x16输入，模态数不能太大
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 训练循环
    train_losses = []
    
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        
        # 生成训练数据（16x16分辨率）
        features, targets = generate_simulation_data(batch_size, train_resolution)
        
        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
            
            # 每50个epoch可视化一次训练结果
            if epoch % 50 == 0:
                visualize_results(features[0], targets[0], outputs[0], 
                                 f"Training - Epoch {epoch}", train_resolution)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    return model

# 测试分辨率不变性
def test_resolution_invariance(model):
    print("\n测试分辨率不变性...")
    model.eval()
    
    # 测试分辨率
    test_resolution = 256
    
    with torch.no_grad():
        # 生成高分辨率测试数据
        features_hr, targets_hr = generate_simulation_data(1, test_resolution)
        
        # 记录推理时间
        start_time = time.time()
        
        # 使用相同模型进行推理
        outputs_hr = model(features_hr)
        
        inference_time = time.time() - start_time
        
        # 可视化高分辨率结果
        visualize_results(features_hr[0], targets_hr[0], outputs_hr[0], 
                         f"High-Resolution Test (256x256)\nInference Time: {inference_time:.3f}s", 
                         test_resolution)
        
        # 计算PSNR
        mse = F.mse_loss(outputs_hr, targets_hr)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # 因为数据范围是[-1, 1]，最大差异为2
        print(f"High-Resolution PSNR: {psnr.item():.2f} dB")
        
        return outputs_hr

# 可视化结果
def visualize_results(feature, target, output, title, resolution):
    """
    可视化特征、目标图像和预测图像
    """
    # 将张量转换为numpy数组
    feature_np = feature.cpu().numpy()
    target_np = target.permute(1, 2, 0).cpu().numpy()
    output_np = output.permute(1, 2, 0).cpu().numpy()
    
    # 从特征中提取一些有代表性的通道进行可视化
    feature_viz = np.zeros((resolution, resolution, 3))
    feature_viz[:, :, 0] = np.mean(feature_np[:64], axis=0)  # 前64个通道的平均值作为红色
    feature_viz[:, :, 1] = np.mean(feature_np[64:128], axis=0)  # 接下来的64个通道作为绿色
    feature_viz[:, :, 2] = np.mean(feature_np[128:192], axis=0)  # 再接下来的64个通道作为蓝色
    
    # 归一化特征可视化
    feature_viz = (feature_viz - feature_viz.min()) / (feature_viz.max() - feature_viz.min())
    
    # 反归一化图像（从[-1, 1]到[0, 1]）
    target_viz = (target_np + 1) / 2
    output_viz = (output_np + 1) / 2
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 特征可视化
    axes[0].imshow(feature_viz)
    axes[0].set_title(f'Input Features\n({resolution}x{resolution}x1024)')
    axes[0].axis('off')
    
    # 目标图像
    axes[1].imshow(target_viz)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 预测图像
    axes[2].imshow(output_viz)
    axes[2].set_title('FNO Prediction')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    print("FNO分辨率不变性Demo")
    print("=" * 50)
    
    # 训练模型
    model = train_model()
    
    # 测试分辨率不变性
    test_resolution_invariance(model)
    
    # 保存模型
    torch.save(model.state_dict(), 'fno_resolution_invariant.pth')
    print("\n模型已保存为 'fno_resolution_invariant.pth'")

if __name__ == "__main__":
    main()