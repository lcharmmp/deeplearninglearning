"""
阶段二 · 练习 3a：CNN 模型定义（MNIST 手写数字识别用）

说明：
    手写数字图片是 28x28 的灰度图（1 个通道）。
    这是经典的"卷积 -> 池化 -> 卷积 -> 池化 -> 全连接"结构，尺寸推演：

      输入 (N, 1, 28, 28)
        Conv1(1->32, 3x3, padding=1)  -> (N, 32, 28, 28)   卷积提取特征，padding 保持尺寸
        ReLU + MaxPool2d(2)           -> (N, 32, 14, 14)   池化下采样，尺寸减半
        Conv2(32->64, 3x3, padding=1) -> (N, 64, 14, 14)
        ReLU + MaxPool2d(2)           -> (N, 64, 7, 7)
        Flatten                       -> (N, 64*7*7=3136)
        Linear(3136 -> 10)            -> (N, 10)           输出 10 个类别的分数

用法：
    被 mnist_train.py / mnist_eval.py 用 `from model import SimpleCNN` 引用
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分：卷积核自动学习"局部特征"（边缘、笔画等）
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),   # 输入 1 通道，输出 32 通道
            # nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 分类部分：把特征展平，全连接得到每个类别的分数
        self.classifier = nn.Sequential(
            nn.Flatten(),                               # (N, 64, 7, 7) -> (N, 3136)
            nn.Linear(64 * 7 * 7, num_classes),         # 3136 -> 10
        )

    def forward(self, x):
        return self.classifier(self.features(x))
