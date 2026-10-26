# 阶段二：PyTorch 入门 + MNIST

对应「学习路线.md」的「阶段 2」。本阶段目标：**跑通第一个深度学习案例，掌握固定套路——数据怎么读、模型怎么搭、loss 怎么算、怎么迭代。**

## 目录结构

```
stage2/
├── README.md
├── ex1_tensor_autograd/     PyTorch 基础：张量 + 自动求导（对应官方 60 分钟入门）
│   ├── tensor_basics.py     创建张量、属性、运算、索引、numpy 互转、设备
│   └── autograd_demo.py     计算图、backward()、链式法则、no_grad
├── ex2_training_loop/       完整训练套路（合成数据，不需要联网下载）
│   └── linear_regression.py 模型->loss->优化器->训练循环，画 loss 曲线
└── ex3_mnist_cnn/           MNIST 手写数字识别（本阶段主项目，首次运行需联网下载数据集）
    ├── model.py             CNN 模型定义（卷积+池化+全连接）
    ├── mnist_train.py       训练 + 每轮验证 + 保存最优模型
    └── mnist_eval.py        加载模型评估 + 画 6 张预测结果图
```

## 运行顺序

```bash
conda activate deplrng
cd dsh_answer/stage2
python ex1_tensor_autograd/tensor_basics.py
python ex1_tensor_autograd/autograd_demo.py
python ex2_training_loop/linear_regression.py
cd ex3_mnist_cnn
python mnist_train.py    # 首次运行自动下载 MNIST（约 11MB），需要联网
python mnist_eval.py     # 跑完 train 之后才能跑（依赖 mnist_cnn.pt）
```

**预期结果**：
- ex2：学到的 w ≈ 2、b ≈ 1，生成 loss_curve.png，loss 曲线单调下降
- ex3：训练 5 个 epoch 后 val_acc ≈ 98-99%（路线里程碑要求 ≥ 97%）

> 如果 MNIST 下载卡住/失败，把报错发给我，我教你换国内镜像。

## 必须合上书能回答的 4 个问题（学习路线自检）

1. **为什么要归一化？** → 看 `mnist_train.py` 里 Normalize 旁边的注释
2. **`model.train()` / `model.eval()` 有什么区别？**
3. **batch 的作用是什么？** → 看 `mnist_train.py` 里 DataLoader 旁边的注释
4. **learning rate 调大/调小会怎样？** → 动手实验（下面）

## 调参小实验（推荐做一遍，练手感）

改 `mnist_train.py` 顶部的超参数，各跑一遍，记录 val_acc：

| 改动 | 预期现象 |
|---|---|
| `LR = 1e-3` → `1e-1` | loss 震荡甚至爆炸（NaN），训练崩掉 |
| `LR = 1e-3` → `1e-5` | 学得很慢，5 个 epoch 准确率明显低于正常 |
| `EPOCHS = 5` → `10` | val_acc 继续小涨，可能到 99%+ |
| `BATCH_SIZE = 64` → `16` | 每步梯度噪音变大，loss 曲线更抖 |

## 自检清单

- [ ] 能默写出训练循环核心 5 行：`forward -> loss -> zero_grad -> backward -> step`
- [ ] 能回答上面 4 个问题
- [ ] MNIST 验证集准确率 ≥ 97%
- [ ] 至少做一次调 lr 实验，并把观察记进笔记
