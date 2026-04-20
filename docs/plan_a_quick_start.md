# 方案A快速开始指南

## 🎯 你选择了方案A！

**恭喜！** 这是最适合本科毕业设计的方案：
- ✅ 技术成熟，易于实现
- ✅ 准确率有保证（95%+）
- ✅ 3-4周可完成
- ✅ 论文评分良好-优秀

---

## 📚 你已经拥有的文档

### 1. 需求文档
📄 `.kiro/specs/fault-diagnosis-system/requirements.md`
- 完整的项目需求
- 用户故事和验收标准

### 2. 设计文档 ⭐ 重点
📄 `.kiro/specs/fault-diagnosis-system/design.md`
- 详细的模型架构设计
- 多尺度CNN + 注意力机制
- 训练策略和评估方案

### 3. 任务列表 ⭐ 执行指南
📄 `.kiro/specs/fault-diagnosis-system/tasks.md`
- 18个主要任务
- 4周详细时间表
- 每日任务建议

### 4. 最新论文汇总
📄 `docs/latest_papers_2023_2025.md`
- 2024-2025年最新论文
- 方案A参考论文详解

---

## 🚀 立即开始（今天就做）

### Step 1: 安装环境（30分钟）

```bash
# 1. 创建虚拟环境
conda create -n fault_diagnosis python=3.8
conda activate fault_diagnosis

# 2. 安装PyTorch（GPU版本，如果有GPU）
# 访问 https://pytorch.org/ 选择合适版本
# 例如：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或CPU版本：
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 3. 安装其他依赖
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install tensorboard
pip install tqdm

# 4. 测试安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 2: 获取数据（30分钟）

**方法1: 克隆现成代码（推荐）**
```bash
git clone https://github.com/Tan-Qiyu/Deep_Learning_For_Fault_Diagnosis
cd Deep_Learning_For_Fault_Diagnosis
```

**方法2: 从官网下载**
1. 访问: https://engineering.case.edu/bearingdatacenter
2. 下载以下文件：
   - Normal Baseline Data (正常数据)
   - 12k Drive End Bearing Fault Data (故障数据)
3. 解压到 `data/raw/` 目录

### Step 3: 运行示例代码（10分钟）

```bash
# 如果使用方法1克隆的代码
cd Deep_Learning_For_Fault_Diagnosis
python train.py

# 查看结果
tensorboard --logdir=logs
```

---

## 📖 今天的学习任务

### 任务1: 理解项目结构（1小时）

阅读以下文档：
1. ✅ 设计文档 - 了解模型架构
2. ✅ 任务列表 - 了解工作计划

### 任务2: 熟悉数据（1小时）

运行以下代码，理解CWRU数据：

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 加载一个样本文件
data = loadmat('path/to/data.mat')
signal = data['X097_DE_time']  # 驱动端加速度信号

# 查看信号形状
print(f"信号形状: {signal.shape}")
print(f"信号长度: {len(signal)}")
print(f"采样点数: {signal.shape[0]}")

# 可视化时域波形
plt.figure(figsize=(12, 4))
plt.plot(signal[:2000])  # 绘制前2000个点
plt.title('Bearing Vibration Signal - Time Domain')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('signal_time_domain.png')
plt.show()

# 可视化频域
from scipy.fft import fft, fftfreq
N = len(signal)
yf = fft(signal.flatten())
xf = fftfreq(N, 1/12000)[:N//2]  # 采样频率12kHz

plt.figure(figsize=(12, 4))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title('Bearing Vibration Signal - Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(0, 5000)  # 只显示0-5kHz
plt.savefig('signal_frequency_domain.png')
plt.show()
```

### 任务3: 理解模型架构（1小时）

阅读设计文档中的"模型架构详细设计"部分，理解：
1. 多尺度卷积的作用
2. 注意力机制的原理
3. 完整网络的数据流

---

## 📅 本周计划（Week 1）

### Day 1（今天）
- [x] 阅读所有文档
- [ ] 安装环境
- [ ] 获取数据
- [ ] 运行示例代码

### Day 2
- [ ] 实现数据加载函数
- [ ] 实现信号分割
- [ ] 测试数据预处理

### Day 3
- [ ] 实现归一化
- [ ] 构建Dataset类
- [ ] 构建DataLoader

### Day 4
- [ ] 数据可视化（时域）
- [ ] 数据可视化（频域）
- [ ] 统计分析

### Day 5
- [ ] 实现简单CNN（Baseline）
- [ ] 测试模型前向传播

### Day 6
- [ ] 训练Baseline模型
- [ ] 达到85%+准确率

### Day 7
- [ ] 复习本周内容
- [ ] 整理代码和笔记
- [ ] 准备下周工作

---

## 💻 核心代码框架

### 1. 数据加载器

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat

class CWRUDataset(Dataset):
    """CWRU轴承数据集"""
    
    def __init__(self, data_dir, window_size=1024, overlap=512, transform=None):
        """
        Args:
            data_dir: 数据目录
            window_size: 窗口大小
            overlap: 重叠大小
            transform: 数据变换
        """
        self.data = []
        self.labels = []
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        
        # 加载数据
        self._load_data(data_dir)
        
    def _load_data(self, data_dir):
        """加载所有数据文件"""
        # TODO: 实现数据加载逻辑
        pass
    
    def _sliding_window(self, signal):
        """滑动窗口分割信号"""
        stride = self.window_size - self.overlap
        samples = []
        for i in range(0, len(signal) - self.window_size, stride):
            samples.append(signal[i:i+self.window_size])
        return np.array(samples)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # 转换为Tensor，添加通道维度
        sample = torch.FloatTensor(sample).unsqueeze(0)
        return sample, label

# 使用示例
train_dataset = CWRUDataset('data/train')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### 2. 多尺度CNN模型

```python
import torch.nn as nn

class MultiScaleCNN(nn.Module):
    """多尺度CNN + 注意力机制"""
    
    def __init__(self, num_classes=10):
        super(MultiScaleCNN, self).__init__()
        
        # 多尺度卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=128, stride=1, padding=64),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=256, stride=1, padding=128),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 通道注意力
        self.attention = ChannelAttention(96, reduction=16)
        
        # 深层特征提取
        self.features = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 多尺度特征提取
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        # 特征拼接
        x = torch.cat([x1, x2, x3], dim=1)
        
        # 通道注意力
        x = self.attention(x) * x
        
        # 深层特征
        x = self.features(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        return x

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y
```

### 3. 训练循环

```python
def train_model(model, train_loader, val_loader, num_epochs=50):
    """训练模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 学习率调度
        scheduler.step()
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved! Val Acc: {val_acc:.2f}%')
    
    return model
```

---

## 📊 预期结果

### Week 1结束时，你应该有：
- ✅ 完整的开发环境
- ✅ CWRU数据集
- ✅ 数据预处理代码
- ✅ 数据可视化图表
- ✅ Baseline模型（85%+准确率）

### Week 2结束时，你应该有：
- ✅ 多尺度CNN+注意力模型
- ✅ 完整的训练代码
- ✅ 模型测试通过

### Week 3结束时，你应该有：
- ✅ 完整模型训练完成（95%+）
- ✅ 对比实验完成
- ✅ 所有图表生成

### Week 4结束时，你应该有：
- ✅ 论文初稿
- ✅ 答辩PPT
- ✅ 演示Demo

---

## 🆘 遇到问题怎么办？

### 常见问题

#### Q1: 环境安装失败
**A**: 
- 检查Python版本（需要3.8）
- 使用清华镜像源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 参考PyTorch官网选择正确版本

#### Q2: 数据下载太慢
**A**:
- 使用方法1克隆GitHub代码（已包含预处理数据）
- 或者找同学分享数据

#### Q3: 没有GPU怎么办
**A**:
- 使用Google Colab免费GPU
- 或者减小batch_size在CPU上训练（会慢一些）

#### Q4: 代码报错
**A**:
- 仔细阅读错误信息
- 检查数据维度是否正确
- 参考GitHub开源代码
- 搜索Stack Overflow

---

## ✅ 今天的检查清单

完成以下任务，你就成功开始了：

- [ ] 阅读完所有文档
- [ ] 安装好开发环境
- [ ] 测试PyTorch是否正常
- [ ] 获取CWRU数据集
- [ ] 运行示例代码（如果有）
- [ ] 理解项目整体结构
- [ ] 制定明天的学习计划

---

## 🎉 恭喜你选择了方案A！

这是一个：
- ✅ 技术成熟的方案
- ✅ 容易实现的方案
- ✅ 效果有保证的方案
- ✅ 适合本科生的方案

**下一步**: 开始任务1.1 - 安装Anaconda

加油！你一定可以顺利完成毕业设计！🚀
