# AGENTS.md - 项目开发指南

## 项目概览

本项目用于显微图像的两阶段处理实验：

1. **N2V2 风格自监督去噪**：从带噪微管 TIFF 图像中学习去除随机噪声。
2. **基于 PSF 重投影的反卷积 / 2x 超分辨率**：根据显微系统 PSF，把网络输出重新卷积回原图，用这个物理一致性约束训练反卷积模型。

项目使用 Python + PyTorch。当前建议保持“去噪”和“反卷积”分开训练，不做联合训练，方便定位伪影来源。

## 快速开始

```powershell
# 训练 N2V2 去噪模型
.\.venv\Scripts\python.exe train_n2v2.py

# 对单张图去噪
.\.venv\Scripts\python.exe infer_n2v2.py --image datasets_full\Microtubule\train_data\01.tif

# 训练单图反卷积模型
.\.venv\Scripts\python.exe train.py

# 训练多图反卷积模型，通常使用 N2V2 去噪后的目录
.\.venv\Scripts\python.exe train_deconv_multi.py

# 反卷积推理
.\.venv\Scripts\python.exe infer.py
```

## 项目结构

| 路径 | 作用 |
|------|------|
| `train_n2v2.py` | 训练 N2V2 风格自监督去噪模型 |
| `infer_n2v2.py` | 对单张 TIFF 或整个目录批量去噪 |
| `train.py` | 单图反卷积训练 |
| `train_deconv_multi.py` | 多图反卷积训练 |
| `infer.py` | 反卷积推理，可选 rolling-ball 背景扣除 |
| `zsdeconv/` | 模型、loss、数据集、PSF、工具函数等共享代码 |
| `datasets/` | 小型样例数据和 PSF |
| `datasets_full/` | 完整训练数据集 |
| `runs/` | 实验输出、checkpoint、loss 记录 |
| `tools/` | 临时分析和可视化工具 |
| `docs/` | 项目说明文档 |
| `MEMRARY.md` | 长期记忆：实验结论、常用命令、注意事项 |

## 去噪模型如何工作

N2V2 去噪代码位于：

- `zsdeconv/n2v2.py`
- `train_n2v2.py`
- `infer_n2v2.py`

训练时不需要干净 GT。数据集会从多张 TIFF 中随机裁剪 patch，然后随机选择少量像素作为 blind spots：

```text
原始 patch
-> 随机选择 masked pixels
-> 用邻域随机像素替换 masked pixels，得到 corrupted input
-> 网络预测整张 patch
-> loss 只在 masked pixels 上计算
```

核心思想是：网络不能直接看到被预测像素的真实值，只能根据周围上下文估计它。因此随机噪声不容易被学到，而连续结构会被保留。

当前 N2V2 UNet 使用 `AvgPool + bilinear upsample`，避免转置卷积带来的棋盘伪影。

## 反卷积模型如何工作

反卷积模型位于：

- `zsdeconv/models.py`
- `zsdeconv/losses.py`
- `train.py`
- `train_deconv_multi.py`
- `infer.py`

模型 `DirectDeconvNet` 是 UNet 风格网络，可选 2x 上采样。训练不是直接和 GT 比较，而是使用 PSF 重投影：

```text
输入图像 patch
-> 网络输出 deconvolved / super-resolved image
-> 用 PSF 将输出重新卷积
-> 如果输出是 2x，则再 resize 回输入尺寸
-> 和输入图像比较 L1 loss
```

这个 loss 表示：

```text
如果网络输出是真实清晰结构，那么经过显微系统 PSF 模糊后，应该能回到观测图像。
```

可选正则项：

- `HessianLoss`：抑制二阶振铃和粗糙伪影。
- `TVLoss`：鼓励局部连续和平滑。
- `detail_branch`：额外预测高频残差，提高细节，但可能产生双边复制影，需要谨慎使用。
- `detail_l1_w`：只约束细节残差幅度，减少伪高频。

## 重要约定

- PSF 只能做 `psf /= psf.sum()`，不能做 min-max 或 percentile normalization。
- 显微图像默认使用 percentile normalization。
- 反卷积 checkpoint 格式：

```python
{"model": state_dict, "psf": tensor, "cfg": dict}
```

- N2V2 checkpoint 格式：

```python
{"model": state_dict, "cfg": dict}
```

- 实验输出统一放到 `runs/`。
- 去噪和反卷积默认分开训练，除非明确做联合训练实验。

## 当前注意事项

- notebook 风格反卷积前向使用 LR PSF + resize，视觉真实性较好，但物理尺度不完全严格。
- HR PSF forward 更符合 2x 输出的物理尺度，但在单图训练时容易产生锐化伪影和结构断裂。
- detail residual branch 可以保留更多高频，但也可能制造微管两侧的复制影。
- N2V2 去噪可能压低亮微管峰值；如果残差图中能看到微管结构，说明去噪不只是去噪声，还改变了结构强度。
- `infer.py --rolling-ball-radius` 会同时保存普通输出和 `_rb{radius}` 背景扣除输出。
