# MEMRARY.md

这是本项目的长期记忆文档。每次实验结论、默认命令、重要判断发生变化时，都应该更新这里。

## 当前目标

构建一个稳定的两阶段显微图像处理流程：

1. 在 `datasets_full/Microtubule/train_data` 上训练 N2V2 风格自监督去噪模型。
2. 使用去噪后的图像训练多图反卷积模型，减少单图 zero-shot 训练带来的伪影。
3. 推理时可选 rolling-ball 背景扣除，用于压低缓慢变化背景。

## 当前数据

- 完整微管训练集：`datasets_full/Microtubule/train_data`
- N2V2 去噪输出目录：`runs/n2v2_microtubule/denoised`
- 小样例图像：`datasets/RawSIMData_level_09.tif`
- 低信噪比样例：`datasets/RawSIMData_level_01.tif`
- 当前 PSF：`datasets/psf_emLambda525_dxy0.0313_NA1.3.tif`

PSF 处理原则：

```text
PSF 只做按和归一化：psf /= psf.sum()
不要对 PSF 做 min-max 或 percentile normalization
```

## N2V2 去噪机制

相关文件：

- `zsdeconv/n2v2.py`
- `train_n2v2.py`
- `infer_n2v2.py`

训练数据不需要干净标签。每次训练随机选择一张 TIFF，裁剪 patch，然后随机 mask 一小部分像素。

流程：

```text
clean/noisy raw patch
-> 随机选择 masked pixels
-> 用邻域随机像素替换这些位置
-> 网络输入 corrupted patch
-> 网络输出 denoised patch
-> 只在 masked pixels 上计算 L1 loss
```

因为被预测像素本身被替换掉，网络只能根据邻域上下文预测中心像素。随机噪声不稳定，因此不容易被学习；连续微管结构有空间相关性，因此更容易保留。

当前 N2V2 模型：

- UNet 结构。
- 使用 `AvgPool2d` 下采样。
- 使用 `bilinear interpolate` 上采样。
- 避免转置卷积导致的棋盘格伪影。

推荐训练命令：

```powershell
.\.venv\Scripts\python.exe train_n2v2.py `
  --data datasets_full\Microtubule\train_data `
  --out-dir runs\n2v2_microtubule `
  --epochs 80 `
  --batch-size 16 `
  --patch-size 128 `
  --n-samples 4000 `
  --mask-ratio 0.02 `
  --neighborhood 5 `
  --base-ch 48 `
  --depth 3 `
  --lr 1e-4 `
  --lr-step 20 `
  --lr-gamma 0.5 `
  --grad-clip 1.0
```

批量去噪命令：

```powershell
.\.venv\Scripts\python.exe infer_n2v2.py `
  --image datasets_full\Microtubule\train_data `
  --checkpoint runs\n2v2_microtubule\best_n2v2.pt `
  --output runs\n2v2_microtubule\denoised
```

## N2V2 当前观察

残差可视化工具：

```powershell
.\.venv\Scripts\python.exe tools\visualize_denoise_residuals.py --stem 01
```

观察到：

- 残差图中有微管结构。
- 主要原因是原本亮的微管峰值在 N2V2 输出中被压低。
- 结构本身仍然保留，但亮度峰值和横截面形状被改变。

这说明 N2V2 不只是去随机噪声，也在做一定的亮结构强度压缩。对普通视觉去噪可能可以接受，但对反卷积不一定最优，因为反卷积依赖峰值和高频边缘。

可尝试的改进：

- 缩短 N2V2 训练，例如 `epochs=30~40`。
- 降低 `mask_ratio`，例如 `0.01`。
- 降低模型容量，例如 `base_ch=32`。
- 用保峰融合输入反卷积：

```text
deconv_input = denoised + beta * max(raw - denoised, 0)
```

或：

```text
deconv_input = alpha * raw + (1 - alpha) * denoised
```

## 反卷积机制

相关文件：

- `zsdeconv/models.py`
- `zsdeconv/losses.py`
- `train.py`
- `train_deconv_multi.py`
- `infer.py`

反卷积模型 `DirectDeconvNet` 是 UNet 风格网络。训练不依赖清晰 GT，而依赖 PSF 重投影：

```text
输入图像 patch
-> 网络输出清晰/超分辨结果 pred
-> 用 PSF 卷积 pred
-> resize 到输入尺寸
-> 与输入图像 patch 计算 L1 loss
```

含义：

```text
如果 pred 是合理的清晰结构，那么经过显微系统 PSF 模糊后，应该能回到观测图像。
```

常用正则：

- `hess_w`：Hessian 正则，抑制振铃和局部粗糙伪影。
- `tv_w`：TV 正则，增强连续性，但过大会抹细节。
- `detail_branch`：残差高频分支，可以变细，但可能产生双边复制影。
- `detail_l1_w`：限制残差分支幅度，减少假高频。

## 单图反卷积结论

notebook 风格模型：

- 结构真实性最好。
- 微管连续性较好。
- 但偏平滑，不够细。

detail residual 分支：

- 能保留更多高频。
- 但容易出现主微管两侧的复制影和 speckle。
- gate 和 detail L1 可以缓解，但不能完全解决残差分支学到伪高频的问题。

HR PSF forward：

- 对 2x 输出更符合物理采样尺度。
- 能让结构更细。
- 但在单图 zero-shot 下容易产生伪影和微管断裂。

当前判断：

```text
单图训练中，细节越强，伪影风险越高。
多图训练应比单图更稳定，因为伪影不容易跨图泛化。
```

## 多图反卷积训练

推荐先用 N2V2 去噪后的目录训练：

```powershell
.\.venv\Scripts\python.exe train_deconv_multi.py `
  --data runs\n2v2_microtubule\denoised `
  --psf datasets\psf_emLambda525_dxy0.0313_NA1.3.tif `
  --out-dir runs\deconv_multi_n2v2_40ep `
  --epochs 40 `
  --batch-size 16 `
  --patch-size 128 `
  --n-samples 4000 `
  --lr 1e-4 `
  --lr-step 10 `
  --lr-gamma 0.5 `
  --hess-w 0.08 `
  --tv-w 0.015 `
  --l1-w 0.0 `
  --noise-std 0.02 `
  --psf-size 25 `
  --psf-center image `
  --base-ch 32 `
  --depth 4 `
  --n-conv 3 `
  --detail-scale 0.12 `
  --detail-l1-w 0.01 `
  --grad-reproj-w 0.0 `
  --grad-clip 1.0 `
  --lr-psf-forward
```

如果训练太慢，可先用快速版本：

```powershell
.\.venv\Scripts\python.exe train_deconv_multi.py `
  --data runs\n2v2_microtubule\denoised `
  --psf datasets\psf_emLambda525_dxy0.0313_NA1.3.tif `
  --out-dir runs\deconv_multi_n2v2_fast `
  --epochs 20 `
  --batch-size 8 `
  --patch-size 96 `
  --n-samples 1000 `
  --hess-w 0.08 `
  --tv-w 0.015 `
  --noise-std 0.02 `
  --detail-scale 0.08 `
  --detail-l1-w 0.01 `
  --lr-psf-forward
```

## 推理命令

先 N2V2 去噪，再反卷积：

```powershell
.\.venv\Scripts\python.exe infer_n2v2.py `
  --image datasets\RawSIMData_level_01.tif `
  --checkpoint runs\n2v2_microtubule\best_n2v2.pt `
  --output runs\n2v2_microtubule\RawSIMData_level_01_n2v2.tif
```

```powershell
.\.venv\Scripts\python.exe infer.py `
  --image runs\n2v2_microtubule\RawSIMData_level_01_n2v2.tif `
  --checkpoint runs\deconv_multi_n2v2_40ep\best_deconv.pt `
  --output runs\deconv_multi_n2v2_40ep\RawSIMData_level_01_n2v2_deconv.tif `
  --rolling-ball-radius 30
```

`infer.py` 会保存两份：

- 普通反卷积结果。
- `_rb30` rolling-ball 背景扣除结果。

## 实用工具

查看去噪残差是否包含结构：

```powershell
.\.venv\Scripts\python.exe tools\visualize_denoise_residuals.py --stem 01
```

TIFF 3D 高度图可视化：

```powershell
.\.venv\Scripts\python.exe tools\visualize_tif_3d.py runs\detail_balanced_40ep\result.tif --stride 4 --height-scale 80
```

## 待解决问题

- 如何让 N2V2 去噪保留亮微管峰值。
- 是否使用保峰融合图训练反卷积。
- 多图反卷积是否能明显降低单图 residual branch 伪影。
- LR PSF forward 和 HR PSF forward 哪个更适合最终结果。
