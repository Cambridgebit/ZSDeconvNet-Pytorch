# 项目结构说明

本仓库围绕两个互相独立但可以串联使用的阶段组织：

1. N2V2 风格自监督去噪。
2. PSF 重投影约束下的反卷积 / 2x 超分辨率。

## 根目录入口脚本

- `train_n2v2.py`：训练自监督去噪模型。输入可以是一张 TIFF，也可以是一个 TIFF 目录。
- `infer_n2v2.py`：使用 N2V2 checkpoint 对单图或目录批量去噪。
- `train.py`：单图反卷积训练，主要用于快速实验和调参。
- `train_deconv_multi.py`：多图反卷积训练，推荐用于 `datasets_full` 或 N2V2 去噪后的数据。
- `infer.py`：反卷积推理。支持 rolling-ball 风格背景扣除，并会额外保存 `_rb{radius}` 结果。

根目录入口脚本尽量保持稳定，因为它们是最常用的命令行接口。

## `zsdeconv/` 包

- `zsdeconv/models.py`：反卷积 UNet、细节残差分支、gate 等。
- `zsdeconv/losses.py`：PSF 重投影 loss、Hessian、TV、梯度重投影辅助函数。
- `zsdeconv/data.py`：单图反卷积 patch 数据集。
- `zsdeconv/n2v2.py`：N2V2 数据集、N2V2 UNet、masked L1 loss。
- `zsdeconv/psf.py`：PSF 加载、裁剪、按和归一化。
- `zsdeconv/utils.py`：TIFF 读取、归一化、padding、保存等工具。
- `zsdeconv/loss.py`：兼容旧导入的 loss 转发文件。

## 数据目录

- `datasets/`：小型样例数据、当前常用 PSF、快速实验输入。
- `datasets_full/`：完整数据集，例如 `datasets_full/Microtubule/train_data`。

## 输出目录

- `runs/`：所有训练结果、checkpoint、loss.csv、推理 TIFF、背景扣除 TIFF。
- `output/`：旧实验可能使用的输出目录，新实验建议使用 `runs/`。

## 工具目录

- `tools/visualize_tif_3d.py`：把 TIFF 像素值作为高度做 3D surface 可视化。
- `tools/visualize_denoise_residuals.py`：对比原图和 N2V2 去噪图的残差，用于判断去噪是否误删结构。

## 文档目录

- `AGENTS.md`：开发约定和项目概览。
- `MEMRARY.md`：长期记忆，记录实验结论、推荐命令和注意事项。
- `docs/PROJECT_STRUCTURE.md`：当前文件，即项目结构说明。

## 建议开发原则

- 可复用逻辑放入 `zsdeconv/`。
- 单次实验工具放入 `tools/`。
- 新实验输出放入 `runs/`，不要散落在根目录。
- 不要轻易删除历史 checkpoint；先通过新 `runs/<experiment_name>` 记录新实验。
- 先保持去噪和反卷积独立，再考虑联合训练。
