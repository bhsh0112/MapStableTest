<div align="center">

# 🗺️ StableHDMap

### Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping

**在线高精地图构建的时序稳定性评估基准与工具包**

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue?style=for-the-badge)](https://iclr.cc/)
[![Project Page](https://img.shields.io/badge/Project%20Page-StableHDMap-green?style=for-the-badge)](https://stablehdmap.github.io/)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-orange?style=for-the-badge)](https://openreview.net/forum?id=mxz5RqhCMe)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Hao Shan, Ruikai Li, Han Jiang, Yizhe Fan, Ziyang Yan, Bohan Li, Xiaoshuai Hao, Hao Zhao, Zhiyong Cui, Yilong Ren, Haiyang Yu*

**本仓库为论文配套的稳定性评估工具包（MapTR Stability Eval），用于复现论文中的 mAS 等稳定性指标。**

</div>

---

## 📌 简介

作为自动驾驶中的基础模块，在线高精地图（Online HD Map）因其成本效益与实时能力受到广泛关注。车辆在高度动态环境中行驶时，车载传感器的空间位移会导致实时建图结果发生漂移，**这种不稳定性对下游任务构成根本性挑战**。然而，现有在线建图模型多聚焦于提升单帧精度，**时序稳定性尚未被系统研究**。

本工作提出了**首个面向在线矢量化地图构建的时序稳定性评估基准**，包含：

- **多维度稳定性评估框架**：Presence（在场一致性）、Localization（位置稳定性）、Shape（形状稳定性）
- **统一指标 mAS**（mean Average Stability）
- **42 个模型与变体**的大规模实验表明：**精度（mAP）与稳定性（mAS）是相对独立的性能维度**
- 公开基准与本工具包，便于社区复现与扩展

本工具包支持 **PKL** 与 **NPZ** 格式的预测结果，无需重新运行建图模型即可评估稳定性，并已适配 MapTR、PivotNet、BEVMapNet、StreamMapNet 等多种模型输出。

---

## 📋 目录

- [功能特性](#-功能特性)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [稳定性指标](#-稳定性指标)
- [配置文件与数据格式](#-配置文件与数据格式)
- [可视化](#-可视化)
- [引用](#-引用)
- [致谢与联系](#-致谢与联系)

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| **多格式支持** | 支持 PKL（单文件）与 NPZ（按 token 的文件夹）预测结果 |
| **多模型兼容** | MapTR、PivotNet、BEVMapNet、StreamMapNet 等 |
| **直接评估** | 加载预测结果即可评估，无需重新前向推理 |
| **灵活配置** | 通过配置文件定义字段映射与评估参数 |
| **稳定性指标** | 在场一致性、位置稳定性、形状稳定性及综合 mAS |
| **几何与对齐** | 折线处理、坐标变换、IoU、与 GT 对齐等 |
| **可视化** | 稳定性结果与轨迹可视化脚本 |
| **NuScenes 集成** | 支持 ego pose 与数据集解析 |

---

## 📁 项目结构

```
maptr_stability_eval/
├── src/maptr_stability_eval/     # 核心代码
│   ├── geometry/                 # 几何（折线、坐标变换）
│   ├── stability/                # 稳定性指标、对齐、分配器
│   ├── data_parser/              # PKL/NPZ/NuScenes 解析
│   └── utils/                    # 配置与通用工具
├── configs/                      # 各模型配置（maptr/pivotnet/bemapnet/streammapnet）
├── src/vis/                      # 可视化脚本
├── tools/                        # 评估与可视化 shell 脚本
├── main.py                       # 评估入口
├── requirements.txt
└── README.md
```

---

## 🔧 安装

### 环境要求

- **Python** ≥ 3.7  
- 无需 GPU 或深度学习框架（仅评估与可视化）  
- 支持 Linux / macOS / Windows  

### 安装步骤

```bash
git clone <repository-url>
cd maptr_stability_eval
pip install -r requirements.txt
pip install -e .
```

**可选**：使用 NuScenes 相关功能时安装：

```bash
pip install nuscenes-devkit
```

**核心依赖**：`numpy`、`scipy`、`shapely`、`tqdm`、`tabulate`、`matplotlib`、`seaborn`、`pandas`（见 `requirements.txt`）。

---

## 🚀 快速开始

### PKL 格式（如 MapTR）

```bash
python main.py \
    --data-format pkl \
    --prediction-file results.pkl \
    --config configs/maptr_trainval.py \
    --output-dir outputs
```

### NPZ 格式（如 PivotNet）

```bash
python main.py \
    --data-format npz \
    --prediction-file npz_folder/ \
    --config configs/pivotnet_trainval.py \
    --output-dir outputs
```

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--data-format` | `pkl` 或 `npz` | `pkl` |
| `--prediction-file` | 预测文件或 NPZ 目录 | 必填 |
| `--config` | 配置文件路径 | 必填 |
| `--output-dir` | 输出目录 | `outputs` |
| `--data-root` | NuScenes 数据根目录 | 可选 |
| `--stability-classes` | 评估类别 | `divider ped_crossing boundary` |
| `--stability-interval` | 帧间隔 | `2` |
| `--localization-weight` | 位置稳定性权重 | `0.5` |
| `--detection-threshold` | 检测阈值 | `0.3` |

更多配置（如 `pred-rotate-deg`、`pred-swap-xy`、`pred-flip-x/y`）见 `main.py --help`。

---

## 📊 稳定性指标

| 指标 | 含义 |
|------|------|
| **Presence** | 连续帧间同一实例的检测一致性（在场一致性） |
| **Localization** | 基于折线 IoU 的位置稳定性 |
| **Shape** | 基于曲率变化的形状稳定性 |
| **mAS** | 综合稳定性：Presence × (Localization × W + Shape × (1−W))，W 为 `localization_weight` |

输出示例：

```
----------------------------------
MapTR Stability Index Results
----------------------------------
| class        | SI    | presence | localization | shape |
|--------------|-------|----------|--------------|-------|
| divider      | 0.8234| 0.9123   | 0.8456       | 0.7891|
| ped_crossing | 0.7891| 0.8765   | 0.8123       | 0.7456|
| boundary     | 0.8567| 0.9234   | 0.8678       | 0.8234|
| mean         | 0.8231| 0.9041   | 0.8419       | 0.7860|
----------------------------------
```

---

## 📂 配置文件与数据格式

- **配置**：`configs/` 下按模型与数据集提供 `*_trainval.py` / `*_mini.py`（如 `maptr_trainval.py`、`pivotnet_trainval.py`），内含字段映射、类别映射、稳定性参数等。
- **PKL**：单文件，列表元素为样本字典，需包含折线、类别、分数、样本索引等（见各 config 的 `field_mapping`）。
- **NPZ**：每 token 一个 `.npz` 文件，目录内多文件；需包含 `pts_3d`、`labels_3d`、`scores_3d` 等（见 config）。

详细字段说明与示例见各配置文件内注释；PKL/NPZ 结构要求与 `field_mapping` 一致。

---

## 🖼️ 可视化

```bash
# 预测结果稳定性可视化
python src/vis/vis_stability.py \
    --prediction-file results.pkl \
    --config configs/maptr_trainval.py \
    --output-dir vis_outputs \
    --data-format pkl
```

GT 与预测的可视化脚本见 `tools/vis_groundtruth.sh`、`tools/vis_prediction.sh` 及 `src/vis/` 下脚本。

---

## 📖 引用

若本基准或工具包对您的研究有帮助，请引用：

```bibtex
@inproceedings{stablehdmap2026,
  title     = {Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping},
  author    = {Shan, Hao and Li, Ruikai and Jiang, Han and Fan, Yizhe and Yan, Ziyang and Li, Bohan and Hao, Xiaoshuai and Zhao, Hao and Cui, Zhiyong and Ren, Yilong and Yu, Haiyang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=mxz5RqhCMe}
}
```

- **论文**：[OpenReview](https://openreview.net/forum?id=mxz5RqhCMe)  
- **项目主页**：[https://stablehdmap.github.io/](https://stablehdmap.github.io/)

---

## 🙏 致谢与联系

- 感谢 OpenMMLab、MapTR、PivotNet 等相关工作的启发与贡献。  
- **作者**：Hao Shan  
- **邮箱**：bhsh0112@163.com  

欢迎通过 Issue 或 Pull Request 反馈与改进。

---

## 📜 许可证

MIT License

---

<details>
<summary><b>📚 更多文档（安装细节、开发指南、测试、更新日志）</b></summary>

### 开发与测试

```bash
# 单元测试
pytest tests/

# NPZ 加载与示例
python test_npz_loader.py
python example_usage.py
python demo_npz_usage.py
```

### 扩展指南

- **新指标**：在 `stability/metrics.py` 中实现并在主评估流程中挂载。  
- **新数据格式**：在 `configs/` 增加配置，在 `data_parser/` 中增加或复用加载器。  
- **新可视化**：在 `src/vis/` 或 `tools/` 中增加脚本，保持与现有输出格式兼容。

### 更新日志

- **v1.0.0**：初始版本；PKL/NPZ 支持；完整稳定性评估与可视化。  
- **近期**：更多模型配置（BEVMapNet、StreamMapNet 等）；NPZ 与可视化增强。

</details>
