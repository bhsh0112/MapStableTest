# MapTR稳定性评估工具包

一个专门用于评估MapTR模型在连续帧间稳定性表现的工具包。该工具支持加载pkl格式和npz格式的预测结果进行评估，无需进行地图推测过程。

## 功能特性

- **多格式支持**: 支持pkl格式（单个文件）和npz格式（文件夹）的预测结果
- **模型兼容**: 兼容MapTR、PivotNet等多种地图构建模型的输出格式
- **NPZ文件支持**: 专门支持PivotNet等模型输出的npz文件格式，每个npz文件对应一个token
- **直接评估**: 直接加载预测结果进行评估，无需重新运行模型
- **灵活配置**: 通过配置文件定义输入文件字段格式，支持多种数据格式
- **稳定性指标**: 提供在场一致性、位置稳定性、形状稳定性等核心指标
- **几何计算**: 包含折线处理、坐标变换、IoU计算等几何功能
- **模块化设计**: 清晰的模块划分，便于扩展和维护
- **易于使用**: 简单的命令行接口和配置选项

## 项目结构

```
maptr_stability_eval/
├── src/maptr_stability_eval/          # 主要源代码
│   ├── geometry/                      # 几何计算模块
│   │   ├── __init__.py
│   │   ├── polyline_utils.py          # 折线处理工具
│   │   └── coordinate_transform.py    # 坐标变换
│   ├── stability/                     # 稳定性评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py                 # 稳定性指标计算
│   │   ├── alignment.py               # 检测与GT对齐
│   │   └── utils.py                   # 稳定性工具函数
│   ├── data_parser/                   # 数据解析模块
│   │   ├── __init__.py
│   │   ├── pkl_loader.py              # PKL文件加载
│   │   ├── npz_loader.py              # NPZ文件加载
│   │   └── utils.py                   # 数据解析工具
│   ├── utils/                         # 通用工具模块
│   │   ├── __init__.py
│   │   └── config.py                  # 配置和参数解析
│   └── __init__.py
├── configs/                           # 配置文件
│   ├── maptr.py                       # MapTR标准配置
│   └── pivotnet.py                    # PivotNet配置
├── tests/                             # 测试文件
├── examples/                          # 使用示例
├── main.py                            # 主程序入口
├── setup.py                           # 安装配置
├── requirements.txt                   # 依赖列表
├── pyproject.toml                     # 项目配置
└── README.md                          # 项目文档
```

## 安装

### 环境要求

- Python >= 3.7
- 无需GPU或深度学习框架

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd maptr_stability_eval
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装项目：
```bash
pip install -e .
```

## 使用方法

### 基本用法

#### PKL格式（单个文件）
```bash
python main.py \
    --data-format pkl \
    --prediction-file results.pkl \
    --config configs/maptr.py \
    --output-dir outputs
```

#### NPZ格式（文件夹）
```bash
python main.py \
    --data-format npz \
    --prediction-file npz_folder/ \
    --config configs/pivotnet.py \
    --output-dir outputs
```

### 主要参数

- `--data-format`: 数据格式，pkl（单个文件）或npz（文件夹）（默认: pkl）
- `--prediction-file`: 预测结果文件路径（pkl文件或npz文件夹）（必需）
- `--config`: 配置文件路径，定义输入文件字段格式（必需）
- `--output-dir`: 输出结果目录（默认: outputs）
- `--stability-classes`: 评估的类别（默认: divider ped_crossing boundary）
- `--stability-interval`: 帧间隔（默认: 2）
- `--localization-weight`: 位置稳定性权重（默认: 0.5）

### 使用MapTR标准配置

```bash
python main.py \
    --data-format pkl \
    --prediction-file maptr_results.pkl \
    --config configs/maptr.py \
    --output-dir outputs
```

### 使用PivotNet配置

```bash
python main.py \
    --data-format npz \
    --prediction-file pivotnet_results/ \
    --config configs/pivotnet.py \
    --output-dir outputs
```

### 自定义参数

```bash
python main.py \
    --data-format pkl \
    --prediction-file results.pkl \
    --config configs/maptr.py \
    --stability-classes divider boundary \
    --stability-interval 3 \
    --localization-weight 0.7 \
    --detection-threshold 0.5
```

## 配置文件

### 基本配置结构

```python
config = {
    # 字段映射配置
    'field_mapping': {
        'required_fields': ['polylines', 'types', 'scores', 'sample_idx', 'timestamp'],
        'polylines_field': 'polylines',
        'types_field': 'types',
        'scores_field': 'scores',
        # ... 其他字段映射
    },
    
    # 稳定性评估配置
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],
        'interval': 2,
        'localization_weight': 0.5,
        'detection_threshold': 0.3,
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
        'num_sample_points': 50
    }
}
```

### PKL文件格式

每个pkl文件应包含一个列表，列表中的每个元素是一个字典，代表一个样本的预测结果：

```python
[
    {
        'pts_3d': torch.Tensor([[[0, 0], [1, 1], [2, 0]], [[0, 1], [1, 2], [2, 1]]]),  # 折线数据
        'labels_3d': torch.Tensor([0, 2]),  # 类别标签: 0=divider, 1=ped_crossing, 2=boundary
        'scores_3d': torch.Tensor([0.8, 0.9]),  # 预测分数
        'sample_idx': 'token_001'  # 样本token
    },
    ...
]
```

### NPZ文件格式

每个npz文件对应一个token，文件夹中包含多个npz文件：

```
npz_folder/
├── token1.npz
├── token2.npz
├── token3.npz
└── ...
```

每个npz文件内容：

```python
# 保存数据
import numpy as np

pts_3d = np.array([
    [[0, 0], [1, 1], [2, 0]],  # 第一条折线
    [[0, 1], [1, 2], [2, 1]]   # 第二条折线
])
labels_3d = np.array([0, 2])  # 类别标签
scores_3d = np.array([0.8, 0.9])  # 预测分数

np.savez('token1.npz', 
         pts_3d=pts_3d,
         labels_3d=labels_3d,
         scores_3d=scores_3d)
```

## 稳定性指标

### 1. 在场一致性 (Presence Consistency)
衡量模型在连续帧间对同一实例检测的一致性。

### 2. 位置稳定性 (Localization Stability)
基于折线IoU计算的位置变化指标。

### 3. 形状稳定性 (Shape Stability)
基于曲率变化的形状一致性指标。

### 4. 综合稳定性指数 (Stability Index)
综合上述指标的加权平均：
```
SI = Presence × (Localization × W + Shape × (1-W))
```

## 输出结果

工具会生成详细的稳定性评估报告，包括：

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

结果会保存到指定的输出目录，包含详细的评估报告和参数设置。

## 配置示例

### 标准MapTR输出格式

使用 `configs/maptr_config.py` 配置文件，适用于标准MapTR模型输出：

```python
config = {
    'field_mapping': {
        'polylines_field': 'pts_3d',      # MapTR使用pts_3d
        'types_field': 'labels_3d',       # MapTR使用labels_3d
        'scores_field': 'scores_3d',      # MapTR使用scores_3d
    },
    'class_mapping': {
        0: 'divider',
        1: 'ped_crossing', 
        2: 'boundary'
    }
}
```

### 自定义格式

使用 `configs/example_config.py` 配置文件，适用于自定义格式：

```python
config = {
    'field_mapping': {
        'polylines_field': 'polylines',
        'types_field': 'types',
        'scores_field': 'scores',
    }
}
```

## 开发指南

### 添加新的稳定性指标

1. 在 `src/maptr_stability_eval/stability/metrics.py` 中添加新函数
2. 在 `eval_maptr_stability_index` 中集成新指标
3. 更新 `print_stability_index_results` 函数以显示新指标

### 支持新的数据格式

1. 创建新的配置文件，定义字段映射
2. 在 `src/maptr_stability_eval/data_parser/pkl_loader.py` 中添加格式支持
3. 更新数据验证逻辑

## 测试

### 运行NPZ加载器测试
```bash
python test_npz_loader.py
```

### 查看使用示例
```bash
python example_usage.py
```

### 运行单元测试
```bash
pytest tests/
```

### 运行导入测试
```bash
python test_imports.py
```

## 贡献

欢迎提交Issue和Pull Request来改进这个工具包。

## 许可证

MIT License

## 致谢

- 基于OpenMMLab的MMDetection3D框架
- 感谢MapTR项目的原始实现

## 联系方式

如有问题或建议，请联系：
- 作者: Zhiqi Li
- 邮箱: zhiqi.li@example.com