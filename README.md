# MapTR稳定性评估工具包

一个专门用于评估MapTR模型在连续帧间稳定性表现的工具包。该工具支持加载pkl格式和npz格式的预测结果进行评估，无需进行地图推测过程。现已支持多种地图构建模型的输出格式，包括MapTR、PivotNet等。

## 功能特性

- **多格式支持**: 支持pkl格式（单个文件）和npz格式（文件夹）的预测结果
- **模型兼容**: 兼容MapTR、PivotNet等多种地图构建模型的输出格式
- **NPZ文件支持**: 专门支持PivotNet等模型输出的npz文件格式，每个npz文件对应一个token
- **直接评估**: 直接加载预测结果进行评估，无需重新运行模型
- **灵活配置**: 通过配置文件定义输入文件字段格式，支持多种数据格式
- **稳定性指标**: 提供在场一致性、位置稳定性、形状稳定性等核心指标
- **几何计算**: 包含折线处理、坐标变换、IoU计算等几何功能
- **可视化支持**: 提供稳定性结果可视化工具和脚本
- **NuScenes集成**: 支持NuScenes数据集的ego pose信息提取
- **模块化设计**: 清晰的模块划分，便于扩展和维护
- **易于使用**: 简单的命令行接口和配置选项
- **开发友好**: 包含完整的测试套件和示例代码

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
│   │   ├── metrics_fixed.py           # 修复版稳定性指标
│   │   ├── alignment.py               # 检测与GT对齐
│   │   ├── maptr_assigner.py          # MapTR分配器
│   │   └── utils.py                   # 稳定性工具函数
│   ├── data_parser/                   # 数据解析模块
│   │   ├── __init__.py
│   │   ├── pkl_loader.py              # PKL文件加载
│   │   ├── npz_loader.py              # NPZ文件加载
│   │   ├── nuscenes_parser.py         # NuScenes数据解析
│   │   └── utils.py                   # 数据解析工具
│   ├── utils/                         # 通用工具模块
│   │   ├── __init__.py
│   │   └── config.py                  # 配置和参数解析
│   └── __init__.py
├── configs/                           # 配置文件
│   ├── maptr_trainval.py              # MapTR trainval配置
│   ├── maptr_mini.py                  # MapTR mini配置
│   ├── pivotnet_trainval.py           # PivotNet trainval配置
│   └── pivotnet_mini.py               # PivotNet mini配置
├── vis/                               # 可视化工具
│   ├── vis_stability.py               # 稳定性可视化主脚本
│   ├── vis_stability_new.py           # 新版可视化脚本
│   ├── vis_stability_gt.sh            # GT可视化脚本
│   └── vis_stability_pred.sh          # 预测结果可视化脚本
├── data/                              # 数据目录
│   └── nuscenes/                      # NuScenes数据集
├── tests/                             # 测试文件
│   ├── __init__.py
│   ├── test_geometry.py               # 几何计算测试
│   ├── test_imports.py                # 导入测试
│   └── test_data.pkl                  # 测试数据
├── test_npz_data/                     # NPZ测试数据
│   ├── test_token_000.npz
│   ├── test_token_001.npz
│   └── test_token_002.npz
├── example_usage.py                   # 使用示例
├── demo_npz_usage.py                  # NPZ使用演示
├── find_token.py                      # Token查找工具
├── token2image.py                     # Token到图像转换
├── test_npz_loader.py                 # NPZ加载器测试
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
- 支持Linux、macOS、Windows

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

3. 安装项目（开发模式）：
```bash
pip install -e .
```

### 可选依赖

如果需要使用NuScenes数据集功能：
```bash
pip install nuscenes-devkit
```

如果需要开发和测试功能：
```bash
pip install -e ".[dev]"
```

### 依赖说明

核心依赖包括：
- `numpy>=1.19.0` - 数值计算
- `scipy>=1.6.0` - 科学计算
- `shapely>=1.7.0` - 几何计算
- `tqdm>=4.60.0` - 进度条
- `tabulate>=0.8.0` - 表格显示
- `matplotlib>=3.3.0` - 可视化
- `seaborn>=0.11.0` - 统计可视化
- `pandas>=1.2.0` - 数据处理

## 使用方法

### 基本用法

#### PKL格式（单个文件）
```bash
python main.py \
    --data-format pkl \
    --prediction-file results.pkl \
    --config configs/maptr_trainval.py \
    --output-dir outputs
```

#### NPZ格式（文件夹）
```bash
python main.py \
    --data-format npz \
    --prediction-file npz_folder/ \
    --config configs/pivotnet_trainval.py \
    --output-dir outputs
```

### 主要参数

- `--data-format`: 数据格式，pkl（单个文件）或npz（文件夹）（默认: pkl）
- `--prediction-file`: 预测结果文件路径（pkl文件或npz文件夹）（必需）
- `--config`: 配置文件路径，定义输入文件字段格式（必需）
- `--output-dir`: 输出结果目录（默认: outputs）
- `--data-root`: NuScenes数据集根目录（可选）
- `--nusc-version`: NuScenes数据集版本（默认: v1.0-trainval）
- `--stability-classes`: 评估的类别（默认: divider ped_crossing boundary）
- `--stability-interval`: 帧间隔（默认: 2）
- `--localization-weight`: 位置稳定性权重（默认: 0.5）
- `--detection-threshold`: 检测阈值（默认: 0.3）
- `--pred-rotate-deg`: 预测结果旋转角度（默认: 0.0）
- `--pred-swap-xy`: 是否交换x/y坐标（默认: False）
- `--pred-flip-x`: 是否翻转x轴（默认: False）
- `--pred-flip-y`: 是否翻转y轴（默认: False）

### 使用MapTR配置

```bash
# 使用trainval配置
python main.py \
    --data-format pkl \
    --prediction-file maptr_results.pkl \
    --config configs/maptr_trainval.py \
    --output-dir outputs

# 使用mini配置
python main.py \
    --data-format pkl \
    --prediction-file maptr_results.pkl \
    --config configs/maptr_mini.py \
    --output-dir outputs
```

### 使用PivotNet配置

```bash
# 使用trainval配置
python main.py \
    --data-format npz \
    --prediction-file pivotnet_results/ \
    --config configs/pivotnet_trainval.py \
    --output-dir outputs

# 使用mini配置
python main.py \
    --data-format npz \
    --prediction-file pivotnet_results/ \
    --config configs/pivotnet_mini.py \
    --output-dir outputs
```

### 带NuScenes数据集的完整评估

```bash
python main.py \
    --data-format pkl \
    --prediction-file maptr_results.pkl \
    --config configs/maptr_trainval.py \
    --data-root /path/to/nuscenes \
    --nusc-version v1.0-trainval \
    --output-dir outputs
```

### 自定义参数示例

```bash
python main.py \
    --data-format pkl \
    --prediction-file results.pkl \
    --config configs/maptr_trainval.py \
    --stability-classes divider boundary \
    --stability-interval 3 \
    --localization-weight 0.7 \
    --detection-threshold 0.5 \
    --pred-rotate-deg 90.0 \
    --pred-swap-xy \
    --output-dir outputs
```

### 快速测试

使用提供的测试数据快速验证功能：

```bash
# 测试PKL加载
python test_npz_loader.py

# 查看使用示例
python example_usage.py

# NPZ功能演示
python demo_npz_usage.py
```

## 可视化工具

项目提供了丰富的可视化工具来帮助分析和理解稳定性评估结果。

### 稳定性可视化

使用 `vis/vis_stability.py` 进行稳定性结果可视化：

```bash
# 可视化预测结果的稳定性
python vis/vis_stability.py \
    --prediction-file results.pkl \
    --config configs/maptr_trainval.py \
    --output-dir vis_outputs \
    --data-format pkl
```

### GT可视化

使用 `vis/vis_stability_gt.sh` 脚本可视化Ground Truth：

```bash
# 可视化GT稳定性
bash vis/vis_stability_gt.sh
```

### 预测结果可视化

使用 `vis/vis_stability_pred.sh` 脚本可视化预测结果：

```bash
# 可视化预测结果稳定性
bash vis/vis_stability_pred.sh
```

### Token到图像转换

使用 `token2image.py` 将token转换为对应的图像：

```bash
# 将token转换为图像
python token2image.py \
    --token your_token_here \
    --data-root /path/to/nuscenes \
    --output-dir token_images
```

### Token查找工具

使用 `find_token.py` 查找特定的token：

```bash
# 查找包含特定内容的token
python find_token.py \
    --prediction-file results.pkl \
    --search-term "specific_content" \
    --output-file found_tokens.txt
```

## 配置文件

项目提供了多种预配置的配置文件，适用于不同的模型和数据格式。

### MapTR配置 (maptr_trainval.py)

适用于MapTR模型输出的标准配置：

```python
config = {
    # 字段映射配置 - 基于实际pkl文件结构
    'field_mapping': {
        'required_fields': [
            'pts_3d',         # MapTR输出中的折线数据
            'labels_3d',      # MapTR输出中的类别标签
            'scores_3d',      # MapTR输出中的预测分数
            'sample_idx'      # 样本索引
        ],
        'polylines_field': 'pts_3d',
        'types_field': 'labels_3d',
        'scores_field': 'scores_3d',
        'instance_ids_field': 'instance_ids',
        'scene_id_field': 'scene_token',
        'timestamp_field': 'timestamp',
    },
    
    # 类别映射 - MapTR使用数字标签
    'class_mapping': {
        0: 'divider',        # 车道分隔线
        1: 'ped_crossing',   # 人行横道
        2: 'boundary'        # 道路边界
    },
    
    # 坐标变换配置
    'coordinate_transform': {
        'rotate_deg': 0.0,
        'swap_xy': False,
        'flip_x': False,
        'flip_y': False
    },
    
    # 稳定性评估配置
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],
        'interval': 2,
        'localization_weight': 0.5,
        'detection_threshold': 0.3,
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
        'num_sample_points': 50
    },
    
    # NuScenes数据集配置
    'nuscenes': {
        'version': 'v1.0-trainval',
        'dataroot': 'data/nuscenes',
        'verbose': False
    }
}
```

### PivotNet配置 (pivotnet_trainval.py)

适用于PivotNet模型输出的NPZ格式配置：

```python
config = {
    # 字段映射配置 - 基于NPZ文件结构
    'field_mapping': {
        'required_fields': [
            'pts_3d',
            'labels_3d', 
            'scores_3d'
        ],
        'polylines_field': 'pts_3d',
        'types_field': 'labels_3d',
        'scores_field': 'scores_3d',
    },
    
    # 类别映射
    'class_mapping': {
        0: 'divider',
        1: 'ped_crossing',
        2: 'boundary'
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

### 配置文件选择指南

- **maptr_trainval.py**: 适用于MapTR模型在trainval数据集上的输出
- **maptr_mini.py**: 适用于MapTR模型在mini数据集上的输出
- **pivotnet_trainval.py**: 适用于PivotNet模型在trainval数据集上的NPZ输出
- **pivotnet_mini.py**: 适用于PivotNet模型在mini数据集上的NPZ输出

### PKL文件格式

每个pkl文件应包含一个列表，列表中的每个元素是一个字典，代表一个样本的预测结果：

```python
[
    {
        'pts_3d': torch.Tensor([[[0, 0], [1, 1], [2, 0]], [[0, 1], [1, 2], [2, 1]]]),  # 折线数据
        'labels_3d': torch.Tensor([0, 2]),  # 类别标签: 0=divider, 1=ped_crossing, 2=boundary
        'scores_3d': torch.Tensor([0.8, 0.9]),  # 预测分数
        'sample_idx': '30e55a3ec6184d8cb1944b39ba19d622'  # 样本token
    },
    ...
]
```

**MapTR PKL文件特点：**
- `pts_3d`: 形状为(N, P, 2)，N为折线数量，P为每条折线的点数
- `labels_3d`: 形状为(N,)，每个元素是0、1或2
- `scores_3d`: 形状为(N,)，每个元素是浮点数
- `sample_idx`: 字符串类型的样本索引

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

**NPZ文件特点：**
- 文件名通常包含token信息
- 支持PivotNet等模型的输出格式
- 每个文件独立存储一个样本的预测结果

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


## 开发指南

### 添加新的稳定性指标

1. 在 `src/maptr_stability_eval/stability/metrics.py` 中添加新函数
2. 在 `eval_maptr_stability_index` 中集成新指标
3. 更新 `print_stability_index_results` 函数以显示新指标

### 支持新的数据格式

1. 创建新的配置文件，定义字段映射
2. 在 `src/maptr_stability_eval/data_parser/` 中添加相应的加载器
3. 更新数据验证逻辑

### 添加新的可视化工具

1. 在 `vis/` 目录下创建新的可视化脚本
2. 参考现有的可视化脚本结构
3. 确保与主程序的输出格式兼容

### 扩展配置文件

1. 在 `configs/` 目录下创建新的配置文件
2. 参考现有配置文件的结构和命名规范
3. 确保包含所有必要的配置项

## 测试

项目提供了完整的测试套件来验证功能正确性。

### 运行NPZ加载器测试
```bash
python test_npz_loader.py
```

### 查看使用示例
```bash
python example_usage.py
```

### NPZ功能演示
```bash
python demo_npz_usage.py
```

### 运行单元测试
```bash
pytest tests/
```

### 运行导入测试
```bash
python test_imports.py
```

### 运行几何计算测试
```bash
python tests/test_geometry.py
```

### 代码质量检查

安装开发依赖后，可以使用以下工具进行代码质量检查：

```bash
# 代码格式化
black src/ tests/ *.py

# 代码风格检查
flake8 src/ tests/ *.py

# 类型检查
mypy src/

# 运行所有测试
pytest tests/ -v --cov=src/
```

## 贡献

欢迎提交Issue和Pull Request来改进这个工具包。

### 贡献指南

1. Fork 项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的注释和文档字符串
- 确保测试覆盖率
- 更新相关文档

## 许可证

MIT License

## 致谢

- 基于OpenMMLab的MMDetection3D框架
- 感谢MapTR项目的原始实现
- 感谢PivotNet项目的贡献

## 联系方式

如有问题或建议，请联系：
- 作者: Hao Shan
- 邮箱: bhsh0112@163.com

## 更新日志

### v1.0.0
- 初始版本发布
- 支持PKL和NPZ格式
- 提供完整的稳定性评估功能
- 包含可视化工具和测试套件

### 最新更新
- 添加了更多配置文件选项
- 增强了NPZ文件支持
- 改进了可视化工具
- 完善了测试覆盖


1111