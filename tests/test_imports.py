#!/usr/bin/env python3
"""
测试导入脚本

验证所有模块是否能正确导入。
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试主包导入
        import maptr_stability_eval
        print("✓ 主包导入成功")
        
        # 测试几何模块
        from maptr_stability_eval.geometry import poly_get_samples, polyline_iou
        print("✓ 几何模块导入成功")
        
        # 测试稳定性模块
        from maptr_stability_eval.stability import compute_presence_consistency
        print("✓ 稳定性模块导入成功")
        
        # 测试数据解析模块
        from maptr_stability_eval.data_parser import load_prediction_results, parse_prediction_for_stability
        print("✓ 数据解析模块导入成功")
        
        # 测试工具模块
        from maptr_stability_eval.utils import parse_args
        print("✓ 工具模块导入成功")
        
        print("\n核心模块导入测试通过！")
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        import numpy as np
        from maptr_stability_eval.geometry import poly_get_samples, polyline_iou
        
        # 测试折线采样
        poly = np.array([[0, 0], [1, 1], [2, 0]])
        samples = poly_get_samples(poly, num_samples=5)
        print(f"✓ 折线采样测试通过: {len(samples)} 个采样点")
        
        # 测试IoU计算
        poly1 = np.array([[0, 0], [1, 1], [2, 0]])
        poly2 = np.array([[0, 0.1], [1, 1.1], [2, 0.1]])
        iou = polyline_iou(poly1, poly2, samples)
        print(f"✓ IoU计算测试通过: {iou:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("MapTR稳定性评估工具包 - 导入测试")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 所有测试通过！项目结构正确。")
    else:
        print("\n❌ 测试失败，请检查项目结构。")
        sys.exit(1)
