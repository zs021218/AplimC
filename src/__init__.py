#!/usr/bin/env python3
"""
AplimC - 多模态藻类分类项目
专为HDF5格式优化的高效多模态机器学习框架
"""

__version__ = "2.0.0"
__author__ = "Sen Zhang"
__description__ = "多模态藻类分类系统 - 支持Stokes参数、荧光信号和多视图图像"

# 核心模块导入
from . import data
from . import models
from . import training
from . import utils

# 主要类和函数导入
from .data import (
    MultimodalHDF5Dataset,
    create_default_dataloader,
    get_train_transforms,
    get_val_transforms
)

from .models import (
    MultimodalClassifier,
    SignalEncoder,
    ImageEncoder,
    AttentionFusion,
    create_model,
    create_simple_model
)

from .training import (
    MultimodalTrainer
)

from .utils import (
    ModelConfig,
    get_preset_config,
    count_parameters,
    analyze_model,
    save_model_checkpoint,
    load_model_checkpoint
)

# 版本信息
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "python_requires": ">=3.8",
    "pytorch_requires": ">=1.8.0"
}

# 项目信息
PROJECT_INFO = {
    "name": "AplimC",
    "full_name": "Algae classification with Polarization, Light and Image Multimodal Classifier",
    "data_format": "HDF5",
    "modalities": ["stokes", "fluorescence", "images"],
    "num_classes": 12,
    "total_samples": 21007
}

# 公开的API
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__description__",
    "VERSION_INFO",
    "PROJECT_INFO",
    
    # 数据模块
    "MultimodalHDF5Dataset",
    "create_default_dataloader",
    "get_train_transforms",
    "get_val_transforms",
    
    # 模型模块
    "MultimodalClassifier",
    "SignalEncoder",
    "ImageEncoder", 
    "AttentionFusion",
    "create_model",
    "create_simple_model",
    
    # 训练模块
    "MultimodalTrainer",
    
    # 工具模块
    "ModelConfig",
    "get_preset_config",
    "count_parameters",
    "analyze_model",
    "save_model_checkpoint", 
    "load_model_checkpoint",
    
    # 子模块
    "data",
    "models", 
    "training",
    "utils"
]


def get_project_info():
    """获取项目信息"""
    return PROJECT_INFO.copy()


def get_version_info():
    """获取版本信息"""
    return VERSION_INFO.copy()


def quick_start_example():
    """快速开始示例代码"""
    example = """
# AplimC 快速开始示例

import torch
from src import (
    ModelConfig, 
    MultimodalClassifier,
    MultimodalTrainer,
    create_default_dataloader,
    get_preset_config
)

# 1. 创建配置
config = get_preset_config('simple')
print(f"使用配置: {config}")

# 2. 创建数据加载器
train_loader = create_default_dataloader(
    hdf5_path="data/processed/multimodal_data.h5",
    split='train',
    batch_size=config.batch_size,
    balanced=True
)

val_loader = create_default_dataloader(
    hdf5_path="data/processed/multimodal_data.h5", 
    split='val',
    batch_size=config.batch_size,
    balanced=False
)

# 3. 创建模型
model = MultimodalClassifier(config)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 4. 创建训练器
trainer = MultimodalTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

# 5. 开始训练
trainer.fit()

# 6. 评估模型
results = trainer.test(val_loader)
print(f"验证准确率: {results['accuracy']:.4f}")
"""
    return example


def check_environment():
    """检查运行环境"""
    import sys
    import torch
    import h5py
    import numpy as np
    
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "h5py_version": h5py.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return info


def print_banner():
    """打印项目横幅"""
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                         AplimC                           ║
║              多模态藻类分类系统 v{__version__}                  ║
║                                                          ║
║  🧬 支持模态: Stokes参数 + 荧光信号 + 多视图图像              ║
║  📊 数据格式: HDF5 优化存储                                ║
║  🎯 分类数量: 12类藻类                                     ║
║  📈 样本总数: 21,007个                                     ║
║                                                          ║
║  遵循奥卡姆剃刀原理 - 从简单到复杂的模型设计                  ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


# 模块初始化时的检查
def _initialize():
    """模块初始化"""
    try:
        # 检查依赖
        import torch
        import h5py
        import numpy as np
        
        # 检查版本
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (1, 8):
            import warnings
            warnings.warn(
                f"PyTorch版本 {torch.__version__} 可能存在兼容性问题，建议使用 >=1.8.0",
                UserWarning
            )
        
        return True
        
    except ImportError as e:
        import warnings
        warnings.warn(f"缺少依赖项: {e}", ImportWarning)
        return False


# 执行初始化
_INITIALIZED = _initialize()


# 便捷函数
def create_simple_classifier(hdf5_path: str, device: str = "auto"):
    """
    创建简单的分类器实例
    
    Args:
        hdf5_path: HDF5数据文件路径
        device: 设备类型 ("auto", "cpu", "cuda:0", 等)
        
    Returns:
        tuple: (model, train_loader, val_loader, config)
    """
    import torch
    
    # 自动选择设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建配置
    config = get_preset_config('simple')
    
    # 创建数据加载器
    train_loader = create_default_dataloader(
        hdf5_path=hdf5_path,
        split='train',
        batch_size=config.batch_size,
        balanced=True
    )
    
    val_loader = create_default_dataloader(
        hdf5_path=hdf5_path,
        split='val', 
        batch_size=config.batch_size,
        balanced=False
    )
    
    # 创建模型
    model = MultimodalClassifier(config).to(device)
    
    return model, train_loader, val_loader, config


def create_lightweight_classifier(hdf5_path: str, device: str = "auto"):
    """
    创建轻量级分类器实例（仅信号模态）
    
    Args:
        hdf5_path: HDF5数据文件路径
        device: 设备类型
        
    Returns:
        tuple: (model, train_loader, val_loader, config)
    """
    import torch
    
    # 自动选择设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建轻量级配置
    config = get_preset_config('lightweight')
    
    # 创建数据加载器
    train_loader = create_default_dataloader(
        hdf5_path=hdf5_path,
        split='train',
        batch_size=config.batch_size,
        balanced=True
    )
    
    val_loader = create_default_dataloader(
        hdf5_path=hdf5_path,
        split='val',
        batch_size=config.batch_size, 
        balanced=False
    )
    
    # 创建模型
    model = MultimodalClassifier(config).to(device)
    
    return model, train_loader, val_loader, config


# 调试信息
if __name__ == "__main__":
    print_banner()
    print("\n📋 项目信息:")
    for key, value in get_project_info().items():
        print(f"  {key}: {value}")
    
    print("\n🔧 环境信息:")
    env_info = check_environment()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    print("\n🚀 快速开始示例:")
    print(quick_start_example())
