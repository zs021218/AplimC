#!/usr/bin/env python3
"""
AplimC - 多模态藻类分类项目
专为HDF5格式优化的高效多模态机器学习框架
"""

__version__ = "3.0.0"
__author__ = "Sen Zhang"
__description__ = "多模态藻类分类系统 - 支持Stokes参数、荧光信号和多视图图像，包含知识蒸馏和平衡融合"

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
    create_simple_model,
    FeatureMimicryDistillation,
    EnhancedSignalClassifier,
    RelationKnowledgeExtractor,
    AdaptiveAttentionTransfer,
    DistillationLoss
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
    "pytorch_requires": ">=1.8.0",
    "new_features": [
        "知识蒸馏框架 (图像→信号模态知识转移)",
        "平衡多模态融合策略",
        "增强版信号编码器",
        "跨模态注意力机制",
        "特征模仿学习"
    ]
}

# 项目信息
PROJECT_INFO = {
    "name": "AplimC",
    "full_name": "Algae classification with Polarization, Light and Image Multimodal Classifier",
    "data_format": "HDF5",
    "modalities": ["stokes", "fluorescence", "images"],
    "num_classes": 12,
    "total_samples": 21007,
    "features": {
        "multimodal_fusion": ["concat", "attention", "balanced", "gradient_balanced", "adaptive"],
        "knowledge_distillation": ["feature_mimicry", "relation_knowledge", "attention_transfer"],
        "signal_enhancement": ["multi_scale_conv", "enhanced_encoder", "adaptive_attention"],
        "performance": {
            "image_only": "95.59%",
            "fluorescence_only": "66.96%", 
            "multimodal": "96.41%",
            "distillation_target": "90%+ (signal only)"
        }
    }
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
    
    # 知识蒸馏模块
    "FeatureMimicryDistillation",
    "EnhancedSignalClassifier", 
    "DistillationLoss",
    "load_pretrained_teacher",
    
    # 平衡融合模块
    "ModalityBalancer",
    "GradientBalancedFusion", 
    "AdaptiveFusion",
    "CrossModalAttention",
    
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
    get_preset_config,
    # 知识蒸馏
    FeatureMimicryDistillation,
    DistillationLoss,
    # 平衡融合
    ModalityBalancer
)

# === 方案1: 标准多模态分类 ===
print("方案1: 多模态分类")
config = get_preset_config('balanced')  # 使用平衡融合
train_loader = create_default_dataloader("data/processed/multimodal_data.h5", 'train')
val_loader = create_default_dataloader("data/processed/multimodal_data.h5", 'val')

model = MultimodalClassifier(config)
trainer = MultimodalTrainer(model, config, train_loader, val_loader)
trainer.fit()

# === 方案2: 知识蒸馏 (信号模态提升) ===
print("方案2: 知识蒸馏 (68% → 90%+)")
from src.models.knowledge_distillation import FeatureMimicryDistillation

# 创建蒸馏模型
distill_model = FeatureMimicryDistillation(
    teacher_config=teacher_config,  # 图像模态配置
    student_config=student_config,  # 信号模态配置  
    distillation_config=distill_config
)

# 蒸馏训练
distill_criterion = DistillationLoss(alpha=0.3, beta=0.4, gamma=0.2, delta=0.1)
# ... 训练过程 ...

# === 方案3: 仅图像分类 (推荐) ===
print("方案3: 图像单模态 (95%+, 最实用)")
config_img = get_preset_config('image_only')
model_img = MultimodalClassifier(config_img)
# ... 简单高效 ...

print("选择方案:")
print("- 方案1: 完整多模态 (96%, 复杂)")
print("- 方案2: 信号蒸馏 (90%+, 研究)")  
print("- 方案3: 图像单模态 (95%, 推荐)")
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
╔══════════════════════════════════════════════════════════════════╗
║                            AplimC v{__version__}                          ║
║               多模态藻类分类系统 + 知识蒸馏框架                    ║
║                                                                  ║
║  🧬 支持模态: Stokes参数 + 荧光信号 + 多视图图像                   ║
║  📊 数据格式: HDF5 优化存储 (84GB → 高效访问)                     ║
║  🎯 分类数量: 12类藻类                                            ║
║  📈 样本总数: 21,007个                                            ║
║                                                                  ║
║  🔄 新功能特性:                                                   ║
║     • 知识蒸馏: 图像(95%) → 信号(90%+)                          ║
║     • 平衡融合: 解决模态参数不平衡问题                           ║
║     • 增强编码器: 多尺度信号特征提取                             ║
║     • 跨模态注意力: 智能特征关联                                 ║
║                                                                  ║
║  📊 性能基准:                                                     ║
║     • 图像单模态: 95.59% (推荐方案)                             ║
║     • 荧光单模态: 66.96% → 蒸馏后90%+                           ║
║     • 多模态融合: 96.41% (复杂度↑, 收益微小)                    ║
║                                                                  ║
║  遵循奥卡姆剃刀原理 - 简单有效优于复杂                           ║
╚══════════════════════════════════════════════════════════════════╝
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
