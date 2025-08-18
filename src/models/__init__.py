#!/usr/bin/env python3
"""
模型模块 - 简单有效的多模态分类器
基于奥卡姆剃刀原理，专注于模型架构和实现
"""

from typing import List

# 核心模型类
from .classifier import (
    MultimodalClassifier,
    SignalEncoder,
    ImageEncoder,
    AttentionFusion,
    ModelConfig,
    create_model,
    create_simple_model
)

__version__ = "1.0.0"
__all__ = [
    # 核心模型
    "MultimodalClassifier",
    "SignalEncoder", 
    "ImageEncoder",
    "AttentionFusion",
    "ModelConfig",
    "create_model",
    "create_simple_model"
]


def get_model_info():
    """获取模型模块信息"""
    info = {
        "version": __version__,
        "supported_modalities": ["stokes", "fluorescence", "images"],
        "fusion_strategies": ["concat", "attention", "weighted"],
        "model_types": ["MultimodalClassifier"],
        "encoders": ["SignalEncoder", "ImageEncoder"]
    }
    return info


def create_model_from_modalities(modalities: List[str], num_classes: int = 12, **kwargs) -> MultimodalClassifier:
    """
    根据模态列表创建模型的便捷函数
    
    Args:
        modalities: 要使用的模态列表 ['stokes', 'fluorescence', 'images']
        num_classes: 分类数目
        **kwargs: 其他模型配置参数
    
    Returns:
        配置好的模型实例
    """
    # 根据模态数量调整默认配置
    num_modalities = len(modalities)
    
    if num_modalities == 1:
        # 单模态，简化网络
        default_hidden_dims = kwargs.get('hidden_dims', [128, 64])
        default_dropout = kwargs.get('dropout_rate', 0.2)
    elif num_modalities == 2:
        # 双模态，中等复杂度
        default_hidden_dims = kwargs.get('hidden_dims', [256, 128])
        default_dropout = kwargs.get('dropout_rate', 0.25)
    else:
        # 多模态，保持复杂度
        default_hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
        default_dropout = kwargs.get('dropout_rate', 0.3)
    
    # 创建模型配置
    config = ModelConfig(
        num_classes=num_classes,
        use_stokes='stokes' in modalities,
        use_fluorescence='fluorescence' in modalities,
        use_images='images' in modalities,
        hidden_dims=default_hidden_dims,
        dropout_rate=default_dropout,
        activation=kwargs.get('activation', 'relu'),
        fusion_strategy=kwargs.get('fusion_strategy', 'concat')
    )
    
    return create_model(config)


def quick_model_comparison():
    """快速模型比较 - 不同模态组合"""
    modality_combinations = [
        ['stokes'],
        ['fluorescence'], 
        ['images'],
        ['stokes', 'fluorescence'],
        ['stokes', 'images'],
        ['fluorescence', 'images'],
        ['stokes', 'fluorescence', 'images']
    ]
    
    print("=" * 70)
    print(f"{'模型比较 - 不同模态组合':^70}")
    print("=" * 70)
    print(f"{'模态组合':<25} {'隐藏层维度':<20} {'融合策略':<15}")
    print("-" * 70)
    
    for modalities in modality_combinations:
        model = create_model_from_modalities(modalities)
        
        modality_str = '+'.join(modalities)
        hidden_str = str(model.config.hidden_dims)
        fusion_str = model.config.fusion_strategy
        
        print(f"{modality_str:<25} {hidden_str:<20} {fusion_str:<15}")
    
    print("=" * 70)


# 设置日志
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())