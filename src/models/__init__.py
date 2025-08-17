"""
模型模块

该模块包含了多模态分类模型的定义和相关工具。
"""

from .multimodal_classifier import (
    MultimodalClassifier,
    StokesEncoder,
    FluorescenceEncoder, 
    ImageEncoder,
    CrossAttentionFusion
)
from .simple_classifier import SimpleClassifier
from .model_factory import ModelFactory
from .config import ModelConfig

__all__ = [
    'MultimodalClassifier',
    'StokesEncoder', 
    'FluorescenceEncoder',
    'ImageEncoder',
    'CrossAttentionFusion',
    'SimpleClassifier',
    'ModelFactory',
    'ModelConfig'
]