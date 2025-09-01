#!/usr/bin/env python3
"""
数据模块 - 专为HDF5格式优化的多模态数据加载和处理
支持大规模数据集的高效加载、平衡采样和内存优化
"""

from typing import Dict, Optional, List, Union

# 核心数据加载类
from .dataset import (
    MultimodalHDF5Dataset,
    MultimodalSubset
)

# 数据加载器和采样器
from .dataloader import (
    BalancedBatchSampler,
    HDF5CollateFunction,
    AdaptiveBatchSizer,
    create_dataloaders,
    create_efficient_dataloader,
    create_balanced_dataloader,
    create_memory_optimized_dataloader
)

# 数据变换
from .transforms import (
    # 基础变换
    BaseTransform,
    Compose,
    ToDevice,
    
    # 信号变换
    SignalNormalize,
    SignalClip,
    SignalRandomCrop,
    SignalGaussianNoise,
    SignalSmoothing,
    
    # 图像变换
    ImageNormalize,
    ImageRandomRotation,
    ImageRandomFlip,
    ImageColorJitter,
    
    # 混合变换
    RandomMixUp,
    
    # 预定义变换组合
    get_train_transforms,
    get_val_transforms
)

__version__ = "2.0.0"
__all__ = [
    # 数据集类
    "MultimodalHDF5Dataset",
    "MultimodalSubset",
    
    # 数据加载器
    "BalancedBatchSampler",
    "HDF5CollateFunction", 
    "AdaptiveBatchSizer",
    "create_dataloaders",
    "create_balanced_dataloader",
    "create_efficient_dataloader",
    "create_memory_optimized_dataloader",
    
    # 变换类
    "BaseTransform",
    "Compose",
    "ToDevice",
    "SignalNormalize",
    "SignalClip", 
    "SignalRandomCrop",
    "SignalGaussianNoise",
    "SignalSmoothing",
    "ImageNormalize",
    "ImageRandomRotation",
    "ImageRandomFlip", 
    "ImageColorJitter",
    "RandomMixUp",
    "get_train_transforms",
    "get_val_transforms"
]


def get_dataset_info():
    """获取数据集信息"""
    info = {
        "format": "HDF5",
        "modalities": ["stokes", "fluorescence", "images"],
        "stokes_shape": (4, 4000),
        "fluorescence_shape": (16, 4000), 
        "images_shape": (3, 224, 224, 3),
        "num_classes": 12,
        "total_samples": 21007,
        "dtype": "float32"
    }
    return info


def create_default_dataloader(
    hdf5_path: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    balanced: bool = True,
    use_cache: bool = True,
    transform_config: Dict = None,  # 添加数据增强配置参数
    selected_classes: Optional[Union[List[str], List[int]]] = None,  # 添加类别过滤参数
    **kwargs
):
    """
    创建默认配置的数据加载器
    
    Args:
        hdf5_path: HDF5文件路径
        split: 数据分割 ('train', 'val', 'test')
        batch_size: 批次大小
        num_workers: 工作进程数
        balanced: 是否使用平衡采样
        use_cache: 是否使用缓存
        transform_config: 数据增强配置字典
        selected_classes: 选择的类别列表（类别名称或类别ID）
        **kwargs: 其他参数
    
    Returns:
        DataLoader对象
    """
    from torch.utils.data import DataLoader
    
    # 创建数据集
    dataset = MultimodalHDF5Dataset(
        hdf5_path=hdf5_path,
        split=split,
        cache_size=kwargs.get('cache_size', 1000) if use_cache else 0,
        memory_map=kwargs.get('memory_map', True),
        transform=get_train_transforms(transform_config) if split == 'train' else get_val_transforms(transform_config),
        selected_classes=selected_classes  # 添加类别过滤参数
    )
    
    # 选择数据加载器创建函数
    if balanced and split == 'train':
        return create_balanced_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
    else:
        # 创建标准DataLoader
        import torch
        from torch.utils.data import DataLoader
        
        default_kwargs = {
            'batch_size': batch_size,
            'shuffle': (split == 'train'),
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': (split == 'train'),
            'collate_fn': HDF5CollateFunction(dataset.load_modalities)
        }
        
        # 更新用户提供的参数
        default_kwargs.update(kwargs)
        
        return DataLoader(dataset, **default_kwargs)


# 版本兼容性提示
import warnings

def check_dependencies():
    """检查依赖项"""
    try:
        import h5py
        import torch
        import numpy as np
    except ImportError as e:
        warnings.warn(f"Missing required dependency: {e}")
        return False
    
    # 检查版本
    import torch
    if torch.__version__ < "1.8.0":
        warnings.warn("PyTorch version < 1.8.0 may have compatibility issues")
    
    return True


# 模块初始化
check_dependencies()

# 设置日志
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
