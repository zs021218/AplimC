#!/usr/bin/env python3
"""
HDF5专用多模态数据集
支持高效的多模态数据加载和变换
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MultimodalHDF5Dataset(Dataset):
    """
    多模态HDF5数据集
    支持Stokes偏振、荧光信号和多视图图像数据
    """
    
    def __init__(
        self,
        hdf5_path: Union[str, Path],
        split: str = 'train',
        transform: Optional[callable] = None,
        load_modalities: List[str] = ['stokes', 'fluorescence', 'images'],
        cache_size: int = 100,
        memory_map: bool = True
    ):
        """
        初始化HDF5数据集
        
        Args:
            hdf5_path: HDF5文件路径
            split: 数据分割 ('train', 'val', 'test', 'full')
            transform: 数据变换函数
            load_modalities: 要加载的模态列表
            cache_size: 缓存大小
            memory_map: 是否使用内存映射
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform = transform
        self.load_modalities = load_modalities
        self.cache_size = cache_size
        self.memory_map = memory_map
        
        # 验证文件存在
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5文件不存在: {self.hdf5_path}")
        
        # 打开HDF5文件并验证分割
        self._validate_dataset()
        
        # 缓存字典
        self._cache = {}
        
        logger.info(f"初始化HDF5数据集: {self.hdf5_path}")
        logger.info(f"分割: {self.split}, 样本数: {self.length}")
        logger.info(f"加载模态: {self.load_modalities}")
    
    def _validate_dataset(self):
        """验证数据集结构和获取基本信息"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # 检查分割是否存在
            if self.split not in f:
                available_splits = list(f.keys())
                raise ValueError(f"分割 '{self.split}' 不存在。可用分割: {available_splits}")
            
            split_group = f[self.split]
            
            # 检查模态是否存在
            available_modalities = list(split_group.keys())
            for modality in self.load_modalities:
                if modality not in available_modalities:
                    raise ValueError(f"模态 '{modality}' 不存在。可用模态: {available_modalities}")
            
            # 获取数据集长度
            self.length = len(split_group['labels'])
            
            # 获取数据形状信息
            self.shapes = {}
            self.dtypes = {}
            for modality in self.load_modalities:
                self.shapes[modality] = split_group[modality].shape[1:]  # 去除样本维度
                self.dtypes[modality] = split_group[modality].dtype
            
            # 获取类别信息
            if 'class_map' in f.attrs:
                self.class_map = eval(f.attrs['class_map'])
            else:
                # 从标签推断类别数
                unique_labels = np.unique(split_group['labels'][:])
                self.class_map = {f"class_{i}": i for i in unique_labels}
            
            self.num_classes = len(self.class_map)
            
            logger.info(f"数据形状: {self.shapes}")
            logger.info(f"类别数: {self.num_classes}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含各模态数据的字典
        """
        # 检查缓存
        if idx in self._cache:
            sample = self._cache[idx]
        else:
            sample = self._load_sample(idx)
            
            # 缓存管理
            if len(self._cache) < self.cache_size:
                self._cache[idx] = sample
            elif self.cache_size > 0:
                # 移除最旧的缓存项
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._cache[idx] = sample
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """从HDF5文件加载单个样本"""
        sample = {}
        
        # 配置HDF5访问参数
        kwargs = {'swmr': True} if self.memory_map else {}
        
        with h5py.File(self.hdf5_path, 'r', **kwargs) as f:
            split_group = f[self.split]
            
            # 加载各模态数据
            for modality in self.load_modalities:
                data = split_group[modality][idx]
                sample[modality] = torch.from_numpy(data.astype(np.float32))
            
            # 加载标签
            sample['labels'] = torch.tensor(split_group['labels'][idx], dtype=torch.long)
            sample['idx'] = torch.tensor(idx, dtype=torch.long)
        
        return sample
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        高效批量加载
        
        Args:
            indices: 样本索引列表
            
        Returns:
            批量数据字典
        """
        # 确保索引是排序的（HDF5要求）
        sorted_indices = sorted(indices)
        
        batch = {modality: [] for modality in self.load_modalities}
        batch['labels'] = []
        batch['idx'] = []
        
        kwargs = {'swmr': True} if self.memory_map else {}
        
        with h5py.File(self.hdf5_path, 'r', **kwargs) as f:
            split_group = f[self.split]
            
            # 批量加载
            for modality in self.load_modalities:
                data = split_group[modality][sorted_indices]
                batch[modality] = torch.from_numpy(data.astype(np.float32))
            
            labels = split_group['labels'][sorted_indices]
            batch['labels'] = torch.from_numpy(labels.astype(np.int64))
            batch['idx'] = torch.tensor(sorted_indices, dtype=torch.long)
        
        return batch
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        with h5py.File(self.hdf5_path, 'r') as f:
            labels = f[self.split]['labels'][:]
            
        distribution = {}
        for class_name, class_idx in self.class_map.items():
            count = np.sum(labels == class_idx)
            distribution[class_name] = count
        
        return distribution
    
    def get_modality_stats(self, modality: str) -> Dict[str, float]:
        """获取模态统计信息"""
        if modality not in self.load_modalities:
            raise ValueError(f"模态 '{modality}' 未加载")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[self.split][modality]
            
            # 计算统计信息（使用小批量以节省内存）
            batch_size = 1000
            means = []
            stds = []
            
            for i in range(0, len(data), batch_size):
                end_i = min(i + batch_size, len(data))
                batch = data[i:end_i]
                means.append(np.mean(batch, axis=0))
                stds.append(np.std(batch, axis=0))
            
            overall_mean = np.mean(means, axis=0)
            overall_std = np.mean(stds, axis=0)
            
        return {
            'mean': overall_mean.tolist() if overall_mean.ndim > 0 else float(overall_mean),
            'std': overall_std.tolist() if overall_std.ndim > 0 else float(overall_std),
            'shape': self.shapes[modality],
            'dtype': str(self.dtypes[modality])
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")
    
    def __repr__(self) -> str:
        return (f"MultimodalHDF5Dataset("
               f"split={self.split}, "
               f"samples={self.length}, "
               f"modalities={self.load_modalities}, "
               f"classes={self.num_classes})")


class MultimodalSubset(Dataset):
    """
    多模态数据集子集
    用于创建特定索引的子集
    """
    
    def __init__(self, dataset: MultimodalHDF5Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[self.indices[idx]]
    
    def get_batch(self, batch_indices: List[int]) -> Dict[str, torch.Tensor]:
        """批量获取数据"""
        actual_indices = [self.indices[i] for i in batch_indices]
        return self.dataset.get_batch(actual_indices)


def create_datasets(
    hdf5_path: Union[str, Path],
    transform_train: Optional[callable] = None,
    transform_val: Optional[callable] = None,
    load_modalities: List[str] = ['stokes', 'fluorescence', 'images']
) -> Tuple[MultimodalHDF5Dataset, MultimodalHDF5Dataset, MultimodalHDF5Dataset]:
    """
    创建训练、验证、测试数据集
    
    Args:
        hdf5_path: HDF5文件路径
        transform_train: 训练数据变换
        transform_val: 验证/测试数据变换
        load_modalities: 要加载的模态列表
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = MultimodalHDF5Dataset(
        hdf5_path=hdf5_path,
        split='train',
        transform=transform_train,
        load_modalities=load_modalities
    )
    
    val_dataset = MultimodalHDF5Dataset(
        hdf5_path=hdf5_path,
        split='val',
        transform=transform_val,
        load_modalities=load_modalities
    )
    
    test_dataset = MultimodalHDF5Dataset(
        hdf5_path=hdf5_path,
        split='test',
        transform=transform_val,
        load_modalities=load_modalities
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # 测试代码
    hdf5_path = "/data3/zs/AplimC/data/processed/multimodal_data.h5"
    
    # 创建数据集
    dataset = MultimodalHDF5Dataset(hdf5_path, split='train')
    print(f"数据集: {dataset}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"样本键: {sample.keys()}")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, dtype: {value.dtype}")
    
    # 测试类别分布
    distribution = dataset.get_class_distribution()
    print(f"类别分布: {distribution}")
    
    # 测试模态统计
    stats = dataset.get_modality_stats('stokes')
    print(f"Stokes统计: {stats}")
