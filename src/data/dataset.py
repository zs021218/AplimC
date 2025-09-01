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
        memory_map: bool = True,
        selected_classes: Optional[Union[List[str], List[int]]] = None
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
            selected_classes: 选择的类别列表（类别名称或类别ID）
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform = transform
        self.load_modalities = load_modalities
        self.cache_size = cache_size
        self.memory_map = memory_map
        self.selected_classes = selected_classes
        
        # 验证文件存在
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5文件不存在: {self.hdf5_path}")
        
        # 打开HDF5文件并验证分割
        self._validate_dataset()
        
        # 过滤类别（如果指定）
        self._filter_classes()
        
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
                try:
                    self.class_map = eval(f.attrs['class_map'])
                except:
                    # 如果解析失败，使用默认类别映射
                    self.class_map = self._get_default_class_map()
            else:
                # 使用默认类别映射
                self.class_map = self._get_default_class_map()
            
            self.num_classes = len(self.class_map)
            
            logger.info(f"数据形状: {self.shapes}")
            logger.info(f"类别数: {self.num_classes}")
    
    def _get_default_class_map(self):
        """获取默认的类别映射"""
        return {
            'CG': 0, 'IG': 1, 'PS3': 2, 'PS6': 3, 'PS10': 4, 'QDDB': 5,
            'QZQG': 6, 'SG': 7, 'TP': 8, 'TS': 9, 'YMXH': 10, 'YXXB': 11
        }
    
    def _filter_classes(self):
        """根据选择的类别过滤数据集"""
        if self.selected_classes is None:
            # 不过滤，使用所有样本
            self.valid_indices = np.arange(self.length)
            self.original_length = self.length
            return
        
        # 读取所有标签
        with h5py.File(self.hdf5_path, 'r') as f:
            all_labels = f[self.split]['labels'][:]
        
        # 转换选择的类别为标签ID
        if isinstance(self.selected_classes[0], str):
            # 如果是类别名称，转换为ID
            selected_label_ids = []
            for class_name in self.selected_classes:
                if class_name in self.class_map:
                    selected_label_ids.append(self.class_map[class_name])
                else:
                    raise ValueError(f"未知类别名称: {class_name}")
        else:
            # 如果是类别ID，直接使用
            selected_label_ids = list(self.selected_classes)
        
        # 找到匹配的样本索引
        self.valid_indices = []
        for i, label in enumerate(all_labels):
            if label in selected_label_ids:
                self.valid_indices.append(i)
        
        self.valid_indices = np.array(self.valid_indices)
        self.original_length = self.length
        self.length = len(self.valid_indices)
        
        # 更新类别映射（重新编号）
        if len(selected_label_ids) < len(self.class_map):
            # 创建新的类别映射
            old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(selected_label_ids))}
            
            # 更新class_map
            new_class_map = {}
            for class_name, old_id in self.class_map.items():
                if old_id in old_to_new:
                    new_class_map[class_name] = old_to_new[old_id]
            
            self.class_map = new_class_map
            self.num_classes = len(selected_label_ids)
            self.label_mapping = old_to_new  # 保存标签映射关系
        
        logger.info(f"类别过滤完成:")
        logger.info(f"  原始样本数: {self.original_length}")
        logger.info(f"  过滤后样本数: {self.length}")
        logger.info(f"  选择的类别: {self.selected_classes}")
        logger.info(f"  更新后的类别映射: {self.class_map}")
    
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
        # 如果有类别过滤，使用映射后的真实索引
        real_idx = self.valid_indices[idx] if hasattr(self, 'valid_indices') else idx
        
        sample = {}
        
        # 配置HDF5访问参数
        kwargs = {'swmr': True} if self.memory_map else {}
        
        with h5py.File(self.hdf5_path, 'r', **kwargs) as f:
            split_group = f[self.split]
            
            # 加载各模态数据
            for modality in self.load_modalities:
                data = split_group[modality][real_idx]
                sample[modality] = torch.from_numpy(data.astype(np.float32))
            
            # 加载标签
            original_label = split_group['labels'][real_idx]
            
            # 如果有标签映射，转换标签
            if hasattr(self, 'label_mapping') and original_label in self.label_mapping:
                mapped_label = self.label_mapping[original_label]
            else:
                mapped_label = original_label
            
            sample['labels'] = torch.tensor(mapped_label, dtype=torch.long)
            sample['idx'] = torch.tensor(idx, dtype=torch.long)  # 使用逻辑索引
            sample['real_idx'] = torch.tensor(real_idx, dtype=torch.long)  # 保存真实索引
        
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
