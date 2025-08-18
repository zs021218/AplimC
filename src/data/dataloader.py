#!/usr/bin/env python3
"""
HDF5优化的DataLoader
提供高效的多模态数据加载和批处理
"""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import Dict, List, Optional, Union, Iterator
import numpy as np
import random
import logging
from .dataset import MultimodalHDF5Dataset, MultimodalSubset

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器
    确保每个批次中各类别样本相对均衡
    """
    
    def __init__(
        self,
        dataset: MultimodalHDF5Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 获取类别分布
        self.class_distribution = dataset.get_class_distribution()
        self.class_indices = self._build_class_indices()
        
        # 计算批次数
        total_samples = len(dataset)
        if self.drop_last:
            self.num_batches = total_samples // batch_size
        else:
            self.num_batches = (total_samples + batch_size - 1) // batch_size
    
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """构建每个类别的样本索引"""
        class_indices = {i: [] for i in range(self.dataset.num_classes)}
        
        # 读取所有标签
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                # 直接从HDF5读取标签，避免加载完整样本
                import h5py
                with h5py.File(self.dataset.hdf5_path, 'r') as f:
                    label = f[self.dataset.split]['labels'][idx]
                class_indices[int(label)].append(idx)
        
        # 打乱每个类别的索引
        if self.shuffle:
            for indices in class_indices.values():
                random.shuffle(indices)
        
        return class_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        # 重新打乱类别索引
        if self.shuffle:
            for indices in self.class_indices.values():
                random.shuffle(indices)
        
        # 计算每个类别在每个批次中的样本数
        num_classes = len(self.class_indices)
        samples_per_class = max(1, self.batch_size // num_classes)
        
        # 创建类别迭代器
        class_iterators = {}
        for class_id, indices in self.class_indices.items():
            # 循环使用索引
            repeated_indices = indices * ((len(self.dataset) // len(indices)) + 1)
            class_iterators[class_id] = iter(repeated_indices)
        
        # 生成批次
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 从每个类别采样
            for class_id in range(num_classes):
                for _ in range(samples_per_class):
                    if len(batch_indices) < self.batch_size:
                        try:
                            idx = next(class_iterators[class_id])
                            batch_indices.append(idx)
                        except StopIteration:
                            # 重新开始迭代
                            class_iterators[class_id] = iter(self.class_indices[class_id])
                            idx = next(class_iterators[class_id])
                            batch_indices.append(idx)
            
            # 如果批次大小不够，随机填充
            while len(batch_indices) < self.batch_size:
                class_id = random.randint(0, num_classes - 1)
                try:
                    idx = next(class_iterators[class_id])
                    batch_indices.append(idx)
                except StopIteration:
                    class_iterators[class_id] = iter(self.class_indices[class_id])
                    idx = next(class_iterators[class_id])
                    batch_indices.append(idx)
            
            # 打乱批次内的顺序
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield batch_indices[:self.batch_size]
    
    def __len__(self) -> int:
        return self.num_batches


class HDF5CollateFunction:
    """
    HDF5数据集的自定义collate函数
    处理多模态数据的批处理
    """
    
    def __init__(self, modalities: List[str] = ['stokes', 'fluorescence', 'images']):
        self.modalities = modalities
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        将样本列表转换为批次张量
        
        Args:
            batch: 样本字典列表
            
        Returns:
            批次数据字典
        """
        collated = {}
        
        # 处理各模态数据
        for modality in self.modalities:
            if modality in batch[0]:
                modality_data = [sample[modality] for sample in batch]
                collated[modality] = torch.stack(modality_data, dim=0)
        
        # 处理标签
        if 'labels' in batch[0]:
            labels = [sample['labels'] for sample in batch]
            collated['labels'] = torch.stack(labels, dim=0)
        
        # 处理索引
        if 'idx' in batch[0]:
            indices = [sample['idx'] for sample in batch]
            collated['idx'] = torch.stack(indices, dim=0)
        
        return collated


def create_dataloaders(
    dataset_train: MultimodalHDF5Dataset,
    dataset_val: MultimodalHDF5Dataset,
    dataset_test: MultimodalHDF5Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = True,
    shuffle_train: bool = True,
    drop_last_train: bool = True
) -> tuple:
    """
    创建优化的数据加载器
    
    Args:
        dataset_train: 训练数据集
        dataset_val: 验证数据集
        dataset_test: 测试数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        balanced_sampling: 是否使用平衡采样
        shuffle_train: 是否打乱训练数据
        drop_last_train: 是否丢弃最后不完整的批次
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # 创建collate函数
    collate_fn = HDF5CollateFunction(modalities=dataset_train.load_modalities)
    
    # 训练数据加载器
    if balanced_sampling:
        train_sampler = BalancedBatchSampler(
            dataset=dataset_train,
            batch_size=batch_size,
            drop_last=drop_last_train,
            shuffle=shuffle_train
        )
        train_loader = DataLoader(
            dataset_train,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0
        )
    else:
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            drop_last=drop_last_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0
        )
    
    # 验证数据加载器
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    # 测试数据加载器
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"创建数据加载器:")
    logger.info(f"  训练: {len(train_loader)} 批次")
    logger.info(f"  验证: {len(val_loader)} 批次")
    logger.info(f"  测试: {len(test_loader)} 批次")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  工作进程: {num_workers}")
    logger.info(f"  平衡采样: {balanced_sampling}")
    
    return train_loader, val_loader, test_loader


class AdaptiveBatchSizer:
    """
    自适应批次大小调整器
    根据GPU内存自动调整批次大小
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 128,
        min_batch_size: int = 4,
        memory_fraction: float = 0.9
    ):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_fraction = memory_fraction
        self.current_batch_size = initial_batch_size
        
    def find_optimal_batch_size(
        self,
        model: torch.nn.Module,
        sample_batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> int:
        """
        通过二分搜索找到最优批次大小
        
        Args:
            model: 模型
            sample_batch: 样本批次
            device: 设备
            
        Returns:
            最优批次大小
        """
        model = model.to(device)
        model.train()
        
        # 清空GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        low, high = self.min_batch_size, self.max_batch_size
        optimal_size = self.min_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # 创建模拟批次
                test_batch = {}
                for key, value in sample_batch.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                        # 复制到目标批次大小
                        repeat_times = max(1, mid // value.shape[0])
                        expanded = value.repeat(repeat_times, *[1] * (len(value.shape) - 1))
                        test_batch[key] = expanded[:mid].to(device)
                    else:
                        test_batch[key] = value.to(device)
                
                # 测试前向传播
                with torch.no_grad():
                    _ = model(test_batch)
                
                # 如果成功，尝试更大的批次
                optimal_size = mid
                low = mid + 1
                
                # 清理
                del test_batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 内存不足，尝试更小的批次
                    high = mid - 1
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise e
        
        self.current_batch_size = optimal_size
        logger.info(f"找到最优批次大小: {optimal_size}")
        
        return optimal_size


def create_efficient_dataloader(
    hdf5_path: str,
    split: str = 'train',
    transform: Optional[callable] = None,
    batch_size: Optional[int] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> DataLoader:
    """
    创建高效的数据加载器，自动优化参数
    
    Args:
        hdf5_path: HDF5文件路径
        split: 数据分割
        transform: 数据变换
        batch_size: 批次大小（None时自动优化）
        model: 模型（用于批次大小优化）
        device: 设备
        **kwargs: 其他DataLoader参数
        
    Returns:
        优化的DataLoader
    """
    # 创建数据集
    dataset = MultimodalHDF5Dataset(
        hdf5_path=hdf5_path,
        split=split,
        transform=transform
    )
    
    # 自动调整批次大小
    if batch_size is None and model is not None and device is not None:
        # 获取样本批次进行测试
        sample = dataset[0]
        sample_batch = {k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        
        batch_sizer = AdaptiveBatchSizer()
        batch_size = batch_sizer.find_optimal_batch_size(model, sample_batch, device)
    elif batch_size is None:
        batch_size = 32  # 默认值
    
    # 设置默认参数
    default_kwargs = {
        'batch_size': batch_size,
        'shuffle': split == 'train',
        'num_workers': 4,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': split == 'train',
        'collate_fn': HDF5CollateFunction(dataset.load_modalities)
    }
    
    # 更新用户提供的参数
    default_kwargs.update(kwargs)
    
    return DataLoader(dataset, **default_kwargs)


def create_balanced_dataloader(
    dataset: MultimodalHDF5Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    创建平衡采样的数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        **kwargs: 其他DataLoader参数
        
    Returns:
        DataLoader对象
    """
    # 创建平衡采样器
    sampler = BalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=kwargs.pop('shuffle', True),
        drop_last=kwargs.pop('drop_last', True)
    )
    
    # 设置默认参数
    default_kwargs = {
        'batch_sampler': sampler,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': HDF5CollateFunction(dataset.load_modalities)
    }
    
    # 更新用户提供的参数
    default_kwargs.update(kwargs)
    
    return DataLoader(dataset, **default_kwargs)


def create_memory_optimized_dataloader(
    dataset: MultimodalHDF5Dataset,
    batch_size: int = 16,
    num_workers: int = 2,
    **kwargs
) -> DataLoader:
    """
    创建内存优化的数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小（较小以节省内存）
        num_workers: 工作进程数（较少以节省内存）
        **kwargs: 其他DataLoader参数
        
    Returns:
        DataLoader对象
    """
    # 设置内存优化参数
    default_kwargs = {
        'batch_size': batch_size,
        'shuffle': kwargs.pop('shuffle', True),
        'num_workers': num_workers,
        'pin_memory': False,  # 禁用pin_memory以节省内存
        'drop_last': kwargs.pop('drop_last', True),
        'collate_fn': HDF5CollateFunction(dataset.load_modalities)
    }
    
    # 更新用户提供的参数
    default_kwargs.update(kwargs)
    
    return DataLoader(dataset, **default_kwargs)


if __name__ == "__main__":
    # 测试代码
    from .dataset import create_datasets
    
    hdf5_path = "/data3/zs/AplimC/data/processed/multimodal_data.h5"
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(hdf5_path)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=16,
        num_workers=2
    )
    
    # 测试批次
    for batch in train_loader:
        print(f"批次键: {batch.keys()}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}, dtype: {value.dtype}")
        break
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
