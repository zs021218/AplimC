import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import MultimodalPolarFluDataset
from .transforms import get_transforms, get_signal_transforms
from typing import Dict, Optional, Tuple, List
import numpy as np
import json
from pathlib import Path

class MultimodalDataLoader:
    """多模态数据加载器管理器"""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        use_weighted_sampling: bool = False,
        pin_memory: bool = True,
        drop_last_train: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ):
        """
        Args:
            data_path: 预处理数据的路径
            batch_size: 批量大小
            num_workers: 数据加载进程数
            use_weighted_sampling: 是否使用加权采样处理类别不平衡
            pin_memory: 是否将数据固定在内存中（GPU训练时推荐）
            drop_last_train: 训练时是否丢弃最后不完整的批次
            persistent_workers: 是否保持工作进程持久化
            prefetch_factor: 每个工作进程预取的批次数
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampling = use_weighted_sampling
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        
        # 获取数据变换
        self.image_transforms = get_transforms()
        self.signal_transforms = get_signal_transforms()
        
        # 验证数据路径
        self._validate_data_path()
        
    def _validate_data_path(self):
        """验证数据路径和文件完整性"""
        required_files = [
            'multimodal_data_train.npz',
            'multimodal_data_val.npz', 
            'multimodal_data_test.npz',
            'metadata_train.json',
            'metadata_val.json',
            'metadata_test.json'
        ]
        
        missing_files = []
        for filename in required_files:
            if not (self.data_path / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(
                f"缺少必要的数据文件: {missing_files}\n"
                f"请确保预处理步骤已完成并生成了分割数据集"
            )
        
        print(f"✓ 数据路径验证通过: {self.data_path}")
        
    def get_dataloader(
        self, 
        split: str = 'train',
        shuffle: Optional[bool] = None,
        apply_augmentation: bool = None
    ) -> DataLoader:
        """
        获取指定分割的数据加载器
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            shuffle: 是否打乱数据，None时自动判断
            apply_augmentation: 是否应用数据增强，None时自动判断
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"无效的数据集分割: {split}，支持: train, val, test")
        
        # 确定是否打乱数据
        if shuffle is None:
            shuffle = (split == 'train')
            
        # 确定是否应用数据增强
        if apply_augmentation is None:
            apply_augmentation = (split == 'train')
        
        # 选择变换
        image_transform = self.image_transforms[split] if apply_augmentation else self.image_transforms['val']
        signal_transform = self.signal_transforms.get('train') if apply_augmentation else None
        
        # 创建数据集
        dataset = MultimodalPolarFluDataset(
            data_path=str(self.data_path),
            split=split,
            transform=image_transform,
            signal_transform=signal_transform
        )
        
        # 创建采样器
        sampler = None
        if self.use_weighted_sampling and split == 'train':
            sampler = self._get_weighted_sampler(dataset)
            shuffle = False  # 使用采样器时不能shuffle
        
        # 确定 drop_last
        drop_last = self.drop_last_train if split == 'train' else False
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2
        )
        
        print(f"✓ {split}集数据加载器创建完成: {len(dataset)} 样本, {len(dataloader)} 批次")
        return dataloader
    
    def _get_weighted_sampler(self, dataset: MultimodalPolarFluDataset) -> WeightedRandomSampler:
        """创建加权采样器以处理类别不平衡"""
        labels = dataset.labels.numpy()
        
        # 计算类别权重
        unique_labels = np.unique(labels)
        class_counts = np.array([np.sum(labels == label) for label in unique_labels])
        
        # 使用倒数作为权重（样本少的类别权重大）
        class_weights = 1.0 / class_counts
        
        # 归一化权重
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # 为每个样本分配权重
        sample_weights = np.zeros(len(labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            sample_weights[mask] = class_weights[i]
        
        print(f"✓ 创建加权采样器，类别权重: {dict(zip(unique_labels, class_weights))}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """自定义批处理函数"""
        # 分离不同模态的数据
        stokes = torch.stack([item['stokes'] for item in batch])
        fluorescence = torch.stack([item['fluorescence'] for item in batch])
        images = torch.stack([item['images'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        sample_ids = torch.tensor([item['sample_id'] for item in batch])
        
        return {
            'stokes': stokes,
            'fluorescence': fluorescence,
            'images': images,
            'labels': labels,
            'sample_ids': sample_ids
        }
    
    def get_all_dataloaders(self) -> Dict[str, DataLoader]:
        """获取所有数据加载器"""
        return {
            'train': self.get_dataloader('train'),
            'val': self.get_dataloader('val'),
            'test': self.get_dataloader('test')
        }
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """获取数据集信息"""
        info = {}
        
        for split in ['train', 'val', 'test']:
            # 加载元数据
            metadata_path = self.data_path / f'metadata_{split}.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            info[split] = {
                'total_samples': metadata['total_samples'],
                'data_shapes': metadata['data_shapes']
            }
        
        return info
    
    def print_dataset_summary(self):
        """打印数据集摘要信息"""
        print(f"\n{'='*60}")
        print("多模态数据集摘要")
        print(f"{'='*60}")
        
        total_samples = 0
        for split in ['train', 'val', 'test']:
            metadata_path = self.data_path / f'metadata_{split}.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            samples = metadata['total_samples']
            total_samples += samples
            
            print(f"{split.upper()}集:")
            print(f"  样本数: {samples:,}")
            print(f"  Stokes形状: {metadata['data_shapes']['stokes']}")
            print(f"  荧光形状: {metadata['data_shapes']['fluorescence']}")
            print(f"  图像形状: {metadata['data_shapes']['images']}")
            print()
        
        print(f"总样本数: {total_samples:,}")
        print(f"批量大小: {self.batch_size}")
        print(f"工作进程数: {self.num_workers}")
        print(f"加权采样: {'启用' if self.use_weighted_sampling else '禁用'}")


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    便捷函数：创建训练、验证和测试数据加载器
    
    Args:
        data_path: 预处理数据的路径
        batch_size: 批量大小
        num_workers: 数据加载进程数
        **kwargs: 其他传递给 MultimodalDataLoader 的参数
    
    Returns:
        训练、验证、测试数据加载器的元组
    """
    loader_manager = MultimodalDataLoader(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
    
    # 打印数据集摘要
    loader_manager.print_dataset_summary()
    
    dataloaders = loader_manager.get_all_dataloaders()
    return dataloaders['train'], dataloaders['val'], dataloaders['test']


class DataLoaderConfig:
    """数据加载器配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'batch_size': 32,
            'num_workers': 4,
            'use_weighted_sampling': False,
            'pin_memory': True,
            'drop_last_train': True,
            'persistent_workers': True,
            'prefetch_factor': 2
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def create_dataloader_manager(self, data_path: str) -> MultimodalDataLoader:
        """使用配置创建数据加载器管理器"""
        return MultimodalDataLoader(data_path=data_path, **self.config)


def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
    """
    基准测试数据加载器性能
    
    Args:
        dataloader: 要测试的数据加载器
        num_batches: 测试的批次数
        
    Returns:
        性能统计信息
    """
    import time
    
    print(f"开始数据加载器性能测试...")
    print(f"测试批次数: {num_batches}")
    
    times = []
    data_transfer_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        start_time = time.time()
        
        # 模拟数据传输到GPU
        if torch.cuda.is_available():
            gpu_start = time.time()
            _ = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            data_transfer_times.append(time.time() - gpu_start)
        
        batch_time = time.time() - start_time
        times.append(batch_time)
        
        if (i + 1) % 5 == 0:
            print(f"  批次 {i+1}/{num_batches}: {batch_time:.4f}s")
    
    avg_time = np.mean(times)
    avg_transfer_time = np.mean(data_transfer_times) if data_transfer_times else 0
    throughput = dataloader.batch_size / avg_time
    
    stats = {
        'avg_batch_time': avg_time,
        'avg_transfer_time': avg_transfer_time,
        'throughput_samples_per_sec': throughput,
        'total_time': sum(times)
    }
    
    print(f"\n性能测试结果:")
    print(f"  平均批次时间: {avg_time:.4f}s")
    print(f"  平均传输时间: {avg_transfer_time:.4f}s")
    print(f"  吞吐量: {throughput:.2f} 样本/秒")
    
    return stats


if __name__ == "__main__":
    # 测试数据加载器
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.append('/data3/zs/AplimC')
    
    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path='/data3/zs/AplimC/data/processed',
            batch_size=8,
            num_workers=2,
            use_weighted_sampling=True
        )
        
        print(f"\n数据加载器创建成功!")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        print(f"\n测试批次数据:")
        for batch in train_loader:
            print(f"  Stokes形状: {batch['stokes'].shape}")
            print(f"  荧光形状: {batch['fluorescence'].shape}")
            print(f"  图像形状: {batch['images'].shape}")
            print(f"  标签形状: {batch['labels'].shape}")
            print(f"  样本ID形状: {batch['sample_ids'].shape}")
            
            # 检查数据范围
            print(f"\n数据范围检查:")
            print(f"  Stokes: [{batch['stokes'].min():.3f}, {batch['stokes'].max():.3f}]")
            print(f"  荧光: [{batch['fluorescence'].min():.3f}, {batch['fluorescence'].max():.3f}]")
            print(f"  图像: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
            print(f"  标签范围: {batch['labels'].unique()}")
            break
        
        # 性能测试
        print(f"\n开始性能测试...")
        benchmark_dataloader(train_loader, num_batches=5)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
