import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from PIL import Image
import torchvision.transforms as transforms

class MultimodalPolarFluDataset(Dataset):
    """多模态偏振荧光数据集（预分割版本）"""
    
    def __init__(
        self, 
        data_path: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        signal_transform: Optional[callable] = None
    ):
        """
        Args:
            data_path: 处理后数据的路径
            split: 数据集分割 ('train', 'val', 'test', 'full')
            transform: 图像变换
            signal_transform: 信号数据变换
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.signal_transform = signal_transform
        
        # 加载预分割的数据
        self._load_split_data()
        
    def _load_split_data(self):
        """加载预分割的数据"""
        if self.split == 'full':
            # 加载完整数据集
            data_file = self.data_path / 'multimodal_data_full.npz'
            metadata_file = self.data_path / 'metadata.json'
        else:
            # 加载预分割的数据
            data_file = self.data_path / f'multimodal_data_{self.split}.npz'
            metadata_file = self.data_path / f'metadata_{self.split}.json'
        
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        # 加载数据
        data = np.load(data_file, allow_pickle=True)
        
        # 提取数据
        self.stokes_data = torch.FloatTensor(data['stokes'])
        self.fluorescence_data = torch.FloatTensor(data['fluorescence'])
        self.images = data['images']  # 保持为numpy，在__getitem__中转换
        self.labels = torch.LongTensor(data['labels'])
        
        # 加载元数据
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            # 如果没有分割特定的元数据，加载通用元数据
            general_metadata_file = self.data_path / 'metadata.json'
            if general_metadata_file.exists():
                with open(general_metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                # 创建基本元数据
                unique_labels = torch.unique(self.labels).numpy()
                self.metadata = {
                    'split': self.split,
                    'total_samples': len(self.labels),
                    'unique_labels': unique_labels.tolist()
                }
        
        print(f"{self.split}集加载完成: {len(self)} 个样本")
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 获取信号数据
        stokes = self.stokes_data[idx]
        fluorescence = self.fluorescence_data[idx]
        
        # 应用信号变换
        if self.signal_transform:
            stokes = self.signal_transform(stokes)
            fluorescence = self.signal_transform(fluorescence)
        
        # 获取图像数据 (3, 224, 224, 3) -> (3, 3, 224, 224)
        images = self.images[idx]  # shape: (3, 224, 224, 3)
        
        # 转换图像格式和应用变换
        processed_images = []
        for view_idx in range(images.shape[0]):
            img = images[view_idx]  # (224, 224, 3)
            
            # 转换为PIL图像
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            
            # 应用变换
            if self.transform:
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = transforms.ToTensor()(img_pil)
                
            processed_images.append(img_tensor)
        
        # 堆叠图像 (3, 3, 224, 224)
        images_tensor = torch.stack(processed_images, dim=0)
        
        # 获取标签
        label = self.labels[idx]
        
        return {
            'stokes': stokes,
            'fluorescence': fluorescence,
            'images': images_tensor,
            'label': label,
            'sample_id': idx
        }
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        if hasattr(self.metadata, 'class_map') and self.metadata.get('class_map'):
            return list(self.metadata['class_map'].keys())
        else:
            # 如果没有class_map，返回默认类别名称
            return [f"Class_{i}" for i in range(self.get_num_classes())]
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        if hasattr(self.metadata, 'class_map') and self.metadata.get('class_map'):
            return len(self.metadata['class_map'])
        else:
            # 从标签推断类别数量
            return len(torch.unique(self.labels))
    
    def get_sample_counts(self) -> Dict[str, int]:
        """获取各类别样本数量"""
        if hasattr(self.metadata, 'sample_counts') and self.metadata.get('sample_counts'):
            return self.metadata['sample_counts']
        else:
            # 动态计算样本数量
            counts = {}
            class_names = self.get_class_names()
            unique_labels = torch.unique(self.labels)
            
            for i, label in enumerate(unique_labels):
                count = (self.labels == label).sum().item()
                class_name = class_names[i] if i < len(class_names) else f"Class_{label}"
                counts[class_name] = count
            
            return counts


def get_dataset_statistics(data_path: str, split: str = 'train') -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        data_path: 数据路径
        split: 数据集分割 ('train', 'val', 'test')
    
    Returns:
        包含数据集统计信息的字典
    """
    try:
        # 创建数据集实例
        dataset = MultimodalPolarFluDataset(
            data_path=data_path,
            split=split,
            transform=None
        )
        
        # 计算统计信息
        stats = {
            'split': split,
            'total_samples': len(dataset),
            'num_classes': dataset.get_num_classes(),
            'class_names': dataset.get_class_names(),
            'sample_counts': dataset.get_sample_counts(),
            'data_shapes': {}
        }
        
        # 获取数据形状信息
        if len(dataset) > 0:
            sample = dataset[0]
            stats['data_shapes'] = {
                'stokes': list(sample['stokes'].shape),
                'fluorescence': list(sample['fluorescence'].shape),
                'images': list(sample['images'].shape),
                'labels': []  # 标量
            }
        
        # 计算类别分布
        labels = dataset.labels.numpy()
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        stats['class_distribution'] = {}
        for label, count in zip(unique_labels, counts):
            class_name = dataset.get_class_names()[label] if label < len(dataset.get_class_names()) else f"Class_{label}"
            stats['class_distribution'][class_name] = {
                'count': int(count),
                'percentage': float(count / len(labels) * 100)
            }
        
        return stats
        
    except Exception as e:
        print(f"获取数据集统计信息失败: {e}")
        return {
            'split': split,
            'error': str(e),
            'total_samples': 0,
            'num_classes': 0
        }


def load_all_dataset_statistics(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载所有分割的数据集统计信息
    
    Args:
        data_path: 数据路径
    
    Returns:
        包含所有分割统计信息的字典
    """
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    for split in splits:
        print(f"加载 {split} 数据集统计信息...")
        all_stats[split] = get_dataset_statistics(data_path, split)
    
    # 计算总体统计
    total_samples = sum(stats['total_samples'] for stats in all_stats.values())
    all_stats['summary'] = {
        'total_samples': total_samples,
        'splits': {
            split: {
                'samples': stats['total_samples'],
                'percentage': stats['total_samples'] / total_samples * 100 if total_samples > 0 else 0
            }
            for split, stats in all_stats.items()
            if split != 'summary'
        }
    }
    
    return all_stats


