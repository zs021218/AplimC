"""
数据变换定义
"""

import torch
import torchvision.transforms as transforms
from typing import Dict, Callable
import numpy as np

def get_transforms() -> Dict[str, transforms.Compose]:
    """获取不同数据集分割的图像变换"""
    
    # 计算ImageNet均值和标准差（用于预训练模型）
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # 验证和测试时的变换（无数据增强）
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return {
        'train': train_transform,
        'val': val_test_transform,
        'test': val_test_transform,
        'full': val_test_transform  # 完整数据集使用无增强变换
    }

def get_signal_transforms() -> Dict[str, Callable]:
    """信号数据的变换函数"""
    
    def normalize_signal(signal: torch.Tensor) -> torch.Tensor:
        """归一化信号数据到均值0，标准差1"""
        mean = signal.mean(dim=-1, keepdim=True)
        std = signal.std(dim=-1, keepdim=True)
        return (signal - mean) / (std + 1e-8)
    
    def standardize_signal(signal: torch.Tensor) -> torch.Tensor:
        """标准化信号数据到[0,1]范围"""
        min_val = signal.min(dim=-1, keepdim=True)[0]
        max_val = signal.max(dim=-1, keepdim=True)[0]
        return (signal - min_val) / (max_val - min_val + 1e-8)
    
    def add_noise(signal: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """添加高斯噪声进行数据增强"""
        noise = torch.randn_like(signal) * noise_level
        return signal + noise
    
    def smooth_signal(signal: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """使用移动平均平滑信号"""
        if kernel_size <= 1:
            return signal
        
        # 创建移动平均核
        kernel = torch.ones(kernel_size) / kernel_size
        
        # 对信号进行卷积（每个通道分别处理）
        if signal.dim() == 2:  # (channels, length)
            smoothed = torch.zeros_like(signal)
            for i in range(signal.shape[0]):
                # 使用torch.nn.functional.conv1d进行一维卷积
                sig = signal[i:i+1].unsqueeze(0)  # (1, 1, length)
                kern = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)
                smoothed_sig = torch.nn.functional.conv1d(
                    sig, kern, padding=kernel_size//2
                )
                smoothed[i] = smoothed_sig.squeeze()
            return smoothed
        else:
            return signal
    
    return {
        'normalize': normalize_signal,
        'standardize': standardize_signal,
        'add_noise': add_noise,
        'smooth': smooth_signal
    }

class SignalAugmentation:
    """信号数据增强类"""
    
    def __init__(
        self, 
        apply_noise: bool = True,
        noise_level: float = 0.01,
        apply_smooth: bool = False,
        smooth_kernel_size: int = 3,
        apply_normalize: bool = True
    ):
        self.apply_noise = apply_noise
        self.noise_level = noise_level
        self.apply_smooth = apply_smooth
        self.smooth_kernel_size = smooth_kernel_size
        self.apply_normalize = apply_normalize
        
        self.transforms = get_signal_transforms()
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """应用信号变换"""
        if self.apply_normalize:
            signal = self.transforms['normalize'](signal)
        
        if self.apply_smooth:
            signal = self.transforms['smooth'](signal, self.smooth_kernel_size)
        
        if self.apply_noise:
            signal = self.transforms['add_noise'](signal, self.noise_level)
        
        return signal

def get_custom_transforms(config: Dict) -> Dict[str, transforms.Compose]:
    """根据配置创建自定义变换"""
    
    # 默认配置
    default_config = {
        'image_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'train_augmentation': {
            'rotation': 10,
            'horizontal_flip': 0.5,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }
        }
    }
    
    # 更新配置
    default_config.update(config)
    
    # 构建变换
    augment_config = default_config['train_augmentation']
    
    train_transforms = [
        transforms.Resize((default_config['image_size'], default_config['image_size'])),
    ]
    
    # 添加数据增强
    if augment_config.get('rotation', 0) > 0:
        train_transforms.append(transforms.RandomRotation(augment_config['rotation']))
    
    if augment_config.get('horizontal_flip', 0) > 0:
        train_transforms.append(transforms.RandomHorizontalFlip(augment_config['horizontal_flip']))
    
    if augment_config.get('color_jitter'):
        cj = augment_config['color_jitter']
        train_transforms.append(transforms.ColorJitter(
            brightness=cj.get('brightness', 0),
            contrast=cj.get('contrast', 0),
            saturation=cj.get('saturation', 0),
            hue=cj.get('hue', 0)
        ))
    
    # 添加基础变换
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=default_config['mean'], 
            std=default_config['std']
        )
    ])
    
    # 验证/测试变换
    val_test_transforms = [
        transforms.Resize((default_config['image_size'], default_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=default_config['mean'], 
            std=default_config['std']
        )
    ]
    
    return {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(val_test_transforms),
        'test': transforms.Compose(val_test_transforms),
        'full': transforms.Compose(val_test_transforms)
    }
