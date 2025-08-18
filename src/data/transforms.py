#!/usr/bin/env python3
"""
多模态数据变换
支持信号数据和图像数据的各种变换操作
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger(__name__)


class BaseTransform:
    """基础变换类"""
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}([\n"
        for t in self.transforms:
            format_string += f"    {t},\n"
        format_string += "])"
        return format_string


# ===============================
# 信号数据变换
# ===============================

class SignalNormalize(BaseTransform):
    """信号数据归一化"""
    
    def __init__(
        self,
        modalities: List[str] = ['stokes', 'fluorescence'],
        method: str = 'zscore',  # 'zscore', 'minmax', 'robust'
        per_channel: bool = True
    ):
        self.modalities = modalities
        self.method = method
        self.per_channel = per_channel
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for modality in self.modalities:
            if modality in sample:
                data = sample[modality]
                
                if self.method == 'zscore':
                    if self.per_channel:
                        # 按通道归一化
                        mean = data.mean(dim=-1, keepdim=True)
                        std = data.std(dim=-1, keepdim=True)
                        sample[modality] = (data - mean) / (std + 1e-8)
                    else:
                        # 全局归一化
                        mean = data.mean()
                        std = data.std()
                        sample[modality] = (data - mean) / (std + 1e-8)
                
                elif self.method == 'minmax':
                    if self.per_channel:
                        min_val = data.min(dim=-1, keepdim=True)[0]
                        max_val = data.max(dim=-1, keepdim=True)[0]
                        sample[modality] = (data - min_val) / (max_val - min_val + 1e-8)
                    else:
                        min_val = data.min()
                        max_val = data.max()
                        sample[modality] = (data - min_val) / (max_val - min_val + 1e-8)
                
                elif self.method == 'robust':
                    if self.per_channel:
                        median = data.median(dim=-1, keepdim=True)[0]
                        mad = torch.median(torch.abs(data - median), dim=-1, keepdim=True)[0]
                        sample[modality] = (data - median) / (mad + 1e-8)
                    else:
                        median = data.median()
                        mad = torch.median(torch.abs(data - median))
                        sample[modality] = (data - median) / (mad + 1e-8)
        
        return sample
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modalities={self.modalities}, method={self.method})"


class SignalClip(BaseTransform):
    """信号数据裁剪"""
    
    def __init__(
        self,
        modalities: List[str] = ['stokes', 'fluorescence'],
        min_val: float = -5.0,
        max_val: float = 5.0
    ):
        self.modalities = modalities
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for modality in self.modalities:
            if modality in sample:
                sample[modality] = torch.clamp(sample[modality], self.min_val, self.max_val)
        return sample


class SignalRandomCrop(BaseTransform):
    """信号随机裁剪"""
    
    def __init__(
        self,
        modalities: List[str] = ['stokes', 'fluorescence'],
        crop_length: int = 3000,
        min_length: int = 2000
    ):
        self.modalities = modalities
        self.crop_length = crop_length
        self.min_length = min_length
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for modality in self.modalities:
            if modality in sample:
                data = sample[modality]
                signal_length = data.shape[-1]
                
                if signal_length > self.crop_length:
                    # 随机选择起始位置
                    start = random.randint(0, signal_length - self.crop_length)
                    sample[modality] = data[..., start:start + self.crop_length]
                elif signal_length < self.min_length:
                    # 填充到最小长度
                    pad_length = self.min_length - signal_length
                    sample[modality] = F.pad(data, (0, pad_length), mode='reflect')
        
        return sample


class SignalGaussianNoise(BaseTransform):
    """信号添加高斯噪声"""
    
    def __init__(
        self,
        modalities: List[str] = ['stokes', 'fluorescence'],
        noise_std: float = 0.01,
        prob: float = 0.5
    ):
        self.modalities = modalities
        self.noise_std = noise_std
        self.prob = prob
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.prob:
            for modality in self.modalities:
                if modality in sample:
                    data = sample[modality]
                    noise = torch.randn_like(data) * self.noise_std
                    sample[modality] = data + noise
        
        return sample


class SignalSmoothing(BaseTransform):
    """信号平滑"""
    
    def __init__(
        self,
        modalities: List[str] = ['stokes', 'fluorescence'],
        kernel_size: int = 5,
        sigma: float = 1.0
    ):
        self.modalities = modalities
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # 创建高斯核
        self.gaussian_kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """创建1D高斯核"""
        x = torch.arange(self.kernel_size, dtype=torch.float32)
        x = x - (self.kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / self.sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for modality in self.modalities:
            if modality in sample:
                data = sample[modality]
                original_shape = data.shape
                
                # 重塑为(batch, channel, length)格式
                if data.dim() == 2:  # (channels, length)
                    data = data.unsqueeze(0)  # (1, channels, length)
                elif data.dim() == 1:  # (length,)
                    data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, length)
                
                # 应用卷积平滑
                smoothed = F.conv1d(data, self.gaussian_kernel, padding=self.kernel_size//2)
                
                # 恢复原始形状
                sample[modality] = smoothed.squeeze().view(original_shape)
        
        return sample


# ===============================
# 图像数据变换
# ===============================

class ImageNormalize(BaseTransform):
    """图像归一化"""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.mean = torch.tensor(mean).view(1, 1, 1, 3)
        self.std = torch.tensor(std).view(1, 1, 1, 3)
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'images' in sample:
            # 假设images shape: (num_views, H, W, 3)
            images = sample['images']
            sample['images'] = (images - self.mean) / self.std
        
        return sample


class ImageRandomRotation(BaseTransform):
    """图像随机旋转"""
    
    def __init__(self, degrees: float = 15.0, prob: float = 0.5):
        self.degrees = degrees
        self.prob = prob
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.prob and 'images' in sample:
            images = sample['images']  # (num_views, H, W, 3)
            
            # 对每个视图应用相同的旋转
            angle = random.uniform(-self.degrees, self.degrees)
            angle_rad = np.deg2rad(angle)
            
            # 创建旋转矩阵
            cos_val = np.cos(angle_rad)
            sin_val = np.sin(angle_rad)
            rotation_matrix = torch.tensor([
                [cos_val, -sin_val, 0],
                [sin_val, cos_val, 0]
            ], dtype=torch.float32)
            
            # 应用旋转
            rotated_images = []
            for view_idx in range(images.shape[0]):
                view = images[view_idx].permute(2, 0, 1)  # (3, H, W)
                grid = F.affine_grid(
                    rotation_matrix.unsqueeze(0),
                    view.unsqueeze(0).shape,
                    align_corners=False
                )
                rotated_view = F.grid_sample(
                    view.unsqueeze(0),
                    grid,
                    align_corners=False
                ).squeeze(0)
                rotated_images.append(rotated_view.permute(1, 2, 0))  # (H, W, 3)
            
            sample['images'] = torch.stack(rotated_images, dim=0)
        
        return sample


class ImageRandomFlip(BaseTransform):
    """图像随机翻转"""
    
    def __init__(self, horizontal_prob: float = 0.5, vertical_prob: float = 0.2):
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'images' in sample:
            images = sample['images']
            
            # 水平翻转
            if random.random() < self.horizontal_prob:
                images = torch.flip(images, dims=[2])  # 沿宽度翻转
            
            # 垂直翻转
            if random.random() < self.vertical_prob:
                images = torch.flip(images, dims=[1])  # 沿高度翻转
            
            sample['images'] = images
        
        return sample


class ImageColorJitter(BaseTransform):
    """图像颜色抖动"""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        prob: float = 0.5
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.prob = prob
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.prob and 'images' in sample:
            images = sample['images']  # (num_views, H, W, 3)
            
            # 亮度调整
            if self.brightness > 0:
                brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                images = images * brightness_factor
            
            # 对比度调整
            if self.contrast > 0:
                contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                mean = images.mean(dim=[1, 2], keepdim=True)
                images = (images - mean) * contrast_factor + mean
            
            # 饱和度调整（简化版本）
            if self.saturation > 0:
                saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
                gray = images.mean(dim=-1, keepdim=True)
                images = gray + (images - gray) * saturation_factor
            
            # 裁剪到有效范围
            images = torch.clamp(images, 0, 1)
            sample['images'] = images
        
        return sample


# ===============================
# 混合变换
# ===============================

class RandomMixUp(BaseTransform):
    """随机MixUp数据增强"""
    
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.3,
        modalities: List[str] = ['stokes', 'fluorescence', 'images']
    ):
        self.alpha = alpha
        self.prob = prob
        self.modalities = modalities
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 注意：这个变换需要批次级别的实现
        # 这里提供单样本的框架
        if random.random() < self.prob:
            # 在实际使用中，需要访问其他样本进行混合
            # 这里只是添加小幅度噪声作为近似
            lambda_val = np.random.beta(self.alpha, self.alpha)
            
            for modality in self.modalities:
                if modality in sample:
                    noise = torch.randn_like(sample[modality]) * 0.01
                    sample[modality] = lambda_val * sample[modality] + (1 - lambda_val) * noise
        
        return sample


class ToDevice(BaseTransform):
    """将数据移动到指定设备"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()}


# ===============================
# 预定义变换组合
# ===============================

def get_train_transforms(config: Optional[Dict] = None) -> Compose:
    """获取训练时的数据变换"""
    if config is None:
        config = {}
    
    transforms = [
        # 信号变换
        SignalNormalize(
            method=config.get('signal_norm_method', 'zscore'),
            per_channel=config.get('signal_norm_per_channel', True)
        ),
        SignalRandomCrop(
            crop_length=config.get('signal_crop_length', 3800)
        ),
        SignalGaussianNoise(
            noise_std=config.get('signal_noise_std', 0.005),
            prob=config.get('signal_noise_prob', 0.3)
        ),
        
        # 图像变换
        ImageRandomRotation(
            degrees=config.get('image_rotation_degrees', 10.0),
            prob=config.get('image_rotation_prob', 0.4)
        ),
        ImageRandomFlip(
            horizontal_prob=config.get('image_hflip_prob', 0.5),
            vertical_prob=config.get('image_vflip_prob', 0.2)
        ),
        ImageColorJitter(
            brightness=config.get('image_brightness', 0.1),
            contrast=config.get('image_contrast', 0.1),
            saturation=config.get('image_saturation', 0.1),
            prob=config.get('image_jitter_prob', 0.3)
        ),
        ImageNormalize()
    ]
    
    return Compose(transforms)


def get_val_transforms(config: Optional[Dict] = None) -> Compose:
    """获取验证/测试时的数据变换"""
    if config is None:
        config = {}
    
    transforms = [
        # 信号变换（仅标准化）
        SignalNormalize(
            method=config.get('signal_norm_method', 'zscore'),
            per_channel=config.get('signal_norm_per_channel', True)
        ),
        
        # 图像变换（仅标准化）
        ImageNormalize()
    ]
    
    return Compose(transforms)


if __name__ == "__main__":
    # 测试变换
    # 创建模拟数据
    sample = {
        'stokes': torch.randn(4, 4000),
        'fluorescence': torch.randn(16, 4000),
        'images': torch.rand(3, 224, 224, 3),
        'labels': torch.tensor(5)
    }
    
    print("原始数据形状:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # 测试训练变换
    train_transform = get_train_transforms()
    transformed_sample = train_transform(sample)
    
    print("\n变换后数据形状:")
    for key, value in transformed_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # 测试验证变换
    val_transform = get_val_transforms()
    val_sample = val_transform(sample)
    
    print("\n验证变换后数据形状:")
    for key, value in val_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
