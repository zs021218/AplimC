#!/usr/bin/env python3
"""
模型配置管理
支持从YAML文件加载配置，预定义常用配置
"""

import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    # 基本参数
    num_classes: int = 12
    modalities: List[str] = field(default_factory=lambda: ['stokes', 'fluorescence', 'images'])
    
    # 模态维度配置
    stokes_dim: int = 4
    stokes_length: int = 4000
    fluorescence_dim: int = 16
    fluorescence_length: int = 4000
    image_channels: int = 3
    image_size: int = 224
    num_views: int = 3
    
    # 网络结构
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    activation: str = 'relu'
    
    # 融合策略
    fusion_strategy: str = 'late'  # 'early', 'late', 'hierarchical'
    
    # 训练参数
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    
    # 正则化
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 20
    save_every: int = 10
    
    # 优化器和调度器
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    
    # 损失函数
    loss_weights: Optional[List[float]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        # 展平嵌套的配置
        flat_dict = {}
        
        if 'model' in config_dict:
            flat_dict.update(config_dict['model'])
        
        if 'training' in config_dict:
            training_config = config_dict['training']
            flat_dict.update(training_config)
        
        if 'data' in config_dict:
            data_config = config_dict['data']
            # 只取我们需要的数据配置
            if 'hdf5_path' in data_config:
                flat_dict['hdf5_path'] = data_config['hdf5_path']
        
        # 创建实例时只使用类中定义的字段
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in flat_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ModelConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ModelConfig':
        """从文件加载配置（自动检测格式）"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return cls.from_yaml(file_path)
        elif file_path.suffix.lower() == '.py':
            # 导入Python配置文件
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", file_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            if hasattr(config_module, 'get_config'):
                return config_module.get_config()
            else:
                raise ValueError("Python配置文件必须包含get_config()函数")
        else:
            raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """保存为YAML文件"""
        config_dict = {
            'model': {
                'num_classes': self.num_classes,
                'modalities': self.modalities,
                'stokes_dim': self.stokes_dim,
                'stokes_length': self.stokes_length,
                'fluorescence_dim': self.fluorescence_dim,
                'fluorescence_length': self.fluorescence_length,
                'image_channels': self.image_channels,
                'image_size': self.image_size,
                'num_views': self.num_views,
                'hidden_dims': self.hidden_dims,
                'dropout_rate': self.dropout_rate,
                'use_batch_norm': self.use_batch_norm,
                'activation': self.activation,
                'fusion_strategy': self.fusion_strategy
            },
            'training': {
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_workers': self.num_workers,
                'gradient_clip_val': self.gradient_clip_val,
                'early_stopping_patience': self.early_stopping_patience,
                'save_every': self.save_every,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ModelConfig(modalities={self.modalities}, num_classes={self.num_classes}, fusion={self.fusion_strategy})"


# 预定义配置


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"从 {config_path} 加载配置")
    return config


def create_model_config_from_dict(config_dict: Dict[str, Any]) -> ModelConfig:
    """从字典创建模型配置"""
    # 提取模型相关配置
    model_config = config_dict.get('model', {})
    
    # 设置默认值
    defaults = {
        'num_classes': 12,
        'use_stokes': True,
        'use_fluorescence': True,
        'use_images': True,
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'activation': 'relu',
        'fusion_strategy': 'concat'
    }
    
    # 合并配置
    for key, default_value in defaults.items():
        if key not in model_config:
            model_config[key] = default_value
    
    return ModelConfig(**model_config)


# 预定义配置

def create_simple_config() -> ModelConfig:
    """创建简单配置"""
    return ModelConfig(
        num_classes=12,
        modalities=['stokes', 'fluorescence', 'images'],
        hidden_dims=[256, 128],
        dropout_rate=0.2,
        fusion_strategy='late',
        batch_size=32,
        max_epochs=50,
        learning_rate=1e-3
    )

def create_lightweight_config() -> ModelConfig:
    """创建轻量级配置（不使用图像）"""
    return ModelConfig(
        num_classes=12,
        modalities=['stokes', 'fluorescence'],
        hidden_dims=[128, 64],
        dropout_rate=0.1,
        fusion_strategy='late',
        batch_size=64,
        max_epochs=80,
        learning_rate=2e-3
    )

def create_signal_only_config() -> ModelConfig:
    """创建仅信号配置"""
    return ModelConfig(
        num_classes=12,
        modalities=['stokes', 'fluorescence'],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
        fusion_strategy='early',
        batch_size=64,
        max_epochs=100,
        learning_rate=1e-3
    )

def create_image_only_config() -> ModelConfig:
    """创建仅图像配置"""
    return ModelConfig(
        num_classes=12,
        modalities=['images'],
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3,
        fusion_strategy='late',
        batch_size=16,
        max_epochs=100,
        learning_rate=5e-4
    )

# 预定义配置字典
PRESET_CONFIGS = {
    'simple': {
        'modalities': ['stokes', 'fluorescence', 'images'],
        'hidden_dims': [256, 128],
        'dropout_rate': 0.2,
        'fusion_strategy': 'late',
        'batch_size': 32,
        'max_epochs': 50,
        'learning_rate': 1e-3
    },
    'lightweight': {
        'modalities': ['stokes', 'fluorescence'],
        'hidden_dims': [128, 64],
        'dropout_rate': 0.1,
        'fusion_strategy': 'late',
        'batch_size': 64,
        'max_epochs': 80,
        'learning_rate': 2e-3
    },
    'signal_only': {
        'modalities': ['stokes', 'fluorescence'],
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.25,
        'fusion_strategy': 'early',
        'batch_size': 48,
        'max_epochs': 100,
        'learning_rate': 1e-3
    },
    'image_only': {
        'modalities': ['images'],
        'hidden_dims': [512, 256],
        'dropout_rate': 0.4,
        'fusion_strategy': 'late',
        'batch_size': 16,
        'max_epochs': 120,
        'learning_rate': 5e-4
    }
}


def get_preset_config(preset_name: str) -> ModelConfig:
    """获取预定义配置"""
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"未知的预设配置: {preset_name}. 可用配置: {available}")
    
    config_dict = PRESET_CONFIGS[preset_name].copy()
    return ModelConfig(**config_dict)


if __name__ == "__main__":
    # 测试配置
    config = ModelConfig()
    print(f"默认配置: {config}")
    
    # 测试预设配置
    simple_config = get_preset_config('simple')
    print(f"简单配置: {simple_config}")
    
    # 测试不同模态组合
    for preset in PRESET_CONFIGS.keys():
        preset_config = get_preset_config(preset)
        print(f"{preset}: {preset_config.modalities}")
