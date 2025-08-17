"""模型配置文件"""

from typing import Dict, List, Any

class ModelConfig:
    """模型配置类"""
    
    # 基础配置
    NUM_CLASSES = 12
    
    # 数据配置
    SIGNAL_LENGTH = 4000
    IMAGE_SIZE = (224, 224)
    NUM_VIEWS = 3
    
    # Stokes 编码器配置
    STOKES_CONFIG = {
        'input_channels': 4,
        'signal_length': SIGNAL_LENGTH,
        'hidden_dims': [64, 128, 256]
    }
    
    # 荧光编码器配置
    FLUORESCENCE_CONFIG = {
        'input_channels': 16,
        'signal_length': SIGNAL_LENGTH,
        'hidden_dims': [64, 128, 256]
    }
    
    # 图像编码器配置
    IMAGE_CONFIG = {
        'num_views': NUM_VIEWS,
        'image_size': IMAGE_SIZE,
        'hidden_dims': [64, 128, 256, 512]
    }
    
    # 融合方法选项
    FUSION_METHODS = ['concat', 'attention', 'weighted_sum']
    
    # 预定义模型配置
    MODEL_CONFIGS = {
        'multimodal_all': {
            'modalities': ['stokes', 'fluorescence', 'images'],
            'fusion_method': 'concat',
            'description': '使用所有三种模态的完整多模态模型'
        },
        'multimodal_signal': {
            'modalities': ['stokes', 'fluorescence'],
            'fusion_method': 'concat',
            'description': '仅使用信号数据的多模态模型'
        },
        'stokes_only': {
            'modalities': ['stokes'],
            'fusion_method': 'concat',
            'description': '仅使用Stokes参数的单模态模型'
        },
        'fluorescence_only': {
            'modalities': ['fluorescence'],
            'fusion_method': 'concat',
            'description': '仅使用荧光数据的单模态模型'
        },
        'images_only': {
            'modalities': ['images'],
            'fusion_method': 'concat',
            'description': '仅使用图像数据的单模态模型'
        },
        'multimodal_attention': {
            'modalities': ['stokes', 'fluorescence', 'images'],
            'fusion_method': 'attention',
            'description': '使用注意力机制融合的多模态模型'
        },
        'multimodal_weighted': {
            'modalities': ['stokes', 'fluorescence', 'images'],
            'fusion_method': 'weighted_sum',
            'description': '使用加权求和融合的多模态模型'
        }
    }
    
    @classmethod
    def get_model_config(cls, config_name: str) -> Dict[str, Any]:
        """获取预定义的模型配置"""
        if config_name not in cls.MODEL_CONFIGS:
            available_configs = list(cls.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
        
        config = cls.MODEL_CONFIGS[config_name].copy()
        
        # 添加编码器配置
        config.update({
            'num_classes': cls.NUM_CLASSES,
            'stokes_config': cls.STOKES_CONFIG,
            'fluorescence_config': cls.FLUORESCENCE_CONFIG,
            'image_config': cls.IMAGE_CONFIG
        })
        
        return config
    
    @classmethod
    def list_available_configs(cls) -> List[str]:
        """列出所有可用的配置"""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def print_config_descriptions(cls):
        """打印所有配置的描述"""
        print("Available Model Configurations:")
        print("=" * 50)
        for name, config in cls.MODEL_CONFIGS.items():
            print(f"{name}:")
            print(f"  模态: {config['modalities']}")
            print(f"  融合方法: {config['fusion_method']}")
            print(f"  描述: {config['description']}")
            print()

# 训练配置
class TrainingConfig:
    """训练配置类"""
    
    # 基础训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    
    # 学习率调度
    LR_SCHEDULER = {
        'type': 'StepLR',
        'step_size': 30,
        'gamma': 0.1
    }
    
    # 早停参数
    EARLY_STOPPING = {
        'patience': 15,
        'min_delta': 0.001
    }
    
    # 优化器配置
    OPTIMIZER_CONFIG = {
        'type': 'Adam',
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.999)
    }
    
    # 损失函数
    LOSS_CONFIG = {
        'type': 'CrossEntropyLoss',
        'label_smoothing': 0.1
    }
    
    # 数据加载器配置
    DATALOADER_CONFIG = {
        'batch_size': BATCH_SIZE,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': True
    }
    
    # 验证配置
    VALIDATION_CONFIG = {
        'val_interval': 1,  # 每多少个epoch验证一次
        'save_best_only': True,
        'save_last': True
    }

# 数据配置
class DataConfig:
    """数据配置类"""
    
    # 数据路径
    DATA_PATH = '/data3/zs/AplimC/data/processed'
    
    # 数据增强配置
    AUGMENTATION_CONFIG = {
        'image_augmentation': {
            'random_rotation': 10,
            'random_horizontal_flip': 0.5,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }
        },
        'signal_augmentation': {
            'noise_level': 0.01,
            'enable_noise': True
        }
    }
    
    # 数据归一化
    NORMALIZATION_CONFIG = {
        'image_normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'signal_normalization': {
            'method': 'z_score',  # 'z_score', 'min_max', 'none'
            'per_channel': True
        }
    }

if __name__ == "__main__":
    # 打印所有可用配置
    ModelConfig.print_config_descriptions()
    
    # 测试获取配置
    config = ModelConfig.get_model_config('multimodal_all')
    print("Multimodal All Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
