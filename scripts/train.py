#!/usr/bin/env python3
"""
多模态分类模型训练脚本
使用配置文件进行训练，支持灵活的模态选择
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from src.utils.config import ModelConfig
from src.training import MultimodalTrainer
from src.data import create_default_dataloader


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path: str) -> ModelConfig:
    """加载配置文件"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return ModelConfig.from_dict(config_dict)
    else:
        # 如果是Python文件，直接导入
        return ModelConfig.from_file(config_path)


def create_data_loaders(config: ModelConfig, data_path: str, full_config: dict = None, selected_classes: list = None):
    """创建数据加载器"""
    # 提取数据增强配置
    transform_config = full_config.get('data_augmentation', {}) if full_config else {}
    
    # 训练数据加载器
    train_loader = create_default_dataloader(
        hdf5_path=data_path,
        split='train',
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        balanced=True,
        use_cache=True,
        transform_config=transform_config,  # 传递数据增强配置
        selected_classes=selected_classes  # 传递类别过滤参数
    )
    
    # 验证数据加载器
    val_loader = create_default_dataloader(
        hdf5_path=data_path,
        split='val',
        batch_size=config.batch_size * 2,  # 验证时可以用更大的batch size
        num_workers=config.num_workers,
        balanced=False,
        use_cache=True,
        transform_config=transform_config,  # 验证时也传递配置（但只会用标准化）
        selected_classes=selected_classes  # 验证数据也应该过滤相同类别
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="多模态分类模型训练")
    
    # 基本参数
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='HDF5数据文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/runs',
        help='输出目录'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='multimodal_training',
        help='实验名称'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖配置文件）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小（覆盖配置文件）'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='学习率（覆盖配置文件）'
    )
    
    # 模态选择
    parser.add_argument(
        '--modalities',
        nargs='+',
        choices=['stokes', 'fluorescence', 'images'],
        default=None,
        help='使用的模态（覆盖配置文件）'
    )
    
    # 类别选择
    parser.add_argument(
        '--classes',
        nargs='+',
        default=None,
        help='选择特定类别进行训练（类别名称，如 CG IG PS3，或类别ID，如 0 1 2）'
    )
    
    # 其他选项
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='使用的GPU ID'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / 'training.log'
    setup_logging(args.log_level, str(log_file))
    logger = logging.getLogger(__name__)
    
    logger.info("开始多模态分类模型训练")
    logger.info(f"输出目录: {output_dir}")
    
    # 设置设备
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    
    # 先加载原始完整配置
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        with open(args.config, 'r', encoding='utf-8') as f:
            full_config_dict = yaml.safe_load(f)
    else:
        full_config_dict = {}
    
    # 然后创建ModelConfig对象
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.modalities is not None:
        config.modalities = args.modalities
    
    # 处理类别选择
    selected_classes = None
    if args.classes is not None:
        # 尝试转换为整数（类别ID）
        try:
            selected_classes = [int(cls) for cls in args.classes]
            logger.info(f"选择的类别ID: {selected_classes}")
        except ValueError:
            # 如果转换失败，则作为类别名称处理
            selected_classes = args.classes
            logger.info(f"选择的类别名称: {selected_classes}")
    
    logger.info(f"模型配置: {config}")
    
    # 创建数据加载器
    logger.info(f"加载数据: {args.data}")
    train_loader, val_loader = create_data_loaders(config, args.data, full_config_dict, selected_classes)  # 传递完整配置和类别选择
    logger.info(f"训练样本数: {len(train_loader.dataset)}")
    logger.info(f"验证样本数: {len(val_loader.dataset)}")
    
    # 如果选择了特定类别，需要更新模型配置中的类别数
    if selected_classes is not None:
        config.num_classes = train_loader.dataset.num_classes
        logger.info(f"更新类别数: {config.num_classes}")
    
    # 创建训练器
    trainer = MultimodalTrainer(config, device=device)
    
    # 设置训练环境
    trainer.setup_training(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=getattr(config, 'optimizer', 'adamw'),
        learning_rate=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 1e-4),
        scheduler_name=getattr(config, 'scheduler', 'cosine'),
        log_dir=str(output_dir / 'tensorboard')
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        logger.info(f"恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 打印模型摘要
    logger.info("模型结构:")
    logger.info(trainer.get_model_summary())
    
    # 保存配置
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
    logger.info(f"配置已保存: {config_save_path}")
    
    # 开始训练
    try:
        history = trainer.train(
            epochs=config.max_epochs,
            save_dir=str(output_dir / 'checkpoints'),
            save_every=getattr(config, 'save_every', 10),
            early_stopping_patience=getattr(config, 'early_stopping_patience', 20)
        )
        
        logger.info("训练完成！")
        logger.info(f"最终验证准确率: {max(history['val_accuracies']):.4f}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        # 保存当前状态
        interrupt_path = output_dir / 'checkpoints' / 'interrupted_model.pth'
        trainer.save_checkpoint(interrupt_path, trainer.current_epoch, 0.0)
        logger.info(f"中断状态已保存: {interrupt_path}")
    
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
