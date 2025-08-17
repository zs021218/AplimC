#!/usr/bin/env python3
"""
多模态分类模型训练脚本
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory
from src.training.trainer import MultimodalTrainer
from src.utils.utils import setup_logging, save_config, load_config

def set_seed(seed=42):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练多模态分类模型')
    
    # 基础参数
    parser.add_argument('--config', type=str, default='configs/full_multimodal_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='数据路径')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='输出目录')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='实验名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批量大小')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='训练设备')
    
    # 模型参数
    parser.add_argument('--model-type', type=str, default=None,
                       choices=['multimodal', 'stokes_only', 'fluorescence_only', 'images_only'],
                       help='模型类型')
    parser.add_argument('--fusion-method', type=str, default=None,
                       choices=['concat', 'attention', 'cross_attention'],
                       help='融合方法')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--no-validate', action='store_true',
                       help='不进行验证')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    return parser.parse_args()

def get_device(device_arg='auto'):
    """获取训练设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cuda':
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("使用CPU")
    
    return device

def create_experiment_name(config, args):
    """创建实验名称"""
    if args.experiment_name:
        return args.experiment_name
    
    # 基于配置自动生成实验名称
    model_type = config['model'].get('type', 'multimodal')
    fusion_method = config['model'].get('fusion_method', 'concat')
    lr = config['training'].get('learning_rate', 1e-3)
    batch_size = config['training'].get('batch_size', 32)
    timestamp = datetime.now().strftime('%M%S%f')[:-3]  # 分钟秒毫秒
    
    return f"{model_type}_{fusion_method}_lr{lr}_bs{batch_size}_{timestamp}"

def override_config_with_args(config, args):
    """用命令行参数覆盖配置"""
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.model_type is not None:
        config['model']['type'] = args.model_type
    if args.fusion_method is not None:
        config['model']['fusion_method'] = args.fusion_method
    
    return config

def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 用命令行参数覆盖配置
    config = override_config_with_args(config, args)
    
    # 获取设备
    device = get_device(args.device)
    
    # 创建实验目录
    experiment_name = create_experiment_name(config, args)
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(experiment_dir / 'training.log')
    logger.info(f"开始实验: {experiment_name}")
    logger.info(f"实验目录: {experiment_dir}")
    logger.info(f"使用设备: {device}")
    
    # 保存配置
    save_config(config, experiment_dir / 'config.yaml')
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=args.data_path,
            batch_size=config['training']['batch_size'],
            num_workers=args.num_workers,
            use_weighted_sampling=config['training'].get('use_weighted_sampling', False)
        )
        
        logger.info(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次")
        logger.info(f"验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次")
        logger.info(f"测试集: {len(test_loader.dataset)} 样本, {len(test_loader)} 批次")
        
        # 创建模型
        print("创建模型...")
        model_factory = ModelFactory()
        model = model_factory.create_model(config['model'])
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        # 创建训练器
        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
            logger=logger
        )
        
        # 恢复训练（如果指定）
        if args.resume:
            print(f"从检查点恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        print("开始训练...")
        trainer.train()
        
        # 在测试集上评估
        if not args.no_validate and test_loader is not None:
            print("在测试集上评估...")
            test_results = trainer.evaluate(test_loader, split='test')
            logger.info(f"测试集结果: {test_results}")
            
            # 保存测试结果
            import json
            with open(experiment_dir / 'test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2)
        
        print(f"训练完成! 结果保存在: {experiment_dir}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        logger.info("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        logger.error(f"训练错误: {e}", exc_info=True)
        raise
    finally:
        # 清理资源
        if 'trainer' in locals():
            trainer.cleanup()

if __name__ == '__main__':
    main()
