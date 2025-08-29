#!/usr/bin/env python3
"""
知识蒸馏训练脚本
目标: 使信号分类从68%提升到90%+
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
import argparse
import yaml
import time
from typing import Dict, Any

from src.models.knowledge_distillation import (
    FeatureMimicryDistillation, 
    DistillationLoss,
    load_pretrained_teacher,
    RelationKnowledgeExtractor,
    AdaptiveAttentionTransfer
)
from src.utils.config import ModelConfig
from src.data.dataset import MultimodalHDF5Dataset

def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'distillation_training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_distillation_model(config: Dict[str, Any], teacher_model_path: str, device: str):
    """创建知识蒸馏模型"""
    
    # 首先加载预训练教师模型来确定其特征维度
    teacher_checkpoint = torch.load(teacher_model_path, map_location=device, weights_only=False)
    
    # 教师模型配置 (图像模态) - 使用与原始模型相同的输出维度
    teacher_config = ModelConfig(
        modalities=['images'],
        num_classes=config['model']['num_classes'],
        num_views=config['model']['num_views'],
        use_pretrained=True,
        backbone=config['model']['backbone'],
        freeze_layers=0
    )
    
    # 创建临时教师模型来确定特征维度
    from src.models.classifier import MultimodalClassifier
    temp_teacher = MultimodalClassifier(teacher_config)
    temp_teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    
    # 确定教师模型的图像编码器实际输出维度
    # 通过检查图像编码器的结构来确定输出维度
    sample_input = torch.randn(1, 3, 224, 224, 3)  # 批次大小1的样本输入
    with torch.no_grad():
        temp_teacher.eval()
        # 创建虚拟图像数据
        dummy_batch = {'images': sample_input}
        encoder_output = temp_teacher.encoders.images(sample_input)
        actual_encoder_dim = encoder_output.shape[1]  # 实际的编码器输出维度
    
    logger = logging.getLogger(__name__)
    logger.info(f"检测到图像编码器输出维度: {actual_encoder_dim}")
    
    # 确定教师模型的特征维度
    # 找到最后的线性层来确定输入特征维度
    classifier_modules = list(temp_teacher.classifier.children())
    final_linear = None
    
    for module in reversed(classifier_modules):
        if isinstance(module, nn.Linear):
            final_linear = module
            break
    
    if final_linear is None:
        raise ValueError("无法找到教师模型的最终线性层")
    
    teacher_feature_dim = final_linear.in_features
    logger.info(f"检测到教师模型分类器输入维度: {teacher_feature_dim}")
    
    # 学生模型配置 (信号模态)
    student_config = ModelConfig(
        modalities=config['model']['student_modalities'],  # ['stokes', 'fluorescence']
        num_classes=config['model']['num_classes'],
        stokes_dim=config['model']['stokes_dim'],
        stokes_length=config['model']['stokes_length'],
        fluorescence_dim=config['model']['fluorescence_dim'],
        fluorescence_length=config['model']['fluorescence_length']
    )
    
    # 蒸馏配置
    distillation_config = config['distillation']
    
    # 动态修改蒸馏配置以匹配教师模型
    distillation_config['teacher_feature_dim'] = teacher_feature_dim
    
    # 创建蒸馏模型
    distill_model = FeatureMimicryDistillation(
        teacher_config=teacher_config,
        student_config=student_config,
        distillation_config=distillation_config,
        num_classes=config['model']['num_classes']
    )
    
    # 检查维度匹配性
    if actual_encoder_dim != teacher_feature_dim:
        logger.warning(f"图像编码器输出维度({actual_encoder_dim})与分类器输入维度({teacher_feature_dim})不匹配")
        logger.info("将使用完整的原始教师模型结构")
        
        # 直接使用原始教师模型，但只保留图像部分
        # 创建一个包装类来适配蒸馏模型的接口
        class TeacherWrapper(nn.Module):
            def __init__(self, full_teacher):
                super().__init__()
                self.image_encoder = full_teacher.encoders.images
                self.classifier = full_teacher.classifier
                
            def forward(self, x):
                # x 应该是图像数据
                features = self.image_encoder(x)
                logits = self.classifier(features)
                return features, logits
        
        # 重建教师网络为包装后的完整模型
        teacher_wrapper = TeacherWrapper(temp_teacher)
        distill_model.teacher = teacher_wrapper.to(device)
        
        # 使用图像编码器的输出维度作为特征维度
        final_feature_dim = actual_encoder_dim
        
    else:
        logger.info("维度匹配，使用重建的教师网络")
        # 手动重建教师网络以匹配预训练模型的维度
        teacher_image_encoder = temp_teacher.encoders.images
        teacher_classifier = nn.Linear(teacher_feature_dim, config['model']['num_classes'])
        
        # 复制分类器权重
        teacher_classifier.weight.data = final_linear.weight.data.clone()
        teacher_classifier.bias.data = final_linear.bias.data.clone()
        
        # 重建教师网络
        distill_model.teacher = nn.Sequential(
            teacher_image_encoder,
            teacher_classifier
        ).to(device)
        
        # 使用图像编码器的输出维度作为特征维度
        final_feature_dim = actual_encoder_dim
    
    # 重建特征对齐器以匹配实际的教师特征维度
    aligners = nn.ModuleDict()
    for layer in ['shallow', 'middle', 'deep']:
        aligners[f'{layer}_aligner'] = nn.Sequential(
            nn.Linear(128, final_feature_dim),  # 学生256 -> 教师final_feature_dim
            nn.ReLU(inplace=True),
            nn.Linear(final_feature_dim, final_feature_dim),
            nn.LayerNorm(final_feature_dim)
        ).to(device)
    distill_model.feature_aligners = aligners
    
    # 重建关系知识提取器
    distill_model.relation_extractor = RelationKnowledgeExtractor(
        teacher_dim=final_feature_dim,
        student_dim=128,
        hidden_dim=128
    ).to(device)
    
    # 重建注意力转移
    distill_model.attention_transfer = AdaptiveAttentionTransfer(
        teacher_dim=final_feature_dim,
        student_dim=128
    ).to(device)
    
    # 冻结教师网络
    for param in distill_model.teacher.parameters():
        param.requires_grad = False
    
    logger.info("教师模型权重加载成功")
    
    return distill_model.to(device)

def train_distillation_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_student_loss = 0.0
    total_distill_loss = 0.0
    total_feature_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 数据移到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        labels = batch['labels']
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch, mode='train')
        
        # 计算损失
        loss_dict = criterion(outputs, labels)
        total_loss_batch = loss_dict['total_loss']
        
        # 反向传播
        total_loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        total_student_loss += loss_dict['student_loss'].item()
        total_distill_loss += loss_dict['distill_loss'].item()
        total_feature_loss += loss_dict['feature_loss'].item()
        
        # 计算准确率 (基于学生网络的输出)
        predictions = outputs['student_logits'].argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        if batch_idx % 50 == 0:
            logger = logging.getLogger(__name__)
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {total_loss_batch.item():.4f}, '
                       f'Student Acc: {100. * correct / total_samples:.2f}%')
    
    return {
        'loss': total_loss / len(train_loader),
        'student_loss': total_student_loss / len(train_loader),
        'distill_loss': total_distill_loss / len(train_loader),
        'feature_loss': total_feature_loss / len(train_loader),
        'accuracy': 100. * correct / total_samples
    }

def validate_student(
    model: nn.Module,
    val_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """验证学生网络性能"""
    model.eval()
    
    correct = 0
    total_samples = 0
    val_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            # 数据移到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            labels = batch['labels']
            
            # 只使用学生网络进行推理
            student_features, student_logits = model.student(batch, return_features=True)
            
            loss = criterion(student_logits, labels)
            val_loss += loss.item()
            
            # 统计准确率
            predictions = student_logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    return {
        'loss': val_loss / len(val_loader),
        'accuracy': 100. * correct / total_samples
    }

def main():
    parser = argparse.ArgumentParser(description='知识蒸馏训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--teacher', type=str, required=True, help='预训练教师模型路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    parser.add_argument('--name', type=str, default='distillation', help='实验名称')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            args.device = 'cpu'
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置实验目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/distillation/{args.name}_{timestamp}")
    setup_logging(exp_dir)
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始知识蒸馏训练: {args.name}")
    logger.info(f"使用设备: {args.device}")
    logger.info(f"教师模型: {args.teacher}")
    
    # 创建数据集
    train_dataset = MultimodalHDF5Dataset(
        args.data, 'train', 
        load_modalities=['stokes', 'fluorescence', 'images']  # 需要所有模态
    )
    val_dataset = MultimodalHDF5Dataset(
        args.data, 'val',
        load_modalities=['stokes', 'fluorescence', 'images']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(val_dataset)}")
    
    # 创建蒸馏模型
    model = create_distillation_model(config, args.teacher, args.device)
    
    # 模型分析
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数和优化器
    criterion = DistillationLoss(
        alpha=config['distillation']['alpha'],
        beta=config['distillation']['beta'],
        gamma=config['distillation']['gamma'],
        delta=config['distillation']['delta'],
        temperature=config['distillation']['temperature']
    )
    
    # 只优化学生网络参数
    student_params = list(model.student.parameters()) + \
                    list(model.feature_aligners.parameters()) + \
                    list(model.relation_extractor.parameters()) + \
                    list(model.attention_transfer.parameters())
    
    optimizer = optim.AdamW(
        student_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    patience = config['training'].get('early_stopping_patience', 15)
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # 训练
        train_metrics = train_distillation_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch+1
        )
        
        # 验证
        val_metrics = validate_student(model, val_loader, args.device)
        
        # 学习率调度
        scheduler.step()
        
        # 记录日志
        logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                   f"Student Loss: {train_metrics['student_loss']:.4f}, "
                   f"Distill Loss: {train_metrics['distill_loss']:.4f}, "
                   f"Feature Loss: {train_metrics['feature_loss']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.2f}%")
        
        logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                   f"Accuracy: {val_metrics['accuracy']:.2f}%")
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # 保存完整蒸馏模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'student_state_dict': model.student.state_dict(),  # 单独保存学生网络
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, exp_dir / 'best_distillation_model.pth')
            
            logger.info(f"保存最佳模型，验证准确率: {val_metrics['accuracy']:.2f}%")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"验证准确率连续 {patience} 轮未提升，早停")
            break
    
    logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存纯学生模型用于推理
    torch.save({
        'model_state_dict': model.student.state_dict(),
        'config': config['model']['student_modalities'],
        'best_val_acc': best_val_acc
    }, exp_dir / 'student_model_only.pth')
    
    logger.info("学生模型已单独保存用于推理")
    
    return best_val_acc

if __name__ == '__main__':
    main()
