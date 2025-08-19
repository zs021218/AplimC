#!/usr/bin/env python3
"""
多模态分类模型训练器
遵循奥卡姆剃刀原理，保持简洁高效
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..models import MultimodalClassifier
from ..utils.config import ModelConfig
from ..utils.utils import count_parameters, analyze_model
from ..data import create_default_dataloader

logger = logging.getLogger(__name__)


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    带Warmup的余弦退火调度器
    """
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=0, warmup_start_lr=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class MultimodalTrainer:
    """多模态分类模型训练器"""
    
    def __init__(
        self,
        config: ModelConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化训练器
        
        Args:
            config: 模型配置
            model: 预训练模型（可选）
            device: 计算设备
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        if model is None:
            self.model = MultimodalClassifier(config)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # TensorBoard
        self.writer = None
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
        logger.info(f"模型参数量: {count_parameters(self.model):,}")
    
    def setup_training(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer_name: str = 'adamw',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_name: str = 'cosine',
        log_dir: Optional[str] = None
    ):
        """
        设置训练环境
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer_name: 优化器名称
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_name: 学习率调度器名称
            log_dir: 日志目录
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设置损失函数
        if hasattr(self.config, 'loss_weights') and self.config.loss_weights:
            weights = torch.tensor(self.config.loss_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(
                weight=weights.to(self.device),
                label_smoothing=getattr(self.config, 'label_smoothing', 0.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=getattr(self.config, 'label_smoothing', 0.0)
            )
        
        # 设置优化器
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)
        
        # 设置学习率调度器
        self.scheduler = self._create_scheduler(scheduler_name)
        
        # 设置TensorBoard
        if log_dir:
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard日志将保存到: {log_dir}")
    
    def _create_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float
    ) -> optim.Optimizer:
        """创建优化器"""
        if optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _create_scheduler(self, scheduler_name: str) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs
            )
        elif scheduler_name.lower() == 'cosine_with_warmup':
            # 获取调度器参数
            lr_params = getattr(self.config, 'lr_scheduler_params', {})
            T_max = lr_params.get('T_max', self.config.max_epochs)
            eta_min = lr_params.get('eta_min', 0)
            warmup_epochs = lr_params.get('warmup_epochs', 0)
            warmup_start_lr = lr_params.get('warmup_start_lr', 0)
            
            return CosineAnnealingWarmupRestarts(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr
            )
        elif scheduler_name.lower() == 'step':
            # 获取step调度器参数
            lr_params = getattr(self.config, 'lr_scheduler_params', {})
            step_size = lr_params.get('step_size', 30)
            gamma = lr_params.get('gamma', 0.1)
            
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=10,
                factor=0.5
            )
        elif scheduler_name.lower() == 'none':
            return None
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if hasattr(self.config, 'gradient_clip_val') and self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            accuracy = correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy:.4f}"
            })
            
            # TensorBoard记录
            if self.writer and batch_idx % 100 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', accuracy, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 数据移动到设备
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(
        self,
        epochs: int,
        save_dir: str,
        save_every: int = 10,
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            save_dir: 模型保存目录
            save_every: 每隔多少轮保存一次
            early_stopping_patience: 早停耐心值
        
        Returns:
            训练历史
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始训练，共 {epochs} 轮")
        logger.info(f"模型将保存到: {save_dir}")
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                
                # 记录到TensorBoard
                if self.writer:
                    self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # 检查是否是最佳模型
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    
                    # 保存最佳模型
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_metrics['accuracy'])
                    logger.info(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                
                # 早停检查
                if patience_counter >= early_stopping_patience:
                    logger.info(f"验证准确率连续 {early_stopping_patience} 轮未提升，早停")
                    break
                
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_metrics.get('accuracy', 0.0))
        
        # 保存最终模型
        final_path = save_dir / 'final_model.pth'
        self.save_checkpoint(final_path, epochs - 1, val_metrics.get('accuracy', 0.0))
        
        total_time = time.time() - start_time
        logger.info(f"训练完成，总用时: {total_time / 3600:.2f} 小时")
        logger.info(f"最佳验证准确率: {best_val_acc:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, path: Union[str, Path], epoch: int, accuracy: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.debug(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> Dict:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        logger.info(f"检查点已加载: {path}")
        logger.info(f"恢复到第 {self.current_epoch + 1} 轮，准确率: {checkpoint.get('accuracy', 0.0):.4f}")
        
        return checkpoint
    
    def get_model_summary(self) -> str:
        """获取模型摘要"""
        analysis = analyze_model(self.model)
        return f"模型分析: {analysis}"


if __name__ == "__main__":
    # 简单测试
    from ..utils.config import ModelConfig
    
    config = ModelConfig()
    trainer = MultimodalTrainer(config)
    print(trainer.get_model_summary())
