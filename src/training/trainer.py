"""
多模态模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import logging

class MultimodalTrainer:
    """多模态模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: Dict,
        device: torch.device,
        experiment_dir: Path,
        logger: logging.Logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger
        
        # 训练配置
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training'].get('weight_decay', 1e-4)
        self.patience = config['training'].get('patience', 10)
        self.min_delta = config['training'].get('min_delta', 1e-4)
        
        # 创建检查点目录
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 设置优化器和损失函数
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_scheduler()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.total_train_time = 0
        
    def _setup_optimizer(self):
        """设置优化器"""
        optimizer_type = self.config['training'].get('optimizer', 'adam')
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        self.logger.info(f"使用优化器: {optimizer_type}")
    
    def _setup_criterion(self):
        """设置损失函数"""
        criterion_type = self.config['training'].get('criterion', 'cross_entropy')
        
        if criterion_type == 'cross_entropy':
            # 检查是否使用类别权重
            use_class_weights = self.config['training'].get('use_class_weights', False)
            if use_class_weights:
                # 计算类别权重
                class_weights = self._calculate_class_weights()
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                self.logger.info("使用加权交叉熵损失")
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.logger.info("使用交叉熵损失")
        elif criterion_type == 'focal_loss':
            # 可以实现Focal Loss来处理类别不平衡
            self.criterion = self._create_focal_loss()
            self.logger.info("使用Focal Loss")
        else:
            raise ValueError(f"不支持的损失函数类型: {criterion_type}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_type = self.config['training'].get('scheduler', 'plateau')
        
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.patience // 2,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == 'step':
            step_size = self.config['training'].get('step_size', 30)
            gamma = self.config['training'].get('gamma', 0.1)
            self.scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            self.scheduler = None
        
        if self.scheduler:
            self.logger.info(f"使用学习率调度器: {scheduler_type}")
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """计算类别权重以处理类别不平衡"""
        # 统计训练集中各类别的样本数
        class_counts = torch.zeros(self.model.num_classes)
        
        for batch in self.train_loader:
            labels = batch['labels']
            for label in labels:
                class_counts[label] += 1
        
        # 计算权重（样本数的倒数）
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.model.num_classes * class_counts)
        
        # 归一化权重
        class_weights = class_weights / class_weights.sum() * self.model.num_classes
        
        return class_weights.to(self.device)
    
    def _create_focal_loss(self):
        """创建Focal Loss"""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 将数据移到设备
            stokes = batch['stokes'].to(self.device)
            fluorescence = batch['fluorescence'].to(self.device)
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(stokes, fluorescence, images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可选）
            max_grad_norm = self.config['training'].get('max_grad_norm', None)
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # 打印进度
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """验证一个epoch"""
        return self.evaluate(self.val_loader, split='val')
    
    def evaluate(self, dataloader, split='test') -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移到设备
                stokes = batch['stokes'].to(self.device)
                fluorescence = batch['fluorescence'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(stokes, fluorescence, images)
                loss = self.criterion(outputs, labels)
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        results = {
            f'{split}_loss': avg_loss,
            f'{split}_accuracy': accuracy,
            f'{split}_precision': precision,
            f'{split}_recall': recall,
            f'{split}_f1': f1
        }
        
        # 如果是验证，返回简化版本以便调度器使用
        if split == 'val':
            return avg_loss, accuracy
        
        return results
    
    def train(self):
        """完整的训练流程"""
        self.logger.info("开始训练...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印进度
            self.logger.info(
                f'Epoch {epoch+1}/{self.epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # 检查是否是最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.logger.info(f"新的最佳模型! 验证准确率: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                self.logger.info(f"验证准确率在 {self.patience} 个epoch内没有改善，早停")
                break
        
        self.total_train_time = time.time() - start_time
        self.logger.info(f"训练完成! 总时间: {self.total_train_time:.2f}s")
        
        # 保存训练历史和报告
        self.save_training_history()
        self.generate_training_report()
        self.plot_training_curves()
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        
        self.logger.info(f"从epoch {self.current_epoch} 恢复训练")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.experiment_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def generate_training_report(self):
        """生成训练报告"""
        report = {
            'experiment_info': {
                'total_epochs': len(self.train_history['train_loss']),
                'best_val_accuracy': float(self.best_val_acc),
                'best_val_loss': float(self.best_val_loss),
                'total_training_time': self.total_train_time,
                'final_learning_rate': self.optimizer.param_groups[0]['lr']
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'data_info': {
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'test_samples': len(self.test_loader.dataset) if self.test_loader else 0,
                'batch_size': self.train_loader.batch_size
            }
        }
        
        report_path = self.experiment_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(epochs, self.train_history['learning_rates'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 验证损失和准确率的关系
        axes[1, 1].scatter(self.train_history['val_loss'], self.train_history['val_acc'])
        axes[1, 1].set_title('Val Loss vs Val Accuracy')
        axes[1, 1].set_xlabel('Val Loss')
        axes[1, 1].set_ylabel('Val Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_confusion_matrix(self, dataloader, split='test'):
        """生成混淆矩阵"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                stokes = batch['stokes'].to(self.device)
                fluorescence = batch['fluorescence'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(stokes, fluorescence, images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {split.title()} Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(self.experiment_dir / f'confusion_matrix_{split}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def cleanup(self):
        """清理资源"""
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
