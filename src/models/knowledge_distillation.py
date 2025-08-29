#!/usr/bin/env python3
"""
基于特征模仿的知识蒸馏框架
目标: 使用图像模态(95%准确率)指导信号模态学习，提升信号分类到90%+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .classifier import SignalEncoder, ImageEncoder

class FeatureMimicryDistillation(nn.Module):
    """特征模仿知识蒸馏网络"""
    
    def __init__(
        self,
        teacher_config,
        student_config, 
        distillation_config,
        num_classes: int = 12
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.distill_config = distillation_config
        
        # 教师网络 (图像分类器) - 冻结参数
        self.teacher = self._build_teacher(teacher_config)
        self._freeze_teacher()
        
        # 学生网络 (信号分类器) - 增强版
        self.student = self._build_enhanced_student(student_config)
        
        # 特征对齐网络
        self.feature_aligners = self._build_aligners()
        
        # 关系知识提取器
        self.relation_extractor = RelationKnowledgeExtractor(
            teacher_dim=256,
            student_dim=128,  # deep层输出维度
            hidden_dim=128
        )
        
        # 自适应注意力转移
        self.attention_transfer = AdaptiveAttentionTransfer(
            teacher_dim=256,
            student_dim=128
        )
        
    def _build_teacher(self, config):
        """构建教师网络 (图像分类器)"""
        teacher = nn.Sequential(
            ImageEncoder(
                num_views=config.num_views,
                output_dim=256,
                pretrained=config.use_pretrained,
                backbone=config.backbone
            ),
            nn.Linear(256, self.num_classes)
        )
        return teacher
    
    def _build_enhanced_student(self, config):
        """构建增强版学生网络 (信号分类器)"""
        return EnhancedSignalClassifier(
            stokes_dim=config.stokes_dim,
            stokes_length=config.stokes_length,
            fluorescence_dim=config.fluorescence_dim,
            fluorescence_length=config.fluorescence_length,
            modalities=config.modalities,
            num_classes=self.num_classes,
            hidden_dims=[512, 256, 128]  # 增强容量
        )
    
    def _build_aligners(self):
        """构建特征对齐网络"""
        aligners = nn.ModuleDict()
        
        # 多层特征对齐
        for layer in ['shallow', 'middle', 'deep']:
            aligners[f'{layer}_aligner'] = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.LayerNorm(256)
            )
        
        return aligners
    
    def _freeze_teacher(self):
        """冻结教师网络参数"""
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def forward(self, batch: Dict[str, torch.Tensor], mode: str = 'train'):
        """前向传播
        
        Args:
            batch: 包含 'images' 和信号数据的批次
            mode: 'train' 或 'inference'
        """
        results = {}
        
        # 1. 教师网络推理 (图像 -> 特征 + 预测)
        with torch.no_grad():
            teacher_features, teacher_logits = self._teacher_forward(batch['images'])
        
        # 2. 学生网络推理 (信号 -> 特征 + 预测)
        student_features, student_logits = self._student_forward(batch)
        
        # 3. 特征对齐
        aligned_features = self._align_features(student_features, teacher_features)
        
        # 4. 关系知识提取
        # 如果student_features是dict，取deep层特征，否则直接用
        if isinstance(student_features, dict):
            student_feat_for_relation = student_features.get('deep', list(student_features.values())[-1])
        else:
            student_feat_for_relation = student_features
        relation_loss = self.relation_extractor(
            teacher_features, student_feat_for_relation, batch.get('labels')
        )
        
        # 5. 注意力转移
        attention_loss = self.attention_transfer(teacher_features, aligned_features['deep'])
        
        results.update({
            'student_logits': student_logits,
            'teacher_logits': teacher_logits,
            'student_features': student_features,
            'teacher_features': teacher_features,
            'aligned_features': aligned_features,
            'relation_loss': relation_loss,
            'attention_loss': attention_loss
        })
        
        return results
    
    def _teacher_forward(self, images):
        """教师网络前向传播"""
        # 检查教师网络的类型
        if hasattr(self.teacher, 'forward') and hasattr(self.teacher, 'image_encoder'):
            # 使用包装后的完整教师模型
            img_features, teacher_logits = self.teacher(images)
        else:
            # 使用Sequential结构
            img_features = self.teacher[0](images)  # ImageEncoder输出
            teacher_logits = self.teacher[1](img_features)  # 分类层
        
        return img_features, teacher_logits
    
    def _student_forward(self, batch):
        """学生网络前向传播"""
        return self.student(batch, return_features=True)
    
    def _align_features(self, student_features, teacher_features):
        """特征对齐"""
        # 这里假设student_features是dict，teacher_features是tensor
        aligned = {}
        
        if isinstance(student_features, dict):
            # 如果学生网络有多个特征层
            for name, feat in student_features.items():
                if name in self.feature_aligners:
                    aligned[name] = self.feature_aligners[name](feat)
                else:
                    aligned[name] = feat
        else:
            # 单一特征对齐
            aligned = self.feature_aligners['deep_aligner'](student_features)
        
        return aligned


class EnhancedSignalClassifier(nn.Module):
    """增强版信号分类器 - 学生网络"""
    
    def __init__(
        self,
        stokes_dim: int,
        stokes_length: int,
        fluorescence_dim: int, 
        fluorescence_length: int,
        modalities: List[str],
        num_classes: int,
        hidden_dims: List[int] = [512, 256, 128]
    ):
        super().__init__()
        
        self.modalities = modalities
        
        # 增强的信号编码器
        self.encoders = nn.ModuleDict()
        
        if 'stokes' in modalities:
            self.encoders['stokes'] = EnhancedSignalEncoder(
                input_channels=stokes_dim,
                input_length=stokes_length,
                output_dim=256
            )
        
        if 'fluorescence' in modalities:
            self.encoders['fluorescence'] = EnhancedSignalEncoder(
                input_channels=fluorescence_dim,
                input_length=fluorescence_length,
                output_dim=256
            )
        
        # 多层特征提取器 (为知识蒸馏提供中间特征)
        self.feature_extractors = nn.ModuleDict({
            'shallow': nn.Sequential(
                nn.Linear(256 * len(modalities), 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ),
            'middle': nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ),
            'deep': nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            )
        })
        
        # 分类头
        self.classifier = nn.Linear(128, num_classes)
        
        # 特征融合注意力
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False):
        """前向传播"""
        # 1. 编码各模态信号
        modal_features = []
        for modality in self.modalities:
            if modality in batch:
                feat = self.encoders[modality](batch[modality])
                modal_features.append(feat)
        
        # 2. 特征融合
        if len(modal_features) > 1:
            # 使用注意力融合多模态特征
            stacked_features = torch.stack(modal_features, dim=1)  # (B, num_modalities, 256)
            fused_features, _ = self.feature_attention(
                stacked_features, stacked_features, stacked_features
            )
            fused_features = fused_features.mean(dim=1)  # (B, 256)
        else:
            fused_features = modal_features[0]
        
        # 复制用于多层特征提取
        concatenated = torch.cat(modal_features, dim=1)  # (B, 256*num_modalities)
        
        # 3. 多层特征提取
        features = {}
        x = concatenated
        
        for layer_name, layer in self.feature_extractors.items():
            x = layer(x)
            features[layer_name] = x
        
        # 4. 分类
        logits = self.classifier(x)
        
        if return_features:
            return features, logits
        else:
            return logits


class EnhancedSignalEncoder(nn.Module):
    """增强版信号编码器 - 更强的特征提取能力"""
    
    def __init__(self, input_channels: int, input_length: int, output_dim: int = 256):
        super().__init__()
        
        # 多尺度卷积分支 - 确保输出长度一致
        self.multi_scale_convs = nn.ModuleList([
            # 分支1: 细粒度特征
            nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(250)  # 统一输出长度
            ),
            # 分支2: 中等尺度特征  
            nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(250)  # 统一输出长度
            ),
            # 分支3: 粗粒度特征
            nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=15, stride=4, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=9, stride=4, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(250)  # 统一输出长度
            )
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=5, stride=2, padding=2),  # 3*64=192
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # 多尺度特征提取
        multi_scale_features = []
        for conv_branch in self.multi_scale_convs:
            branch_out = conv_branch(x)
            multi_scale_features.append(branch_out)
        
        # 拼接多尺度特征
        concatenated = torch.cat(multi_scale_features, dim=1)
        
        # 特征融合
        fused = self.feature_fusion(concatenated)
        fused = fused.squeeze(-1)  # 移除时间维度
        
        # 输出
        output = self.output_layer(fused)
        
        return output


class RelationKnowledgeExtractor(nn.Module):
    """关系知识提取器 - 学习样本间的关系知识"""
    
    def __init__(self, teacher_dim: int, student_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.teacher_proj = nn.Linear(teacher_dim, hidden_dim)
        self.student_proj = nn.Linear(student_dim, hidden_dim)
        
        # 关系建模网络
        self.relation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, teacher_features, student_features, labels=None):
        """计算关系知识蒸馏损失"""
        batch_size = teacher_features.size(0)
        
        # 特征投影
        teacher_proj = self.teacher_proj(teacher_features)  # (B, H)
        student_proj = self.student_proj(student_features)  # (B, H)
        
        # 计算样本间关系
        teacher_relations = self._compute_relations(teacher_proj)  # (B, B)
        student_relations = self._compute_relations(student_proj)  # (B, B)
        
        # 关系知识蒸馏损失
        relation_loss = F.mse_loss(student_relations, teacher_relations)
        
        return relation_loss
    
    def _compute_relations(self, features):
        """计算特征间的关系矩阵"""
        batch_size = features.size(0)
        relations = torch.zeros(batch_size, batch_size).to(features.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    combined = torch.cat([features[i], features[j]], dim=0)
                    relation = self.relation_net(combined.unsqueeze(0))
                    relations[i, j] = relation.squeeze()
        
        return relations


class AdaptiveAttentionTransfer(nn.Module):
    """自适应注意力转移模块"""
    
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        
        self.teacher_attention = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(teacher_dim // 2, teacher_dim),
            nn.Softmax(dim=1)
        )
        
        self.student_attention = nn.Sequential(
            nn.Linear(student_dim, student_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(student_dim // 2, student_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, teacher_features, student_features):
        """计算注意力转移损失"""
        # 计算注意力权重
        teacher_attn = self.teacher_attention(teacher_features)
        student_attn = self.student_attention(student_features)
        
        # 注意力转移损失
        attention_loss = F.kl_div(
            F.log_softmax(student_attn, dim=1),
            F.softmax(teacher_attn, dim=1),
            reduction='batchmean'
        )
        
        return attention_loss


class DistillationLoss(nn.Module):
    """知识蒸馏综合损失函数"""
    
    def __init__(
        self,
        alpha: float = 0.3,    # 学生损失权重
        beta: float = 0.4,     # 知识蒸馏权重
        gamma: float = 0.2,    # 特征模仿权重
        delta: float = 0.1,    # 关系知识权重
        temperature: float = 4.0
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, labels):
        """计算综合蒸馏损失"""
        student_logits = outputs['student_logits']
        teacher_logits = outputs['teacher_logits']
        student_features = outputs['aligned_features']
        teacher_features = outputs['teacher_features']
        
        # 1. 学生网络分类损失
        student_loss = self.ce_loss(student_logits, labels)
        
        # 2. 知识蒸馏损失 (软标签)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distill_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 3. 特征模仿损失
        if isinstance(student_features, dict):
            feature_loss = 0
            for key in student_features:
                if key in ['deep', 'middle', 'shallow']:  # 选择关键层
                    feature_loss += self.mse_loss(student_features[key], teacher_features)
            feature_loss /= len(['deep', 'middle', 'shallow'])
        else:
            feature_loss = self.mse_loss(student_features, teacher_features)
        
        # 4. 关系知识损失
        relation_loss = outputs.get('relation_loss', torch.tensor(0.0))
        attention_loss = outputs.get('attention_loss', torch.tensor(0.0))
        
        # 综合损失
        total_loss = (
            self.alpha * student_loss +
            self.beta * distill_loss +
            self.gamma * feature_loss +
            self.delta * (relation_loss + attention_loss)
        )
        
        return {
            'total_loss': total_loss,
            'student_loss': student_loss,
            'distill_loss': distill_loss,
            'feature_loss': feature_loss,
            'relation_loss': relation_loss,
            'attention_loss': attention_loss
        }


def load_pretrained_teacher(teacher_path: str, config) -> nn.Module:
    """加载预训练的教师模型"""
    from .classifier import MultimodalClassifier
    
    # 创建图像单模态配置
    teacher_config = config.copy()
    teacher_config.modalities = ['images']
    
    # 加载预训练模型
    teacher_model = MultimodalClassifier(teacher_config)
    checkpoint = torch.load(teacher_path, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    
    return teacher_model
