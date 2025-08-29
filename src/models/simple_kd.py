#!/usr/bin/env python3
"""
极简知识蒸馏模型（Occam 版）
原则：仅保留必要组件 —— 教师/学生 logits 蒸馏 + 学生监督CE

组成：
- Teacher：`ImageEncoder` + 线性分类头
- Student：若干 `SignalEncoder`（按模态）拼接 + 简单 MLP 分类头
- Loss：alpha * CE(student, y) + (1-alpha) * T^2 * KL(student/T, teacher/T)

依赖于 `src/models/classifier.py` 中的 `SignalEncoder` 与 `ImageEncoder`。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .classifier import SignalEncoder, ImageEncoder


class SimpleTeacher(nn.Module):
    """基于图像的教师网络（冻结或可训练由外部控制）。"""

    def __init__(
        self,
        num_classes: int,
        num_views: int = 3,
        output_dim: int = 256,
        backbone: str = 'resnet18',
        use_pretrained: bool = True,
        freeze_layers: int = 6,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            num_views=num_views,
            output_dim=output_dim,
            dropout_rate=0.0,
            pretrained=use_pretrained,
            backbone=backbone,
            freeze_layers=freeze_layers,
        )
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        logits = self.classifier(features)
        return logits


class SimpleStudent(nn.Module):
    """基于信号的学生网络（多模态信号拼接 + 简单MLP）。"""

    def __init__(
        self,
        modalities: List[str],
        num_classes: int,
        stokes_dim: int = 4,
        stokes_length: int = 4000,
        fluorescence_dim: int = 16,
        fluorescence_length: int = 4000,
        encoder_output_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # 模态命名与范围校验
        valid_modalities = {'stokes', 'fluorescence'}
        normalized = []
        for m in modalities:
            if m not in valid_modalities:
                raise ValueError(f"不支持的学生模态: {m}. 允许: {sorted(valid_modalities)}")
            normalized.append(m)
        self.modalities = normalized
        self.encoders = nn.ModuleDict()

        if 'stokes' in modalities:
            self.encoders['stokes'] = SignalEncoder(
                input_channels=stokes_dim,
                input_length=stokes_length,
                output_dim=encoder_output_dim,
                dropout_rate=dropout_rate,
            )
        if 'fluorescence' in modalities:
            self.encoders['fluorescence'] = SignalEncoder(
                input_channels=fluorescence_dim,
                input_length=fluorescence_length,
                output_dim=encoder_output_dim,
                dropout_rate=dropout_rate,
            )

        fusion_dim = encoder_output_dim * len(self.encoders)

        layers: List[nn.Module] = []
        in_dim = fusion_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features: List[torch.Tensor] = []
        for m in self.modalities:
            if m in batch:
                features.append(self.encoders[m](batch[m]))
        fused = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        logits = self.classifier(fused)
        return logits


class SimpleKDLoss(nn.Module):
    """标准KD损失（Hinton）+ 学生CE支持标签平滑。"""

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        # 标签平滑（需要 PyTorch 支持该参数的版本）
        try:
            self.ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        except TypeError:
            # 旧版本PyTorch不支持该参数，退化为普通CE
            self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        T = self.temperature
        ce_loss = self.ce(student_logits, labels)
        kd_loss = self.kl(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T)
        total = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
        return {
            'total_loss': total,
            'ce_loss': ce_loss,
            'kd_loss': kd_loss,
        }


class SimpleKDModel(nn.Module):
    """封装：教师 + 学生 + 前向产出 logits（不在此处计算损失）。"""

    def __init__(
        self,
        teacher: SimpleTeacher,
        student: SimpleStudent,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_logits = self.teacher(batch['images'])
        student_logits = self.student(batch)
        return {
            'teacher_logits': teacher_logits,
            'student_logits': student_logits,
        }


__all__ = [
    'SimpleTeacher',
    'SimpleStudent',
    'SimpleKDLoss',
    'SimpleKDModel',
]


