#!/usr/bin/env python3
"""
多模态分类器 - 基于奥卡姆剃刀原理的简单有效设计
支持灵活的模态选择和配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass

# 导入配置类
from ..utils.config import ModelConfig

logger = logging.getLogger(__name__)


class SignalEncoder(nn.Module):
    """信号编码器 - 简单的1D CNN + 全连接"""
    
    def __init__(
        self, 
        input_channels: int, 
        input_length: int,
        output_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_length = input_length
        
        # 1D卷积层 - 简单有效
        self.conv_layers = nn.Sequential(
            # 第一层：降采样
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # 第二层：特征提取
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # 第三层：深层特征
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, length)
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # 移除最后一维
        x = self.fc(x)
        return x


class ImageEncoder(nn.Module):
    """基于预训练ResNet的图像编码器 - 防止过拟合"""
    
    def __init__(
        self, 
        num_views: int = 3,
        output_dim: int = 256,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        backbone: str = 'resnet18'  # 可选择 resnet18, resnet34, resnet50
    ):
        super().__init__()
        
        self.num_views = num_views
        
        # 选择预训练模型
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 移除最后的分类层
        self.view_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 冻结早期层以防止过拟合
        self._freeze_early_layers()
        
        # 多视图融合层
        self.view_fusion = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * num_views, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制用于多视图融合
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"使用预训练{backbone}作为图像编码器骨干网络")
    
    def _freeze_early_layers(self):
        """冻结早期层以防止过拟合"""
        # 冻结前两个残差块
        for i, child in enumerate(self.view_encoder.children()):
            if i < 6:  # 冻结conv1, bn1, relu, maxpool, layer1, layer2
                for param in child.parameters():
                    param.requires_grad = False
        
        logger.info("已冻结ResNet的早期层以防止过拟合")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_views, height, width, channels)
        batch_size = x.shape[0]
        
        # 处理每个视图
        view_features = []
        attention_weights = []
        
        for i in range(self.num_views):
            # 调整维度: (batch, H, W, C) -> (batch, C, H, W)
            view = x[:, i].permute(0, 3, 1, 2)  # (batch, 3, H, W)
            
            # 特征提取
            view_feat = self.view_encoder(view)  # (batch, feature_dim, 1, 1)
            view_feat = view_feat.squeeze(-1).squeeze(-1)  # (batch, feature_dim)
            
            # 计算注意力权重
            attention = self.attention(view_feat)  # (batch, 1)
            
            view_features.append(view_feat)
            attention_weights.append(attention)
        
        # 使用注意力机制融合多视图特征
        stacked_features = torch.stack(view_features, dim=1)  # (batch, num_views, feature_dim)
        stacked_weights = torch.stack(attention_weights, dim=1)  # (batch, num_views, 1)
        
        # 归一化注意力权重
        stacked_weights = F.softmax(stacked_weights, dim=1)
        
        # 加权特征融合
        weighted_features = stacked_features * stacked_weights
        
        # 简单拼接所有视图特征（包含注意力信息）
        fused_features = weighted_features.view(batch_size, -1)  # (batch, num_views * feature_dim)
        
        # 最终输出
        output = self.view_fusion(fused_features)
        
        return output


class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # 特征投影
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, 1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 投影到统一维度
        projected_features = []
        for i, feat in enumerate(features):
            proj_feat = self.projections[i](feat)
            projected_features.append(proj_feat)
        
        # 计算注意力权重
        attention_weights = []
        for feat in projected_features:
            weight = self.attention(feat)
            attention_weights.append(weight)
        
        # 归一化权重
        attention_weights = torch.softmax(torch.cat(attention_weights, dim=1), dim=1)
        
        # 加权融合
        fused_feature = torch.zeros_like(projected_features[0])
        for i, feat in enumerate(projected_features):
            weight = attention_weights[:, i:i+1]
            fused_feature += weight * feat
        
        return fused_feature


class MultimodalClassifier(nn.Module):
    """多模态分类器 - 主模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        
        # 模态编码器
        self.encoders = nn.ModuleDict()
        # 收集编码器输出维度
        encoder_dims = []
        
        # 检查是否使用 Stokes 参数
        if 'stokes' in config.modalities:
            self.encoders['stokes'] = SignalEncoder(
                input_channels=config.stokes_dim,
                input_length=config.stokes_length,
                output_dim=256,
                dropout_rate=config.dropout_rate
            )
            encoder_dims.append(256)
        
        # 检查是否使用荧光信号
        if 'fluorescence' in config.modalities:
            self.encoders['fluorescence'] = SignalEncoder(
                input_channels=config.fluorescence_dim,
                input_length=config.fluorescence_length,
                output_dim=256,
                dropout_rate=config.dropout_rate
            )
            encoder_dims.append(256)
        
        # 检查是否使用图像
        if 'images' in config.modalities:
            self.encoders['images'] = ImageEncoder(
                num_views=config.num_views,
                output_dim=256,
                dropout_rate=config.dropout_rate
            )
            encoder_dims.append(256)
        
        # 特征融合
        if config.fusion_strategy == 'concat':
            fusion_dim = sum(encoder_dims)
            self.fusion = nn.Identity()
        elif config.fusion_strategy == 'attention':
            fusion_dim = 256
            self.fusion = AttentionFusion(encoder_dims, fusion_dim)
        else:  # weighted average
            fusion_dim = 256
            self.fusion = nn.Sequential(
                nn.Linear(sum(encoder_dims), fusion_dim),
                nn.ReLU(inplace=True)
            )
        
        # 分类头 - 简单的MLP
        classifier_layers = []
        input_dim = fusion_dim
        
        for hidden_dim in config.hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 输出层
        classifier_layers.append(nn.Linear(input_dim, self.num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        features = []
        
        # 编码各模态
        if 'stokes' in self.config.modalities and 'stokes' in batch:
            stokes_feat = self.encoders['stokes'](batch['stokes'])
            features.append(stokes_feat)
        
        if 'fluorescence' in self.config.modalities and 'fluorescence' in batch:
            flu_feat = self.encoders['fluorescence'](batch['fluorescence'])
            features.append(flu_feat)
        
        if 'images' in self.config.modalities and 'images' in batch:
            img_feat = self.encoders['images'](batch['images'])
            features.append(img_feat)
        
        # 特征融合
        if self.config.fusion_strategy == 'concat':
            fused_features = torch.cat(features, dim=1)
        elif self.config.fusion_strategy == 'attention':
            fused_features = self.fusion(features)
        else:  # weighted
            concatenated = torch.cat(features, dim=1)
            fused_features = self.fusion(concatenated)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_feature_dim(self) -> int:
        """获取融合特征维度"""
        if self.config.fusion_strategy == 'concat':
            total_dim = 0
            if 'stokes' in self.config.modalities:
                total_dim += 256
            if 'fluorescence' in self.config.modalities:
                total_dim += 256
            if 'images' in self.config.modalities:
                total_dim += 256
            return total_dim
        else:
            return 256


def create_model(config: Union[Dict, ModelConfig]) -> MultimodalClassifier:
    """创建模型的工厂函数"""
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    model = MultimodalClassifier(config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"创建模型: {model.__class__.__name__}")
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info(f"使用模态: {[k for k in ['stokes', 'fluorescence', 'images'] if getattr(config, f'use_{k}', False)]}")
    
    return model


def create_simple_model(num_classes: int = 12, modalities: List[str] = None) -> MultimodalClassifier:
    """创建简单配置的模型"""
    if modalities is None:
        modalities = ['stokes', 'fluorescence', 'images']
    
    config = ModelConfig(
        num_classes=num_classes,
        use_stokes='stokes' in modalities,
        use_fluorescence='fluorescence' in modalities,
        use_images='images' in modalities,
        hidden_dims=[256, 128],  # 更简单的分类头
        dropout_rate=0.2,
        fusion_strategy='concat'
    )
    
    return create_model(config)


if __name__ == "__main__":
    # 测试模型
    import torch
    
    # 创建测试数据
    batch_size = 4
    test_batch = {
        'stokes': torch.randn(batch_size, 4, 4000),
        'fluorescence': torch.randn(batch_size, 16, 4000),
        'images': torch.rand(batch_size, 3, 224, 224, 3),
    }
    
    # 测试完整模型
    print("=== 测试完整模型 ===")
    full_model = create_simple_model(num_classes=12)
    with torch.no_grad():
        outputs = full_model(test_batch)
        print(f"输出形状: {outputs.shape}")
        print(f"参数量: {sum(p.numel() for p in full_model.parameters()):,}")
    
    # 测试单模态模型
    print("\n=== 测试单模态模型 ===")
    stokes_model = create_simple_model(num_classes=12, modalities=['stokes'])
    with torch.no_grad():
        outputs = stokes_model({'stokes': test_batch['stokes']})
        print(f"Stokes模型输出: {outputs.shape}")
        print(f"参数量: {sum(p.numel() for p in stokes_model.parameters()):,}")
    
    # 测试双模态模型
    print("\n=== 测试双模态模型 ===")
    dual_model = create_simple_model(num_classes=12, modalities=['stokes', 'fluorescence'])
    with torch.no_grad():
        outputs = dual_model({
            'stokes': test_batch['stokes'],
            'fluorescence': test_batch['fluorescence']
        })
        print(f"双模态模型输出: {outputs.shape}")
        print(f"参数量: {sum(p.numel() for p in dual_model.parameters()):,}")
