import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from abc import ABC, abstractmethod

class BaseEncoder(nn.Module, ABC):
    """编码器基类"""
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def get_output_dim(self):
        pass

class StokesEncoder(BaseEncoder):
    """Stokes参数编码器 - 1D CNN"""
    
    def __init__(self, input_channels=4, signal_length=4000, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.input_channels = input_channels
        self.signal_length = signal_length
        self.hidden_dims = hidden_dims
        
        # 1D CNN layers
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.1)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积层输出的长度
        conv_output_length = signal_length
        for _ in hidden_dims:
            conv_output_length = conv_output_length // 2
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1] * conv_output_length, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        self.output_dim = 256
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 4, signal_length)
        Returns:
            features: (batch_size, 256)
        """
        batch_size = x.size(0)
        
        # 1D CNN
        x = self.conv_layers(x)  # (batch_size, hidden_dims[-1], reduced_length)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # FC layers
        features = self.fc_layers(x)
        
        return features
    
    def get_output_dim(self):
        return self.output_dim

class FluorescenceEncoder(BaseEncoder):
    """荧光数据编码器 - 1D CNN + Attention"""
    
    def __init__(self, input_channels=16, signal_length=4000, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.input_channels = input_channels
        self.signal_length = signal_length
        self.hidden_dims = hidden_dims
        
        # 1D CNN layers
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.1)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 计算卷积层输出的长度
        conv_output_length = signal_length
        for _ in hidden_dims:
            conv_output_length = conv_output_length // 2
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1] * conv_output_length, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        self.output_dim = 256
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 16, signal_length)
        Returns:
            features: (batch_size, 256)
        """
        batch_size = x.size(0)
        
        # 1D CNN
        x = self.conv_layers(x)  # (batch_size, hidden_dims[-1], reduced_length)
        
        # Prepare for attention: (batch_size, seq_len, features)
        x_att = x.transpose(1, 2)  # (batch_size, reduced_length, hidden_dims[-1])
        
        # Self-attention
        x_att, _ = self.attention(x_att, x_att, x_att)
        
        # Back to original format: (batch_size, hidden_dims[-1], reduced_length)
        x = x_att.transpose(1, 2)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # FC layers
        features = self.fc_layers(x)
        
        return features
    
    def get_output_dim(self):
        return self.output_dim

class ImageEncoder(BaseEncoder):
    """图像编码器 - 3D CNN + ResNet-like blocks"""
    
    def __init__(self, num_views=3, image_size=(224, 224), hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        self.num_views = num_views
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        
        # 3D CNN for multi-view images
        # Input: (batch_size, 3, 224, 224, 3) -> treat as (batch_size, 3*3, 224, 224)
        self.view_conv = nn.Sequential(
            nn.Conv2d(3 * num_views, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # ResNet-like blocks
        self.res_blocks = nn.ModuleList()
        in_channels = 64
        
        for hidden_dim in hidden_dims:
            self.res_blocks.append(
                self._make_res_block(in_channels, hidden_dim)
            )
            in_channels = hidden_dim
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 最终特征维度
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        self.output_dim = 256
    
    def _make_res_block(self, in_channels, out_channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 224, 224, 3)
        Returns:
            features: (batch_size, 256)
        """
        batch_size = x.size(0)
        
        # Reshape multi-view images: (batch_size, 3, 224, 224, 3) -> (batch_size, 9, 224, 224)
        x = x.view(batch_size, -1, self.image_size[0], self.image_size[1])
        
        # Initial convolution
        x = self.view_conv(x)
        
        # ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, hidden_dims[-1], 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, hidden_dims[-1])
        
        # Final FC
        features = self.fc(x)
        
        return features
    
    def get_output_dim(self):
        return self.output_dim

class MultiModalFusionLayer(nn.Module):
    """多模态融合层"""
    
    def __init__(self, input_dims: List[int], fusion_method='concat', output_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.fusion_method = fusion_method
        self.output_dim = output_dim
        
        if fusion_method == 'concat':
            total_dim = sum(input_dims)
            self.fusion_fc = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif fusion_method == 'attention':
            # Cross-modal attention
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                ) for dim in input_dims
            ])
            # 假设所有模态特征维度相同
            assert len(set(input_dims)) == 1, "Attention fusion requires same feature dimensions"
            self.fusion_fc = nn.Sequential(
                nn.Linear(input_dims[0], output_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif fusion_method == 'weighted_sum':
            # 假设所有模态特征维度相同
            assert len(set(input_dims)) == 1, "Weighted sum requires same feature dimensions"
            self.weights = nn.Parameter(torch.ones(len(input_dims)) / len(input_dims))
            self.fusion_fc = nn.Sequential(
                nn.Linear(input_dims[0], output_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
    
    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: List of feature tensors from different modalities
        Returns:
            fused_features: (batch_size, output_dim)
        """
        if self.fusion_method == 'concat':
            # 简单拼接
            fused = torch.cat(features, dim=1)
            return self.fusion_fc(fused)
        
        elif self.fusion_method == 'attention':
            # 交叉注意力机制
            attended_features = []
            for i, (feature, attention) in enumerate(zip(features, self.attention_layers)):
                # 使用其他模态作为query, key, value
                other_features = [f for j, f in enumerate(features) if j != i]
                if other_features:
                    context = torch.stack(other_features, dim=1)  # (batch, num_others, dim)
                    feature_expanded = feature.unsqueeze(1)  # (batch, 1, dim)
                    attended, _ = attention(feature_expanded, context, context)
                    attended_features.append(attended.squeeze(1))
                else:
                    attended_features.append(feature)
            
            # 平均融合
            fused = torch.stack(attended_features, dim=0).mean(dim=0)
            return self.fusion_fc(fused)
        
        elif self.fusion_method == 'weighted_sum':
            # 加权求和
            weights = F.softmax(self.weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, features))
            return self.fusion_fc(fused)

class MultiModalClassifier(nn.Module):
    """多模态分类器主模型"""
    
    def __init__(
        self,
        num_classes: int,
        use_stokes: bool = True,
        use_fluorescence: bool = True,
        use_images: bool = True,
        fusion_method: str = 'concat',
        stokes_config: Dict = None,
        fluorescence_config: Dict = None,
        image_config: Dict = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_stokes = use_stokes
        self.use_fluorescence = use_fluorescence
        self.use_images = use_images
        self.fusion_method = fusion_method
        
        # 默认配置
        default_stokes_config = {'input_channels': 4, 'signal_length': 4000}
        default_fluorescence_config = {'input_channels': 16, 'signal_length': 4000}
        default_image_config = {'num_views': 3, 'image_size': (224, 224)}
        
        self.stokes_config = {**default_stokes_config, **(stokes_config or {})}
        self.fluorescence_config = {**default_fluorescence_config, **(fluorescence_config or {})}
        self.image_config = {**default_image_config, **(image_config or {})}
        
        # 编码器
        self.encoders = nn.ModuleDict()
        feature_dims = []
        
        if use_stokes:
            self.encoders['stokes'] = StokesEncoder(**self.stokes_config)
            feature_dims.append(self.encoders['stokes'].get_output_dim())
        
        if use_fluorescence:
            self.encoders['fluorescence'] = FluorescenceEncoder(**self.fluorescence_config)
            feature_dims.append(self.encoders['fluorescence'].get_output_dim())
        
        if use_images:
            self.encoders['images'] = ImageEncoder(**self.image_config)
            feature_dims.append(self.encoders['images'].get_output_dim())
        
        # 融合层
        if len(feature_dims) > 1:
            self.fusion_layer = MultiModalFusionLayer(
                input_dims=feature_dims,
                fusion_method=fusion_method,
                output_dim=512
            )
            classifier_input_dim = 512
        else:
            self.fusion_layer = None
            classifier_input_dim = feature_dims[0] if feature_dims else 256
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: Dict[str, torch.Tensor], return_features=False):
        """
        Args:
            batch: Dictionary containing:
                - 'stokes': (batch_size, 4, signal_length) [optional]
                - 'fluorescence': (batch_size, 16, signal_length) [optional]
                - 'images': (batch_size, 3, 224, 224, 3) [optional]
            return_features: Whether to return intermediate features
        
        Returns:
            logits: (batch_size, num_classes)
            features: Dictionary of features (if return_features=True)
        """
        features = {}
        encoded_features = []
        
        # 编码各个模态
        if self.use_stokes and 'stokes' in batch:
            features['stokes'] = self.encoders['stokes'](batch['stokes'])
            encoded_features.append(features['stokes'])
        
        if self.use_fluorescence and 'fluorescence' in batch:
            features['fluorescence'] = self.encoders['fluorescence'](batch['fluorescence'])
            encoded_features.append(features['fluorescence'])
        
        if self.use_images and 'images' in batch:
            features['images'] = self.encoders['images'](batch['images'])
            encoded_features.append(features['images'])
        
        # 融合特征
        if len(encoded_features) > 1 and self.fusion_layer is not None:
            fused_features = self.fusion_layer(encoded_features)
            features['fused'] = fused_features
        elif len(encoded_features) == 1:
            fused_features = encoded_features[0]
            features['fused'] = fused_features
        else:
            raise ValueError("No valid modality data provided")
        
        # 分类
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, features
        else:
            return logits
    
    def get_num_parameters(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model(
    num_classes: int,
    modalities: List[str] = ['stokes', 'fluorescence', 'images'],
    fusion_method: str = 'concat',
    **kwargs
) -> MultiModalClassifier:
    """便捷函数：创建多模态分类器"""
    
    use_stokes = 'stokes' in modalities
    use_fluorescence = 'fluorescence' in modalities
    use_images = 'images' in modalities
    
    model = MultiModalClassifier(
        num_classes=num_classes,
        use_stokes=use_stokes,
        use_fluorescence=use_fluorescence,
        use_images=use_images,
        fusion_method=fusion_method,
        **kwargs
    )
    
    return model

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建多模态模型
    model = create_model(
        num_classes=12,
        modalities=['stokes', 'fluorescence', 'images'],
        fusion_method='concat'
    ).to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # 测试数据
    batch_size = 4
    test_batch = {
        'stokes': torch.randn(batch_size, 4, 4000).to(device),
        'fluorescence': torch.randn(batch_size, 16, 4000).to(device),
        'images': torch.randn(batch_size, 3, 224, 224, 3).to(device)
    }
    
    # 前向传播
    with torch.no_grad():
        logits, features = model(test_batch, return_features=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Features keys: {features.keys()}")
    for key, feature in features.items():
        print(f"  {key}: {feature.shape}")
    
    # 测试单模态
    print("\n--- Single Modality Tests ---")
    
    # 仅使用 Stokes
    stokes_model = create_model(num_classes=12, modalities=['stokes']).to(device)
    logits_stokes = stokes_model({'stokes': test_batch['stokes']})
    print(f"Stokes-only output: {logits_stokes.shape}")
    
    # 仅使用荧光
    fluor_model = create_model(num_classes=12, modalities=['fluorescence']).to(device)
    logits_fluor = fluor_model({'fluorescence': test_batch['fluorescence']})
    print(f"Fluorescence-only output: {logits_fluor.shape}")
    
    # 仅使用图像
    image_model = create_model(num_classes=12, modalities=['images']).to(device)
    logits_image = image_model({'images': test_batch['images']})
    print(f"Images-only output: {logits_image.shape}")
