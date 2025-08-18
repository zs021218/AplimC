#!/usr/bin/env python3
"""
模型工具函数
包括权重初始化、模型分析、参数统计等实用功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def analyze_model(model: nn.Module) -> Dict[str, Any]:
    """分析模型结构和参数"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # 按层类型统计参数
    layer_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            layer_type = module.__class__.__name__
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {'count': 0, 'params': 0}
            
            layer_stats[layer_type]['count'] += 1
            layer_stats[layer_type]['params'] += count_parameters(module)
    
    # 计算模型大小（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_size / (1024 * 1024)
    
    analysis = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'layer_statistics': layer_stats
    }
    
    return analysis


def print_model_summary(model: nn.Module, input_shapes: Optional[Dict[str, Tuple]] = None):
    """打印模型摘要"""
    analysis = analyze_model(model)
    
    print("=" * 60)
    print(f"{'模型摘要':^60}")
    print("=" * 60)
    print(f"模型类型: {model.__class__.__name__}")
    print(f"总参数量: {analysis['total_parameters']:,}")
    print(f"可训练参数: {analysis['trainable_parameters']:,}")
    print(f"非可训练参数: {analysis['non_trainable_parameters']:,}")
    print(f"模型大小: {analysis['model_size_mb']:.2f} MB")
    
    print("\n" + "-" * 60)
    print(f"{'层类型':^20} {'数量':^10} {'参数数':^15} {'占比':^15}")
    print("-" * 60)
    
    for layer_type, stats in analysis['layer_statistics'].items():
        percentage = (stats['params'] / analysis['total_parameters']) * 100
        print(f"{layer_type:<20} {stats['count']:^10} {stats['params']:^15,} {percentage:^15.2f}%")
    
    if input_shapes:
        print("\n" + "-" * 60)
        print("输入形状:")
        for modality, shape in input_shapes.items():
            print(f"  {modality}: {shape}")
    
    print("=" * 60)


def initialize_weights(model: nn.Module, init_type: str = 'kaiming'):
    """初始化模型权重"""
    
    def init_func(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_func)
    logger.info(f"使用 {init_type} 初始化完成")


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """冻结指定层的参数"""
    frozen_params = 0
    
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                frozen_params += param.numel()
                break
    
    logger.info(f"冻结了 {frozen_params:,} 个参数")


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """解冻指定层的参数"""
    unfrozen_params = 0
    
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
                break
    
    logger.info(f"解冻了 {unfrozen_params:,} 个参数")


def get_model_complexity(model: nn.Module, input_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """计算模型计算复杂度"""
    model.eval()
    
    # 记录FLOPs
    flops = 0
    handles = []
    
    def conv_flop_hook(module, input, output):
        nonlocal flops
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            input_dims = input[0].shape
            output_dims = output.shape
            kernel_flops = np.prod(module.kernel_size) * module.in_channels
            output_elements = np.prod(output_dims)
            flops += kernel_flops * output_elements
    
    def linear_flop_hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Linear):
            flops += module.in_features * module.out_features * input[0].shape[0]
    
    # 注册钩子
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(conv_flop_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_flop_hook))
    
    # 前向传播计算FLOPs
    with torch.no_grad():
        _ = model(input_batch)
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 计算内存使用
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    complexity = {
        'flops': flops,
        'flops_m': flops / 1e6,  # MFLOPs
        'flops_g': flops / 1e9,  # GFLOPs
        'param_memory_mb': param_memory / (1024 * 1024),
        'parameters': count_parameters(model)
    }
    
    return complexity


def compare_models(models: Dict[str, nn.Module], input_batch: Dict[str, torch.Tensor]):
    """比较多个模型"""
    print("=" * 80)
    print(f"{'模型比较':^80}")
    print("=" * 80)
    print(f"{'模型名称':<15} {'参数量':<12} {'大小(MB)':<10} {'FLOPs(M)':<12} {'内存(MB)':<12}")
    print("-" * 80)
    
    for name, model in models.items():
        analysis = analyze_model(model)
        complexity = get_model_complexity(model, input_batch)
        
        print(f"{name:<15} {analysis['total_parameters']:<12,} "
              f"{analysis['model_size_mb']:<10.2f} {complexity['flops_m']:<12.2f} "
              f"{complexity['param_memory_mb']:<12.2f}")
    
    print("=" * 80)


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Optional[Dict] = None
):
    """保存模型检查点"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': getattr(model, 'config', None),
        'model_class': model.__class__.__name__
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    logger.info(f"模型检查点已保存: {save_path}")


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    load_optimizer: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """加载模型检查点"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'model_class': checkpoint.get('model_class', 'Unknown')
    }
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        info['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    logger.info(f"模型检查点已加载: {checkpoint_path}")
    logger.info(f"轮次: {info['epoch']}, 损失: {info['loss']:.4f}")
    
    return info


def find_optimal_batch_size(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    max_batch_size: int = 256,
    device: Optional[torch.device] = None
) -> int:
    """寻找最优批次大小"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # 测试不同批次大小
    batch_sizes = [2**i for i in range(1, int(np.log2(max_batch_size)) + 1)]
    optimal_batch_size = 1
    
    for batch_size in batch_sizes:
        try:
            # 创建测试批次
            test_batch = {}
            for key, tensor in sample_input.items():
                # 复制到指定批次大小
                batch_tensor = tensor.repeat(batch_size, *([1] * (tensor.dim() - 1)))
                test_batch[key] = batch_tensor.to(device)
            
            # 测试前向传播
            with torch.no_grad():
                outputs = model(test_batch)
            
            # 测试反向传播
            loss = outputs.sum()
            loss.backward()
            
            # 清理
            del test_batch, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            optimal_batch_size = batch_size
            logger.info(f"批次大小 {batch_size} 测试成功")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"批次大小 {batch_size} 内存不足")
                break
            else:
                raise e
    
    logger.info(f"推荐的最优批次大小: {optimal_batch_size}")
    return optimal_batch_size


if __name__ == "__main__":
    # 测试工具函数
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from models.classifier import create_simple_model
    
    # 创建测试模型
    model = create_simple_model(num_classes=12)
    
    # 创建测试输入
    test_input = {
        'stokes': torch.randn(2, 4, 4000),
        'fluorescence': torch.randn(2, 16, 4000),
        'images': torch.rand(2, 3, 224, 224, 3)
    }
    
    # 测试分析函数
    print("=== 模型分析 ===")
    print_model_summary(model, {
        'stokes': (4, 4000),
        'fluorescence': (16, 4000),
        'images': (3, 224, 224, 3)
    })
    
    # 测试复杂度计算
    print("\n=== 计算复杂度 ===")
    complexity = get_model_complexity(model, test_input)
    for key, value in complexity.items():
        print(f"{key}: {value}")
    
    # 测试不同模型配置
    print("\n=== 模型配置比较 ===")
    from models.classifier import create_simple_model
    
    models = {
        'Full': create_simple_model(12, ['stokes', 'fluorescence', 'images']),
        'Signal': create_simple_model(12, ['stokes', 'fluorescence']),
        'Stokes': create_simple_model(12, ['stokes']),
    }
    
    # 为不同模型创建对应输入
    inputs = {
        'Full': test_input,
        'Signal': {k: v for k, v in test_input.items() if k != 'images'},
        'Stokes': {'stokes': test_input['stokes']}
    }
    
    for name, model in models.items():
        complexity = get_model_complexity(model, inputs[name])
        print(f"{name}: {complexity['parameters']:,} 参数, {complexity['flops_m']:.2f} MFLOPs")
