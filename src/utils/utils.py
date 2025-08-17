"""
通用工具函数
"""

import yaml
import json
import logging
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger('multimodal_training')
    logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    return config

def save_config(config: Dict[str, Any], save_path: Path):
    """保存配置文件"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, ensure_ascii=False, indent=2)
        elif save_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")

def set_seed(seed: int = 42):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> tuple:
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_device(device_name: str = 'auto') -> torch.device:
    """获取计算设备"""
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    return device

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_experiment_name(config: Dict[str, Any], timestamp: bool = True) -> str:
    """创建实验名称"""
    model_type = config.get('model', {}).get('type', 'multimodal')
    fusion_method = config.get('model', {}).get('fusion_method', 'concat')
    lr = config.get('training', {}).get('learning_rate', 1e-3)
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    name = f"{model_type}_{fusion_method}_lr{lr}_bs{batch_size}"
    
    if timestamp:
        ts = datetime.now().strftime('%M%S%f')[:-3]  # 分钟秒毫秒
        name += f"_{ts}"
    
    return name

def save_metrics(metrics: Dict[str, Any], save_path: Path):
    """保存指标到JSON文件"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    metrics = convert_numpy(metrics)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_metrics(load_path: Path) -> Dict[str, Any]:
    """从JSON文件加载指标"""
    with open(load_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """保存最佳权重"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def print_model_summary(model: torch.nn.Module, input_size: tuple = None):
    """打印模型摘要"""
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    if input_size and torch.cuda.is_available():
        try:
            from torchsummary import summary
            summary(model, input_size)
        except ImportError:
            print("安装 torchsummary 包以查看详细的模型结构")
    
    print("=" * 80)

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_iters: int, warmup_factor: float):
    """学习率预热调度器"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def save_model_state(model: torch.nn.Module, save_path: Path, metadata: Dict[str, Any] = None):
    """保存模型状态"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        state['metadata'] = metadata
    
    torch.save(state, save_path)

def load_model_state(model: torch.nn.Module, load_path: Path, device: torch.device = None):
    """加载模型状态"""
    if device is None:
        device = torch.device('cpu')
    
    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    return state.get('metadata', {})

class ProgressMeter:
    """进度显示器"""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
