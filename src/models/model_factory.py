"""模型工厂和管理器"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

from .multimodal_classifier import MultiModalClassifier, create_model
from .config import ModelConfig, TrainingConfig

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model_from_config(config_name: str, **kwargs) -> MultiModalClassifier:
        """根据配置名称创建模型"""
        config = ModelConfig.get_model_config(config_name)
        config.update(kwargs)  # 覆盖默认配置
        
        return create_model(**config)
    
    @staticmethod
    def create_custom_model(
        num_classes: int,
        modalities: List[str],
        fusion_method: str = 'concat',
        **encoder_configs
    ) -> MultiModalClassifier:
        """创建自定义模型"""
        return create_model(
            num_classes=num_classes,
            modalities=modalities,
            fusion_method=fusion_method,
            **encoder_configs
        )
    
    @staticmethod
    def list_available_models() -> List[str]:
        """列出所有可用的预定义模型"""
        return ModelConfig.list_available_configs()

class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_dir: str = './models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def save_model(
        self,
        model: MultiModalClassifier,
        model_name: str,
        epoch: int = None,
        metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None
    ):
        """保存模型"""
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_classes': model.num_classes,
                'use_stokes': model.use_stokes,
                'use_fluorescence': model.use_fluorescence,
                'use_images': model.use_images,
                'fusion_method': model.fusion_method,
                'stokes_config': model.stokes_config,
                'fluorescence_config': model.fluorescence_config,
                'image_config': model.image_config
            },
            'num_parameters': model.get_num_parameters()
        }
        
        if epoch is not None:
            save_dict['epoch'] = epoch
        
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        if config is not None:
            save_dict['training_config'] = config
        
        # 保存模型
        model_path = self.model_dir / f"{model_name}.pth"
        torch.save(save_dict, model_path)
        
        # 保存配置为JSON（便于查看）
        config_path = self.model_dir / f"{model_name}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(save_dict['model_config'], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(
        self,
        model_name: str,
        device: torch.device = None,
        strict: bool = True
    ) -> tuple[MultiModalClassifier, Dict[str, Any]]:
        """加载模型"""
        model_path = self.model_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        
        # 重建模型
        model_config = checkpoint['model_config']
        model = MultiModalClassifier(**model_config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)
        
        self.logger.info(f"Model loaded from {model_path}")
        
        return model, checkpoint
    
    def list_saved_models(self) -> List[str]:
        """列出所有保存的模型"""
        model_files = list(self.model_dir.glob("*.pth"))
        return [f.stem for f in model_files]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        model_path = self.model_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'model_name': model_name,
            'num_parameters': checkpoint.get('num_parameters', 'Unknown'),
            'model_config': checkpoint.get('model_config', {}),
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'file_size_mb': model_path.stat().st_size / (1024 * 1024)
        }
        
        return info
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """比较多个模型"""
        comparison = {}
        
        for model_name in model_names:
            try:
                info = self.get_model_info(model_name)
                comparison[model_name] = info
            except FileNotFoundError:
                comparison[model_name] = {'error': 'Model not found'}
        
        return comparison

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(
        self,
        model: MultiModalClassifier,
        dataloader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                inputs = {}
                for key in ['stokes', 'fluorescence', 'images']:
                    if key in batch:
                        inputs[key] = batch[key].to(self.device)
                
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['labels'] = all_labels
        
        return metrics
    
    def benchmark_models(
        self,
        models: Dict[str, MultiModalClassifier],
        dataloader,
        model_manager: ModelManager = None
    ) -> Dict[str, Dict[str, float]]:
        """对比多个模型的性能"""
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                metrics = self.evaluate_model(model, dataloader)
                results[model_name] = metrics
                
                # 如果提供了模型管理器，保存评估结果
                if model_manager:
                    eval_results_path = model_manager.model_dir / f"{model_name}_evaluation.json"
                    with open(eval_results_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results

def get_model_summary(model: MultiModalClassifier) -> str:
    """获取模型摘要"""
    summary = []
    summary.append("="*60)
    summary.append("MODEL SUMMARY")
    summary.append("="*60)
    summary.append(f"Total Parameters: {model.get_num_parameters():,}")
    summary.append(f"Number of Classes: {model.num_classes}")
    summary.append(f"Fusion Method: {model.fusion_method}")
    summary.append("")
    summary.append("Modalities:")
    if model.use_stokes:
        summary.append(f"  ✓ Stokes Parameters: {model.stokes_config}")
    if model.use_fluorescence:
        summary.append(f"  ✓ Fluorescence Data: {model.fluorescence_config}")
    if model.use_images:
        summary.append(f"  ✓ Images: {model.image_config}")
    summary.append("")
    summary.append("Model Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只显示叶子模块
            summary.append(f"  {name}: {module}")
    summary.append("="*60)
    
    return "\n".join(summary)

if __name__ == "__main__":
    # 测试模型工厂
    print("Available models:")
    for model_name in ModelFactory.list_available_models():
        print(f"  - {model_name}")
    
    # 创建模型
    model = ModelFactory.create_model_from_config('multimodal_all')
    print(f"\nCreated model with {model.get_num_parameters():,} parameters")
    
    # 打印模型摘要
    print(get_model_summary(model))
    
    # 测试模型管理器
    manager = ModelManager('./test_models')
    
    # 保存模型
    save_path = manager.save_model(
        model=model,
        model_name='test_multimodal',
        epoch=10,
        metrics={'accuracy': 85.5, 'loss': 0.42}
    )
    print(f"\nModel saved to: {save_path}")
    
    # 加载模型
    loaded_model, checkpoint = manager.load_model('test_multimodal')
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'Unknown')}")
    
    # 获取模型信息
    info = manager.get_model_info('test_multimodal')
    print(f"\nModel info: {info}")
