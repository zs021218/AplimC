#!/usr/bin/env python3
"""
模型评估脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.utils.config import ModelConfig
from src.models import MultimodalClassifier
from src.data import create_default_dataloader


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_model(checkpoint_path: str, device: torch.device):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点恢复配置
    config = ModelConfig.from_dict(checkpoint['config'])
    
    # 创建模型
    model = MultimodalClassifier(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, data_loader, device, class_names=None):
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            # 数据移动到设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # 生成分类报告
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(max(all_labels) + 1)]
    
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_classification_report(report, save_path=None):
    """绘制分类报告热图"""
    # 提取数值数据
    data = []
    labels = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            data.append([metrics['precision'], metrics['recall'], metrics['f1-score']])
            labels.append(class_name)
    
    data = np.array(data)
    
    plt.figure(figsize=(8, len(labels) * 0.5))
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        xticklabels=['Precision', 'Recall', 'F1-Score'],
        yticklabels=labels
    )
    plt.title('分类性能指标')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='HDF5数据文件路径'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='评估的数据分割'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='结果保存目录'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='批次大小'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='使用的GPU ID'
    )
    parser.add_argument(
        '--class-names',
        nargs='+',
        default=None,
        help='类别名称列表'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"加载模型: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # 创建数据加载器
    logger.info(f"加载数据: {args.data}")
    data_loader = create_default_dataloader(
        hdf5_path=args.data,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        balanced=False,
        use_cache=True
    )
    
    logger.info(f"评估样本数: {len(data_loader.dataset)}")
    
    # 评估模型
    logger.info("开始评估...")
    results = evaluate_model(model, data_loader, device, args.class_names)
    
    # 打印结果
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    # 保存分类报告
    report_path = output_dir / 'classification_report.txt'
    from sklearn.metrics import classification_report
    with open(report_path, 'w') as f:
        f.write(classification_report(
            results['labels'],
            results['predictions'],
            target_names=results['class_names']
        ))
    
    logger.info(f"分类报告已保存: {report_path}")
    
    # 绘制混淆矩阵
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        results['confusion_matrix'],
        results['class_names'],
        str(cm_path)
    )
    logger.info(f"混淆矩阵已保存: {cm_path}")
    
    # 绘制分类报告
    report_plot_path = output_dir / 'classification_metrics.png'
    plot_classification_report(
        results['classification_report'],
        str(report_plot_path)
    )
    logger.info(f"分类指标图已保存: {report_plot_path}")
    
    # 保存详细结果
    np.savez(
        output_dir / 'evaluation_results.npz',
        predictions=results['predictions'],
        labels=results['labels'],
        probabilities=results['probabilities'],
        confusion_matrix=results['confusion_matrix']
    )
    
    logger.info("评估完成！")


if __name__ == "__main__":
    main()
