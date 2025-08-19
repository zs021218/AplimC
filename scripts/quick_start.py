#!/usr/bin/env python3
"""
快速开始脚本 - 快速训练和评估多模态分类模型
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import ModelConfig, get_preset_config


def setup_experiment_directory(name: str, base_dir: str = "experiments") -> Path:
    """设置实验目录"""
    exp_dir = Path(base_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="快速开始多模态分类训练")
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='HDF5数据文件路径'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='simple',
        choices=['simple', 'lightweight', 'signal_only', 'image_only'],
        help='预设配置'
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        choices=['stokes', 'fluorescence', 'images'],
        default=None,
        help='指定使用的模态（覆盖预设）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖预设）'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='实验名称'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='仅进行评估（需要提供检查点）'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='检查点路径（用于评估或恢复训练）'
    )
    
    args = parser.parse_args()
    
    # 设置实验名称
    if args.name is None:
        modality_str = "_".join(args.modalities) if args.modalities else args.preset
        args.name = f"{modality_str}_experiment"
    
    # 设置实验目录
    exp_dir = setup_experiment_directory(args.name)
    print(f"实验目录: {exp_dir}")
    
    # 获取配置
    config = get_preset_config(args.preset)
    
    # 覆盖配置
    if args.modalities:
        config.modalities = args.modalities
    if args.epochs:
        config.max_epochs = args.epochs
    
    print(f"使用配置: {config}")
    
    # 保存配置
    config.save_yaml(exp_dir / "config.yaml")
    
    if args.eval_only:
        # 仅评估
        if not args.checkpoint:
            print("错误: 评估模式需要提供检查点路径")
            sys.exit(1)
        
        print("开始评估...")
        eval_cmd = f"""
        python scripts/evaluate.py \\
            --checkpoint {args.checkpoint} \\
            --data {args.data} \\
            --output-dir {exp_dir}/results \\
            --split test
        """
        print(f"执行命令: {eval_cmd}")
        os.system(eval_cmd.strip())
    
    else:
        # 训练
        print("开始训练...")
        train_cmd = f"""
        python scripts/train.py \\
            --config {exp_dir}/config.yaml \\
            --data {args.data} \\
            --output-dir {exp_dir} \\
            --name training
        """
        
        if args.checkpoint:
            train_cmd += f" --resume {args.checkpoint}"
        
        print(f"执行命令: {train_cmd}")
        os.system(train_cmd.strip())
        
        # 训练完成后自动评估最佳模型
        best_model = exp_dir / "training" / "checkpoints" / "best_model.pth"
        if best_model.exists():
            print("训练完成，开始评估最佳模型...")
            eval_cmd = f"""
            python scripts/evaluate.py \\
                --checkpoint {best_model} \\
                --data {args.data} \\
                --output-dir {exp_dir}/results \\
                --split test
            """
            print(f"执行命令: {eval_cmd}")
            os.system(eval_cmd.strip())
    
    print(f"实验完成！结果保存在: {exp_dir}")


if __name__ == "__main__":
    main()
