#!/usr/bin/env python3
"""
极简知识蒸馏训练脚本（Occam 版）
依赖 HDF5 数据集：需同时加载 `images` 与信号模态（如 'stokes','fluorescence'）。

示例：
python scripts/train_simple_kd.py \
  --data /path/to/multimodal_data.h5 \
  --num-classes 12 \
  --student-modalities stokes fluorescence \
  --batch-size 32 --epochs 80 --lr 1e-3 \
  --alpha 0.5 --temperature 4.0 --device cuda:0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import logging

from src.models.simple_kd import SimpleTeacher, SimpleStudent, SimpleKDLoss, SimpleKDModel
from src.data.dataset import MultimodalHDF5Dataset
from src.utils.config import load_config_from_yaml
from torch.utils.tensorboard import SummaryWriter


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train_simple_kd.log'),
            logging.StreamHandler()
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Occam 简洁KD训练')
    parser.add_argument('--config', type=str, default='', help='YAML 配置文件路径（可选）')
    parser.add_argument('--data', type=str, default='', help='HDF5 数据路径（若使用 --config 可在 data.hdf5_path 指定）')
    parser.add_argument('--num-classes', type=int, default=12)
    parser.add_argument('--num-views', type=int, default=3)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--student-modalities', nargs='+', default=['stokes','fluorescence'])
    parser.add_argument('--stokes-dim', type=int, default=4)
    parser.add_argument('--stokes-length', type=int, default=4000)
    parser.add_argument('--fluorescence-dim', type=int, default=16)
    parser.add_argument('--fluorescence-length', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5, help='CE权重')
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='学生CE的标签平滑系数')
    parser.add_argument('--exp-name', type=str, default='simple_kd')
    parser.add_argument('--teacher', type=str, default='', help='教师模型权重路径（可选）')
    parser.add_argument('--freeze-teacher', action='store_true', help='冻结教师参数（默认不冻结由此标志控制）')
    return parser.parse_args()


def create_dataloaders(hdf5_path: str, batch_size: int, modalities):
    # 模态命名与范围校验
    valid_student = {'stokes', 'fluorescence'}
    normalized_student = []
    for m in modalities:
        if m not in valid_student:
            raise ValueError(f"学生模态无效: {m}. 允许: {sorted(valid_student)}")
        normalized_student.append(m)

    # 数据加载需要包含 images 给教师网络
    load_modalities = list(sorted(set(normalized_student + ['images'])))
    train_ds = MultimodalHDF5Dataset(hdf5_path, split='train', load_modalities=load_modalities)
    val_ds = MultimodalHDF5Dataset(hdf5_path, split='val', load_modalities=load_modalities)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def evaluate_student(model: SimpleKDModel, val_loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()
    loss_sum = 0.0
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            labels = batch['labels']
            outputs = model(batch)
            student_logits = outputs['student_logits']
            loss = ce(student_logits, labels)
            loss_sum += loss.item()
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return {
        'loss': loss_sum / max(1, len(val_loader)),
        'acc': 100.0 * correct / max(1, total)
    }


def main():
    args = parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA 不可用，使用 CPU')
        args.device = 'cpu'
    device = torch.device(args.device)

    exp_dir = Path(f'experiments/simple_kd/{args.exp_name}')
    setup_logging(exp_dir)
    writer = SummaryWriter(log_dir=str(exp_dir / 'tb'))
    logger = logging.getLogger('simple_kd')
    logger.info('启动 Occam 简洁KD 训练')
    if args.teacher:
        logger.info(f'使用教师权重: {args.teacher}')

    # 配置文件覆盖命令行（可选）
    if args.config:
        cfg = load_config_from_yaml(args.config)
        # 必需字段
        if 'data' in cfg and 'hdf5_path' in cfg['data']:
            args.data = cfg['data']['hdf5_path']
        if 'model' in cfg:
            m = cfg['model']
            args.num_classes = m.get('num_classes', args.num_classes)
            args.num_views = m.get('num_views', args.num_views)
            args.backbone = m.get('backbone', args.backbone)
            # 学生模态
            if 'student_modalities' in m:
                args.student_modalities = m['student_modalities']
            # 尺度
            args.stokes_dim = m.get('stokes_dim', args.stokes_dim)
            args.stokes_length = m.get('stokes_length', args.stokes_length)
            args.fluorescence_dim = m.get('fluorescence_dim', args.fluorescence_dim)
            args.fluorescence_length = m.get('fluorescence_length', args.fluorescence_length)
        if 'training' in cfg:
            t = cfg['training']
            args.batch_size = int(t.get('batch_size', args.batch_size))
            args.epochs = int(t.get('num_epochs', t.get('epochs', args.epochs)))
            args.lr = float(t.get('learning_rate', args.lr))
            args.weight_decay = float(t.get('weight_decay', args.weight_decay))
        if 'distillation' in cfg:
            d = cfg['distillation']
            args.alpha = float(d.get('alpha', args.alpha))
            args.temperature = float(d.get('temperature', args.temperature))
            args.label_smoothing = float(d.get('label_smoothing', d.get('label-smoothing', args.label_smoothing)))

    # 最终数据路径校验
    if not args.data:
        raise SystemExit("必须提供数据路径：使用 --data 或在 --config 的 data.hdf5_path 指定")

    # 数据
    train_loader, val_loader = create_dataloaders(args.data, args.batch_size, args.student_modalities)
    logger.info(f'Train iters: {len(train_loader)}, Val iters: {len(val_loader)}')

    # 模型
    teacher = SimpleTeacher(
        num_classes=args.num_classes,
        num_views=args.num_views,
        backbone=args.backbone,
        use_pretrained=True,
        freeze_layers=6,
    )

    # 可选：加载教师权重
    if args.teacher and os.path.exists(args.teacher):
        try:
            # 优先使用更安全的 weights_only 模式，若不支持则回退
            try:
                ckpt = torch.load(args.teacher, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(args.teacher, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            remapped = {}
            loaded = 0
            for k, v in state.items():
                new_k = None
                if k.startswith('encoders.images.'):
                    new_k = k.replace('encoders.images.', 'image_encoder.')
                elif k.startswith('image_encoder.'):
                    new_k = k
                elif k.startswith('classifier.'):
                    new_k = k
                elif k.startswith('teacher.image_encoder.'):
                    new_k = k.replace('teacher.', '')
                elif k.startswith('teacher.classifier.'):
                    new_k = k.replace('teacher.', '')
                if new_k is not None:
                    remapped[new_k] = v
                    loaded += 1
            if loaded == 0:
                # 尝试直接加载
                teacher.load_state_dict(state, strict=False)
                logger.info('直接加载教师权重（strict=False）完成')
            else:
                teacher.load_state_dict(remapped, strict=False)
                logger.info(f'重映射并加载教师权重完成，匹配参数: {loaded}')
        except Exception as e:
            logger.warning(f'加载教师权重失败，将使用随机初始化的教师: {e}')
    student = SimpleStudent(
        modalities=args.student_modalities,
        num_classes=args.num_classes,
        stokes_dim=args.stokes_dim,
        stokes_length=args.stokes_length,
        fluorescence_dim=args.fluorescence_dim,
        fluorescence_length=args.fluorescence_length,
        hidden_dims=[256, 128],
        dropout_rate=0.2,
    )

    model = SimpleKDModel(teacher=teacher, student=student, freeze_teacher=bool(args.freeze_teacher)).to(device)
    kd_loss = SimpleKDLoss(alpha=args.alpha, temperature=args.temperature, label_smoothing=args.label_smoothing)

    # 优化器（只优化学生）
    optimizer = optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = {'total': 0.0, 'ce': 0.0, 'kd': 0.0}
        correct = 0
        total = 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            labels = batch['labels']

            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss_dict = kd_loss(out['student_logits'], out['teacher_logits'], labels)
            loss_dict['total_loss'].backward()
            nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=1.0)
            optimizer.step()

            running['total'] += loss_dict['total_loss'].item()
            running['ce'] += loss_dict['ce_loss'].item()
            running['kd'] += loss_dict['kd_loss'].item()

            pred = out['student_logits'].argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running['total'] / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        val_metrics = evaluate_student(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Val Loss {val_metrics['loss']:.4f} Acc {val_metrics['acc']:.2f}%")

        # TensorBoard 记录
        global_step = epoch + 1
        writer.add_scalar('train/loss_total', train_loss, global_step)
        writer.add_scalar('train/loss_ce', running['ce'] / max(1, len(train_loader)), global_step)
        writer.add_scalar('train/loss_kd', running['kd'] / max(1, len(train_loader)), global_step)
        writer.add_scalar('train/acc', train_acc, global_step)
        writer.add_scalar('val/loss', val_metrics['loss'], global_step)
        writer.add_scalar('val/acc', val_metrics['acc'], global_step)
        # 学习率
        try:
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train/lr', current_lr, global_step)
        except Exception:
            pass

        # 保存最佳学生
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save({
                'epoch': epoch + 1,
                'student_state_dict': model.student.state_dict(),
                'best_val_acc': best_acc,
                'args': vars(args),
            }, exp_dir / 'best_student.pth')
            logger.info(f'保存最佳学生模型：Val Acc {best_acc:.2f}%')

    logger.info(f'完成训练，最佳 Val Acc: {best_acc:.2f}%')
    writer.close()


if __name__ == '__main__':
    main()


