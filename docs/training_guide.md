# 多模态分类器训练命令指南

## 🎯 训练命令概览

本项目遵循**奥卡姆剃刀原理**，提供从简单到复杂的模型配置，支持灵活的模态选择。

## 📁 配置文件说明

### 单模态配置
- `configs/stokes_only.yaml` - 仅Stokes参数分类器
- `configs/fluorescence_only.yaml` - 仅荧光信号分类器  
- `configs/images_only.yaml` - 仅图像分类器

### 双模态配置
- `configs/signals_dual.yaml` - Stokes + 荧光信号
- `configs/stokes_images_dual.yaml` - Stokes + 图像
- `configs/fluorescence_images_dual.yaml` - 荧光 + 图像

### 全模态配置
- `configs/full_multimodal.yaml` - 所有模态组合

## 🚀 快速开始

### 方式一：使用训练脚本（推荐）

```bash
# 进入项目目录
cd /data3/zs/AplimC

# 激活环境
source /data3/zs/miniconda3/bin/activate pytorch_env

# 使用训练脚本
./scripts/train_multimodal.sh single    # 训练所有单模态
./scripts/train_multimodal.sh dual      # 训练所有双模态
./scripts/train_multimodal.sh full      # 训练全模态
./scripts/train_multimodal.sh all       # 训练所有配置（比较实验）
./scripts/train_multimodal.sh quick     # 快速测试
```

### 方式二：直接使用Python命令

## 1️⃣ 单模态训练命令

### 1.1 Stokes参数分类器（最简单）
```bash
python scripts/train.py \
    --config configs/stokes_only.yaml \
    --gpu 0 \
    --experiment_name "stokes_only_experiment"
```

### 1.2 荧光信号分类器
```bash
python scripts/train.py \
    --config configs/fluorescence_only.yaml \
    --gpu 0 \
    --experiment_name "fluorescence_only_experiment"
```

### 1.3 图像分类器
```bash
python scripts/train.py \
    --config configs/images_only.yaml \
    --gpu 0 \
    --experiment_name "images_only_experiment"
```

## 2️⃣ 双模态训练命令

### 2.1 信号双模态（Stokes + 荧光）
```bash
python scripts/train.py \
    --config configs/signals_dual.yaml \
    --gpu 0 \
    --experiment_name "signals_dual_experiment"
```

### 2.2 Stokes + 图像
```bash
python scripts/train.py \
    --config configs/stokes_images_dual.yaml \
    --gpu 0 \
    --experiment_name "stokes_images_experiment"
```

### 2.3 荧光 + 图像
```bash
python scripts/train.py \
    --config configs/fluorescence_images_dual.yaml \
    --gpu 0 \
    --experiment_name "fluorescence_images_experiment"
```

## 3️⃣ 全模态训练命令

### 3.1 完整多模态分类器
```bash
python scripts/train.py \
    --config configs/full_multimodal.yaml \
    --gpu 0 \
    --experiment_name "full_multimodal_experiment"
```

## 🔧 高级训练选项

### 自定义训练参数
```bash
python scripts/train.py \
    --config configs/stokes_only.yaml \
    --gpu 0 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --max_epochs 50 \
    --experiment_name "custom_experiment"
```

### 分布式训练
```bash
# 多GPU训练
python scripts/train.py \
    --config configs/full_multimodal.yaml \
    --gpu 0,1,2,3 \
    --distributed true \
    --experiment_name "distributed_experiment"
```

### 断点续训
```bash
python scripts/train.py \
    --config configs/signals_dual.yaml \
    --resume experiments/signals_dual/checkpoints/last.ckpt \
    --gpu 0
```

## 📊 批量比较实验

### 完整性能对比
```bash
# 训练所有配置进行性能对比
./scripts/train_multimodal.sh all

# 或者手动批量训练
for config in stokes_only fluorescence_only images_only signals_dual stokes_images_dual fluorescence_images_dual full_multimodal; do
    python scripts/train.py \
        --config "configs/${config}.yaml" \
        --gpu 0 \
        --experiment_name "${config}_comparison"
done
```

## 📈 模型评估

### 单个模型评估
```bash
python scripts/evaluate.py \
    --config configs/stokes_only.yaml \
    --checkpoint experiments/stokes_only/best_model.pth \
    --output_dir results/stokes_only
```

### 批量模型评估
```bash
./scripts/train_multimodal.sh eval
```

## 🎯 推荐训练策略

### 奥卡姆剃刀策略（从简单到复杂）

1. **第一步：单模态基线**
```bash
# 建立单模态基线性能
./scripts/train_multimodal.sh single
```

2. **第二步：双模态改进**
```bash
# 测试模态组合效果
./scripts/train_multimodal.sh dual
```

3. **第三步：全模态验证**
```bash
# 验证是否需要全部模态
./scripts/train_multimodal.sh full
```

### 快速原型开发
```bash
# 快速测试（5个epoch）
./scripts/train_multimodal.sh quick
```

### 生产环境训练
```bash
# 完整比较实验
./scripts/train_multimodal.sh all
```

## 📋 输出结果

训练完成后，结果保存在：
- `experiments/{experiment_name}/` - 实验结果
  - `checkpoints/` - 模型检查点
  - `logs/` - 训练日志
  - `config.yaml` - 使用的配置
  - `metrics.json` - 训练指标

## 🛠 故障排除

### 内存不足
```bash
# 减小batch size
python scripts/train.py --config configs/full_multimodal.yaml --batch_size 8
```

### GPU内存不足
```bash
# 使用CPU训练
python scripts/train.py --config configs/stokes_only.yaml --gpu -1
```

### 调试模式
```bash
# 快速运行模式
python scripts/train.py --config configs/stokes_only.yaml --fast_dev_run true
```
