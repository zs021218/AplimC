#!/bin/bash
# 多模态分类器训练命令集合
# 遵循奥卡姆剃刀原理：从简单到复杂

# 确保在正确的环境中
source /data3/zs/miniconda3/bin/activate pytorch_env

# 设置项目根目录
PROJECT_ROOT="/data3/zs/AplimC"
cd $PROJECT_ROOT

echo "=== 多模态分类器训练命令 ==="
echo "选择训练模式："
echo "1. 单模态训练"
echo "2. 双模态训练" 
echo "3. 全模态训练"
echo "4. 全部训练（比较实验）"
echo

# ===================================
# 1. 单模态训练命令
# ===================================

single_modal_training() {
    echo "=== 单模态训练 ==="
    
    echo "1.1 训练 Stokes 单模态分类器"
    python scripts/train.py \
        --config configs/stokes_only.yaml \
        --gpu 0 \
        --experiment_name "stokes_only_$(date +%Y%m%d_%H%M%S)"
    
    echo "1.2 训练荧光信号单模态分类器"
    python scripts/train.py \
        --config configs/fluorescence_only.yaml \
        --gpu 0 \
        --experiment_name "fluorescence_only_$(date +%Y%m%d_%H%M%S)"
    
    echo "1.3 训练图像单模态分类器"
    python scripts/train.py \
        --config configs/images_only.yaml \
        --gpu 0 \
        --experiment_name "images_only_$(date +%Y%m%d_%H%M%S)"
}

# ===================================
# 2. 双模态训练命令
# ===================================

dual_modal_training() {
    echo "=== 双模态训练 ==="
    
    echo "2.1 训练信号双模态分类器（Stokes + 荧光）"
    python scripts/train.py \
        --config configs/signals_dual.yaml \
        --gpu 0 \
        --experiment_name "signals_dual_$(date +%Y%m%d_%H%M%S)"
    
    echo "2.2 训练 Stokes + 图像双模态分类器"
    python scripts/train.py \
        --config configs/stokes_images_dual.yaml \
        --gpu 0 \
        --experiment_name "stokes_images_dual_$(date +%Y%m%d_%H%M%S)"
    
    echo "2.3 训练荧光 + 图像双模态分类器"
    python scripts/train.py \
        --config configs/fluorescence_images_dual.yaml \
        --gpu 0 \
        --experiment_name "fluorescence_images_dual_$(date +%Y%m%d_%H%M%S)"
}

# ===================================
# 3. 全模态训练命令
# ===================================

full_modal_training() {
    echo "=== 全模态训练 ==="
    
    echo "3.1 训练全模态分类器（Stokes + 荧光 + 图像）"
    python scripts/train.py \
        --config configs/full_multimodal.yaml \
        --gpu 0 \
        --experiment_name "full_multimodal_$(date +%Y%m%d_%H%M%S)"
}

# ===================================
# 4. 比较实验（全部训练）
# ===================================

comparison_experiments() {
    echo "=== 比较实验：训练所有配置 ==="
    
    # 创建实验批次目录
    BATCH_DIR="experiments/batch_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BATCH_DIR
    
    # 所有配置文件
    configs=(
        "stokes_only"
        "fluorescence_only" 
        "images_only"
        "signals_dual"
        "stokes_images_dual"
        "fluorescence_images_dual"
        "full_multimodal"
    )
    
    echo "开始批量训练，结果保存到: $BATCH_DIR"
    
    for config in "${configs[@]}"; do
        echo "正在训练: $config"
        python scripts/train.py \
            --config "configs/${config}.yaml" \
            --gpu 0 \
            --experiment_name "${config}_batch" \
            --save_dir "$BATCH_DIR/${config}" \
            --log_file "$BATCH_DIR/${config}_train.log"
        
        echo "完成训练: $config"
        echo "---"
    done
    
    echo "批量训练完成！"
    echo "结果目录: $BATCH_DIR"
}

# ===================================
# 快速训练命令（调试用）
# ===================================

quick_training() {
    echo "=== 快速训练（调试模式）==="
    
    echo "快速训练最简单的配置..."
    python scripts/train.py \
        --config configs/stokes_only.yaml \
        --gpu 0 \
        --experiment_name "quick_test" \
        --max_epochs 5 \
        --batch_size 8 \
        --fast_dev_run true
}

# ===================================
# 评估命令
# ===================================

evaluate_models() {
    echo "=== 模型评估 ==="
    
    # 评估单模态模型
    echo "评估单模态模型..."
    python scripts/evaluate.py \
        --config configs/stokes_only.yaml \
        --checkpoint experiments/stokes_only/best_model.pth \
        --output_dir results/stokes_only
    
    # 评估双模态模型
    echo "评估双模态模型..."
    python scripts/evaluate.py \
        --config configs/signals_dual.yaml \
        --checkpoint experiments/signals_dual/best_model.pth \
        --output_dir results/signals_dual
    
    # 评估全模态模型
    echo "评估全模态模型..."
    python scripts/evaluate.py \
        --config configs/full_multimodal.yaml \
        --checkpoint experiments/full_multimodal/best_model.pth \
        --output_dir results/full_multimodal
}

# ===================================
# 主菜单
# ===================================

case "$1" in
    "single")
        single_modal_training
        ;;
    "dual")
        dual_modal_training
        ;;
    "full")
        full_modal_training
        ;;
    "all")
        comparison_experiments
        ;;
    "quick")
        quick_training
        ;;
    "eval")
        evaluate_models
        ;;
    *)
        echo "用法: $0 {single|dual|full|all|quick|eval}"
        echo
        echo "命令说明："
        echo "  single  - 训练所有单模态模型"
        echo "  dual    - 训练所有双模态模型"
        echo "  full    - 训练全模态模型"
        echo "  all     - 训练所有模型（比较实验）"
        echo "  quick   - 快速训练（调试用）"
        echo "  eval    - 评估训练好的模型"
        echo
        echo "示例："
        echo "  $0 single   # 训练单模态"
        echo "  $0 all      # 完整比较实验"
        exit 1
        ;;
esac
