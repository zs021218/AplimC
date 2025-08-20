#!/bin/bash
# 统计量分类实验执行脚本

echo "=========================================="
echo "藻类统计量特征分类实验"
echo "=========================================="

# 设置路径
PROJECT_DIR="/data3/zs/AplimC"
cd $PROJECT_DIR

# 检查输入文件
HDF5_FILE="data/processed/multimodal_data.h5"
STATS_FILE="data/processed/multimodal_data_statistics.h5"

if [ ! -f "$HDF5_FILE" ]; then
    echo "错误: 找不到原始HDF5文件: $HDF5_FILE"
    exit 1
fi

echo "原始HDF5文件: $HDF5_FILE"
echo "文件大小: $(du -h $HDF5_FILE | cut -f1)"

# 步骤1: 处理统计量（如果还没有）
if [ ! -f "$STATS_FILE" ]; then
    echo -e "\n步骤1: 计算信号统计量..."
    echo "----------------------------------------"
    
    python data/raw/polar_flu_data/signal_statistics_processor.py
    
    if [ $? -eq 0 ]; then
        echo "✓ 统计量计算完成"
        echo "统计量文件: $STATS_FILE"
        if [ -f "$STATS_FILE" ]; then
            echo "文件大小: $(du -h $STATS_FILE | cut -f1)"
        fi
    else
        echo "✗ 统计量计算失败"
        exit 1
    fi
else
    echo "✓ 统计量文件已存在: $STATS_FILE"
fi

# 步骤2: 快速测试
echo -e "\n步骤2: 快速分类测试..."
echo "----------------------------------------"

python scripts/quick_statistics_test.py $STATS_FILE

if [ $? -eq 0 ]; then
    echo "✓ 快速测试完成"
else
    echo "✗ 快速测试失败"
fi

# 步骤3: 完整的机器学习实验
echo -e "\n步骤3: 完整机器学习实验..."
echo "----------------------------------------"

# 创建输出目录
OUTPUT_DIR="experiments/statistics_classification/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "输出目录: $OUTPUT_DIR"

# 运行完整实验
python scripts/train_statistics_classifier.py \
    --data $STATS_FILE \
    --config configs/statistics_classifier.yaml \
    --output $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✓ 完整实验完成"
    echo "结果保存在: $OUTPUT_DIR"
    
    # 显示结果文件
    echo -e "\n生成的文件:"
    ls -la $OUTPUT_DIR/
    
else
    echo "✗ 完整实验失败"
fi

echo -e "\n=========================================="
echo "实验执行完成!"
echo "=========================================="
