#!/bin/bash
# 模型评估脚本 - 评估标签平滑训练的最佳模型
# 用法: ./scripts/evaluate_best_model.sh

set -e  # 遇到错误时退出

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== 模型评估脚本 ===${NC}"
echo -e "项目目录: ${PROJECT_ROOT}"

# 配置参数
DATA_FILE="data/processed/multimodal_data.h5"
GPU_ID=1
BATCH_SIZE=64

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}错误: 数据文件不存在: $DATA_FILE${NC}"
    exit 1
fi

# 查找最新的标签平滑实验
echo -e "${YELLOW}正在查找最新的标签平滑实验...${NC}"
LATEST_EXP=$(find experiments/runs -name "label_smoothing_*" -type d | sort | tail -1)

if [ -z "$LATEST_EXP" ]; then
    echo -e "${RED}错误: 未找到标签平滑实验目录${NC}"
    exit 1
fi

echo -e "找到实验目录: ${GREEN}$LATEST_EXP${NC}"

# 查找最佳模型检查点
BEST_MODEL="$LATEST_EXP/checkpoints/best_model.pth"
if [ ! -f "$BEST_MODEL" ]; then
    echo -e "${RED}错误: 未找到最佳模型检查点: $BEST_MODEL${NC}"
    exit 1
fi

echo -e "最佳模型路径: ${GREEN}$BEST_MODEL${NC}"

# 创建评估结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="evaluation_results/label_smoothing_$TIMESTAMP"
mkdir -p "$EVAL_DIR"

echo -e "${YELLOW}评估结果将保存到: $EVAL_DIR${NC}"

# 函数: 运行评估
run_evaluation() {
    local split=$1
    local output_dir="$EVAL_DIR/$split"
    
    echo -e "${BLUE}=== 评估 $split 数据集 ===${NC}"
    
    python scripts/evaluate.py \
        --checkpoint "$BEST_MODEL" \
        --data "$DATA_FILE" \
        --split "$split" \
        --output-dir "$output_dir" \
        --batch-size $BATCH_SIZE \
        --gpu $GPU_ID
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $split 数据集评估完成${NC}"
        echo -e "结果保存在: $output_dir"
    else
        echo -e "${RED}✗ $split 数据集评估失败${NC}"
        return 1
    fi
}

# 评估测试集
echo -e "${YELLOW}开始评估...${NC}"
run_evaluation "test"

# 评估验证集
run_evaluation "val"

# 评估训练集（检查过拟合）
echo -e "${YELLOW}评估训练集以检查过拟合情况...${NC}"
run_evaluation "train"

# 生成评估摘要
echo -e "${BLUE}=== 生成评估摘要 ===${NC}"
SUMMARY_FILE="$EVAL_DIR/evaluation_summary.txt"

cat > "$SUMMARY_FILE" << EOF
模型评估摘要
============

评估时间: $(date)
模型路径: $BEST_MODEL
实验目录: $LATEST_EXP

数据集分割评估结果:
- 测试集: $EVAL_DIR/test/
- 验证集: $EVAL_DIR/val/
- 训练集: $EVAL_DIR/train/

评估内容:
- 分类报告 (classification_report.txt)
- 混淆矩阵 (confusion_matrix.png)
- 分类指标图 (classification_metrics.png)
- 原始结果 (evaluation_results.npz)

建议查看顺序:
1. 测试集准确率 (真实性能)
2. 对比训练集和测试集结果 (过拟合检查)
3. 分析混淆矩阵 (类别混淆情况)
4. 查看各类别F1分数 (类别性能差异)
EOF

echo -e "${GREEN}✓ 评估摘要已保存: $SUMMARY_FILE${NC}"

# 快速显示准确率
echo -e "${BLUE}=== 快速结果预览 ===${NC}"
for split in test val train; do
    report_file="$EVAL_DIR/$split/classification_report.txt"
    if [ -f "$report_file" ]; then
        accuracy=$(grep "accuracy" "$report_file" | awk '{print $2}')
        echo -e "$split 准确率: ${GREEN}$accuracy${NC}"
    fi
done

echo -e "${GREEN}=== 评估完成! ===${NC}"
echo -e "详细结果请查看: ${YELLOW}$EVAL_DIR${NC}"
echo -e "快速查看测试集混淆矩阵: ${BLUE}$EVAL_DIR/test/confusion_matrix.png${NC}"

# 询问是否打开结果目录
read -p "是否打开结果目录? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v nautilus >/dev/null 2>&1; then
        nautilus "$EVAL_DIR" &
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$EVAL_DIR" &
    else
        echo -e "${YELLOW}请手动打开目录: $EVAL_DIR${NC}"
    fi
fi
