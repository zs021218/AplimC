#!/bin/bash
# 通用模型评估脚本
# 用法: ./scripts/evaluate_model.sh [选项]

set -e

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 显示帮助信息
show_help() {
    cat << EOF
模型评估脚本

用法: $0 [选项]

选项:
    -c, --checkpoint PATH    模型检查点路径 (必需)
    -d, --data PATH         数据文件路径 (默认: data/processed/multimodal_data.h5)
    -s, --split SPLIT       数据分割 (test/val/train, 默认: test)
    -o, --output DIR        输出目录 (默认: evaluation_results/auto_TIMESTAMP)
    -b, --batch-size SIZE   批次大小 (默认: 64)
    -g, --gpu ID            GPU ID (默认: 0)
    -a, --all-splits        评估所有数据分割 (test, val, train)
    -h, --help              显示此帮助信息

示例:
    # 评估特定模型的测试集
    $0 -c experiments/runs/my_exp/checkpoints/best_model.pth
    
    # 评估所有数据分割
    $0 -c experiments/runs/my_exp/checkpoints/best_model.pth -a
    
    # 指定GPU和批次大小
    $0 -c path/to/model.pth -g 1 -b 32

EOF
}

# 默认参数
CHECKPOINT=""
DATA_FILE="data/processed/multimodal_data.h5"
SPLIT="test"
OUTPUT_DIR=""
BATCH_SIZE=64
GPU_ID=0
ALL_SPLITS=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -d|--data)
            DATA_FILE="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -a|--all-splits)
            ALL_SPLITS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}错误: 必须指定模型检查点路径 (-c)${NC}"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== 模型评估脚本 ===${NC}"
echo -e "项目目录: ${PROJECT_ROOT}"

# 检查文件存在性
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}错误: 模型检查点不存在: $CHECKPOINT${NC}"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}错误: 数据文件不存在: $DATA_FILE${NC}"
    exit 1
fi

# 设置输出目录
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MODEL_NAME=$(basename "$CHECKPOINT" .pth)
    OUTPUT_DIR="evaluation_results/${MODEL_NAME}_$TIMESTAMP"
fi

mkdir -p "$OUTPUT_DIR"

echo -e "模型检查点: ${GREEN}$CHECKPOINT${NC}"
echo -e "数据文件: ${GREEN}$DATA_FILE${NC}"
echo -e "输出目录: ${GREEN}$OUTPUT_DIR${NC}"

# 函数: 运行评估
run_evaluation() {
    local split=$1
    local output_subdir="$OUTPUT_DIR/$split"
    
    echo -e "${BLUE}=== 评估 $split 数据集 ===${NC}"
    
    python scripts/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data "$DATA_FILE" \
        --split "$split" \
        --output-dir "$output_subdir" \
        --batch-size "$BATCH_SIZE" \
        --gpu "$GPU_ID"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $split 数据集评估完成${NC}"
        
        # 提取准确率
        report_file="$output_subdir/classification_report.txt"
        if [ -f "$report_file" ]; then
            accuracy=$(grep "accuracy" "$report_file" | awk '{print $2}')
            echo -e "$split 准确率: ${GREEN}$accuracy${NC}"
        fi
    else
        echo -e "${RED}✗ $split 数据集评估失败${NC}"
        return 1
    fi
}

# 执行评估
if [ "$ALL_SPLITS" = true ]; then
    echo -e "${YELLOW}评估所有数据分割...${NC}"
    for split in test val train; do
        run_evaluation "$split"
        echo
    done
else
    run_evaluation "$SPLIT"
fi

# 生成摘要
SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.txt"
cat > "$SUMMARY_FILE" << EOF
模型评估摘要
============

评估时间: $(date)
模型检查点: $CHECKPOINT
数据文件: $DATA_FILE
批次大小: $BATCH_SIZE
GPU ID: $GPU_ID

EOF

if [ "$ALL_SPLITS" = true ]; then
    cat >> "$SUMMARY_FILE" << EOF
评估的数据分割: test, val, train

结果目录:
- 测试集: $OUTPUT_DIR/test/
- 验证集: $OUTPUT_DIR/val/
- 训练集: $OUTPUT_DIR/train/
EOF
else
    cat >> "$SUMMARY_FILE" << EOF
评估的数据分割: $SPLIT
结果目录: $OUTPUT_DIR/$SPLIT/
EOF
fi

echo -e "${GREEN}✓ 评估摘要已保存: $SUMMARY_FILE${NC}"
echo -e "${GREEN}=== 评估完成! ===${NC}"
echo -e "详细结果请查看: ${YELLOW}$OUTPUT_DIR${NC}"
