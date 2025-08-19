#!/bin/bash
# TensorBoard 启动脚本

# 激活环境
source /data3/zs/miniconda3/bin/activate pytorch_env

# 设置项目目录
PROJECT_ROOT="/data3/zs/AplimC"
LOG_DIR="$PROJECT_ROOT/experiments"

echo "🚀 启动 TensorBoard..."
echo "📁 日志目录: $LOG_DIR"
echo "🌐 访问地址: http://localhost:6006"
echo "⏹️  停止服务: Ctrl+C"
echo

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 日志目录不存在: $LOG_DIR"
    echo "请先运行训练生成日志文件"
    exit 1
fi

# 启动 TensorBoard
tensorboard --logdir="$LOG_DIR" --port=6006 --host=0.0.0.0
