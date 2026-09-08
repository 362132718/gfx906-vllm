#!/bin/bash
# 服务器连接检查脚本
# 使用方法: bash check_server.sh

SERVER="192.168.10.11"
USER="x"

echo "检查服务器连接状态..."
echo ""

# 尝试 SSH 连接
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${USER}@${SERVER} "echo 'SSH 连接成功'" 2>/dev/null; then
    echo ""
    echo "✓ 服务器连接正常"
    echo ""

    # 显示服务器状态
    echo "服务器信息："
    ssh ${USER}@${SERVER} "hostname && uptime && docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' | head -10"

    echo ""
    echo "=========================================="
    echo "准备开始构建流程"
    echo "=========================================="
    echo ""
    echo "请执行以下命令上传构建文件："
    echo ""
    echo "scp D:/gfx906-vllm/docker/Dockerfile.rocm_gfx906 x@192.168.10.11:/home/x/vllm-work/"
    echo "scp D:/gfx906-vllm/docker/deploy_and_build.sh x@192.168.10.11:/home/x/vllm-work/"
    echo ""
    echo "然后在服务器上执行："
    echo "ssh x@192.168.10.11"
    echo "cd /home/x/vllm-work"
    echo "bash deploy_and_build.sh"
    echo ""
else
    echo "✗ 服务器连接失败"
    echo ""
    echo "正在重试... (最多 3 次)"
    for i in 1 2 3; do
        echo "重试 $i/3..."
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${USER}@${SERVER} "echo '连接成功'" 2>/dev/null; then
            echo "✓ 第 $i 次重试成功"
            exit 0
        fi
        sleep 5
    done
    echo ""
    echo "ERROR: 无法连接到服务器，请检查网络或服务器状态"
    exit 1
fi
