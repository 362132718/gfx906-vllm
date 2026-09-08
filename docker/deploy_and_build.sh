#!/bin/bash
# 部署和构建脚本 - 在服务器上执行
# 使用方法: bash deploy_and_build.sh

set -e

WORK_DIR="/home/x/vllm-work"
GFX906_SUPPLEMENTS="$WORK_DIR/gfx906-supplements"

echo "=========================================="
echo "ROCm 7.2.4 + gfx906 vLLM 构建流程"
echo "=========================================="

# Step 1: 检查工作目录
echo ""
echo "Step 1: 检查工作目录..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Step 2: 拉取 ROCm 7.2.4 基础镜像
echo ""
echo "Step 2: 拉取 ROCm 7.2.4 基础镜像..."

if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "rocm/dev-ubuntu-24.04:7.2.4-complete"; then
    echo "✓ 基础镜像已存在"
else
    echo "尝试拉取基础镜像..."

    # 尝试多个镜像源
    MIRRORS=(
        "rocm/dev-ubuntu-24.04:7.2.4-complete"
        "docker.m.daocloud.io/rocm/dev-ubuntu-24.04:7.2.4-complete"
    )

    PULLED=false
    for mirror in "${MIRRORS[@]}"; do
        echo "尝试: $mirror"
        if timeout 300 docker pull "$mirror" 2>&1; then
            echo "✓ 成功从 $mirror 拉取"
            if [[ "$mirror" != "rocm/dev-ubuntu-24.04:7.2.4-complete" ]]; then
                docker tag "$mirror" "rocm/dev-ubuntu-24.04:7.2.4-complete"
            fi
            PULLED=true
            break
        else
            echo "✗ 失败: $mirror"
        fi
    done

    if [ "$PULLED" = false ]; then
        echo "ERROR: 无法拉取基础镜像，请检查网络连接"
        exit 1
    fi
fi

# Step 3: 准备 gfx906 补充文件
echo ""
echo "Step 3: 准备 gfx906 补充文件..."

if [ -d "/home/x/backup-rocm-tensors" ]; then
    mkdir -p "$GFX906_SUPPLEMENTS"
    cp /home/x/backup-rocm-tensors/* "$GFX906_SUPPLEMENTS/"
    echo "✓ 已复制 gfx906 补充文件 ($(ls "$GFX906_SUPPLEMENTS" | wc -l) 个文件)"
else
    echo "⚠ 警告: 未找到 /home/x/backup-rocm-tensors，将跳过补充文件"
fi

# Step 4: 克隆官方 vLLM 0.28.0（如果还没有）
echo ""
echo "Step 4: 准备 vLLM 源码..."

if [ ! -d "$WORK_DIR/vllm-official" ]; then
    echo "克隆官方 vLLM 0.28.0..."
    git clone https://github.com/vllm-project/vllm.git "$WORK_DIR/vllm-official" || {
        echo "GitHub 克隆失败，尝试其他源..."
        git clone https://gitee.com/mirrors/vllm.git "$WORK_DIR/vllm-official" || {
            echo "ERROR: 无法克隆 vLLM 源码"
            exit 1
        }
    }

    cd "$WORK_DIR/vllm-official"
    git checkout v0.28.0 || {
        echo "WARNING: v0.28.0 tag 不存在，使用当前分支"
    }
    echo "✓ vLLM 源码已准备"
else
    echo "✓ vLLM 源码已存在"
fi

# Step 5: 构建 Docker 镜像
echo ""
echo "Step 5: 构建 Docker 镜像..."
echo "镜像名称: vllm-gfx906-rocm724:base"
echo "预计时间: 4-8 小时"
echo ""

cd "$WORK_DIR"

# 检查 Dockerfile 是否存在
if [ ! -f "$WORK_DIR/Dockerfile.rocm_gfx906" ]; then
    echo "ERROR: Dockerfile.rocm_gfx906 不存在"
    echo "请先上传 Dockerfile 到 $WORK_DIR"
    exit 1
fi

# 开始构建
docker build \
    -f Dockerfile.rocm_gfx906 \
    -t vllm-gfx906-rocm724:base \
    --build-arg BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4-complete \
    "$WORK_DIR"

echo ""
echo "=========================================="
echo "✓ 构建完成！"
echo "=========================================="
echo ""
echo "镜像: vllm-gfx906-rocm724:base"
echo ""
echo "下一步："
echo "1. 创建容器: docker run -it --name vllm-gfx906 --device /dev/kfd --device /dev/dri --group-add video -v $WORK_DIR:/work vllm-gfx906-rocm724:base bash"
echo "2. 安装 vLLM: cd /work/vllm-official && pip install -e ."
echo "3. 测试: python3 -c 'import vllm; print(vllm.__version__)'"
echo ""
