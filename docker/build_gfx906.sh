#!/bin/bash
set -e

echo "=========================================="
echo "Building ROCm 7.2.4 + gfx906 Docker Image"
echo "=========================================="

# Configuration
IMAGE_NAME="vllm-gfx906-rocm724"
TAG="base"
FULL_TAG="${IMAGE_NAME}:${TAG}"

# Check if base image exists
echo ""
echo "Step 1: Checking for ROCm 7.2.4 base image..."
if ! docker images | grep -q "rocm/dev-ubuntu-24.04.*7.2.4"; then
    echo "Base image not found. Attempting to pull from mirrors..."

    # Try multiple mirrors
    MIRRORS=(
        "rocm/dev-ubuntu-24.04:7.2.4-complete"
        "docker.m.daocloud.io/rocm/dev-ubuntu-24.04:7.2.4-complete"
        "registry.cn-hangzhou.aliyuncs.com/rocm/dev-ubuntu-24.04:7.2.4-complete"
    )

    PULLED=false
    for mirror in "${MIRRORS[@]}"; do
        echo "Trying: $mirror"
        if docker pull "$mirror" 2>&1 | tee /tmp/docker_pull.log; then
            echo "Successfully pulled from: $mirror"
            # Tag it for consistency
            docker tag "$mirror" "rocm/dev-ubuntu-24.04:7.2.4-complete"
            PULLED=true
            break
        else
            echo "Failed: $mirror"
        fi
    done

    if [ "$PULLED" = false ]; then
        echo "ERROR: Could not pull ROCm 7.2.4 base image from any mirror."
        echo "Please check network connectivity or use an alternative base image."
        exit 1
    fi
else
    echo "Base image already exists locally."
fi

# Prepare gfx906 supplement files
echo ""
echo "Step 2: Preparing gfx906 supplement files..."
SUPPLEMENTS_DIR="./gfx906-supplements"
mkdir -p "$SUPPLEMENTS_DIR"

# Copy from server backup if available (this needs to be done on the server)
echo "Note: gfx906 supplement files should be copied from /home/x/backup-rocm-tensors/"
echo "This will be done during the build process on the server."

# Build the Docker image
echo ""
echo "Step 3: Building Docker image..."
echo "Image: ${FULL_TAG}"
echo "This will take 4-8 hours depending on your system."
echo ""

docker build \
    -f Dockerfile.rocm_gfx906 \
    -t "${FULL_TAG}" \
    --build-arg BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4-complete \
    .

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "Image: ${FULL_TAG}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a container: docker run -it --name vllm-gfx906 --device /dev/kfd --device /dev/dri --group-add video ${FULL_TAG} bash"
echo "2. Clone official vLLM 0.28.0 inside the container"
echo "3. Install vLLM: pip install -e ."
