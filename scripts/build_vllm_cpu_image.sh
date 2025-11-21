#!/bin/bash

LOG_FILE="/tmp/build_vllm_cpu_image.log"

# Function to detect Docker or Podman
detect_container_tool() {
    if command -v docker &> /dev/null; then
        echo "docker"
    elif command -v podman &> /dev/null; then
        echo "podman"
    else
        echo "Error: Neither Docker nor Podman is installed." | tee -a "$LOG_FILE" >&2
        exit 1
    fi
}

# Clone the vLLM repository
if [ ! -d "vllm_source" ]; then
    echo "Cloning the vLLM repository..." | tee -a "$LOG_FILE"
    git clone https://github.com/vllm-project/vllm.git vllm_source 2>>"$LOG_FILE"
else
    echo "vLLM repository already exists. Skipping clone step." | tee -a "$LOG_FILE"
fi

cd vllm_source || { echo "Error: Failed to change directory to vllm_source." | tee -a "$LOG_FILE" >&2; exit 1; }

# Detect container tool
CONTAINER_TOOL=$(detect_container_tool)
echo "Using $CONTAINER_TOOL to build the image..." | tee -a "$LOG_FILE"

# Apply podman-build.patch if using podman
if [ "$CONTAINER_TOOL" = "podman" ]; then
    if [ -f "../podman-build.patch" ]; then
        echo "Applying podman-build.patch for Podman compatibility..." | tee -a "$LOG_FILE"
        git apply ../podman-build.patch 2>>"$LOG_FILE" || {
            echo "Warning: Failed to apply podman-build.patch. It may already be applied." | tee -a "$LOG_FILE"
        }
    else
        echo "Warning: podman-build.patch not found at ../podman-build.patch" | tee -a "$LOG_FILE"
    fi
fi

echo "Build started at: $(date)" | tee -a "$LOG_FILE"

# Build the CPU image
$CONTAINER_TOOL build -f docker/Dockerfile.cpu \
    --build-arg VLLM_CPU_AVX512BF16=false \
    --build-arg VLLM_CPU_AVX512VNNI=false \
    --build-arg VLLM_CPU_DISABLE_AVX512=false \
    --tag localhost/vllm-cpu-env:latest \
    --target vllm-openai . 2>>"$LOG_FILE" | tee -a "$LOG_FILE"

echo "Build completed at: $(date)" | tee -a "$LOG_FILE"

# Manually tag the last built image if the tag was dropped
LATEST_ID=$($CONTAINER_TOOL images -q --filter dangling=true | head -n 1)
if [ -n "$LATEST_ID" ]; then
    echo "Tagging intermediate image $LATEST_ID as localhost/vllm-cpu-env:latest" | tee -a "$LOG_FILE"
    $CONTAINER_TOOL tag "$LATEST_ID" localhost/vllm-cpu-env:latest
fi

