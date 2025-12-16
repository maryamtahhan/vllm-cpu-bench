# Image name and tag
IMAGE_DIR = image
IMAGE_NAME ?= quay.io/mtahhan/vllm
IMAGE_TAG ?= cpu
IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
DOCKERFILE = Containerfile.cpu
BASE_IMAGE ?= quay.io/mtahhan/vllm:cpu-base
# Determine whether to use podman or docker
CONTAINER_TOOL = $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)

# Display help for available commands
help:
	@echo "Available commands:"
	@echo "  make check-deps   - Check for required dependencies"
	@echo "  make build-base   - Build the base image"
	@echo "  make build        - Build the container image"
	@echo "  make push         - Push the container image to the registry"
	@echo "  make clean        - Remove the local container image"
	@echo "  make test         - Test the container image"

# Check for required dependencies
check-deps:
	@echo "Checking for required dependencies..."
	@command -v git >/dev/null 2>&1 || { echo "Error: git is not installed." >&2; exit 1; }
	@command -v $(CONTAINER_TOOL) >/dev/null 2>&1 || { echo "Error: $(CONTAINER_TOOL) is not installed." >&2; exit 1; }
	@echo "All dependencies are installed."

# Build the base image
build-base: check-deps
	@echo "Building the base image using the script..."
	@./scripts/build_vllm_cpu_image.sh

# Build the container image
build: build-base
	@echo "Using $(CONTAINER_TOOL) to build the container image..."
	@$(CONTAINER_TOOL) build --build-arg ENTRYPOINT_PATH=image -t $(IMAGE) -f $(IMAGE_DIR)/$(DOCKERFILE) .

# Build the container image
build-cpu-skip-base:
	@echo "Using $(CONTAINER_TOOL) to build the container image..."
	@$(CONTAINER_TOOL) build --build-arg BASE_IMAGE=$(BASE_IMAGE) --build-arg ENTRYPOINT_PATH=image -t $(IMAGE) -f $(IMAGE_DIR)/$(DOCKERFILE) .

# Build the container image
build-avx-skip-base:
	@echo "Using $(CONTAINER_TOOL) to build the container image..."
	@$(CONTAINER_TOOL) build --build-arg BASE_IMAGE=quay.io/mtahhan/vllm:cpu-base-avx --build-arg ENTRYPOINT_PATH=image -t $(IMAGE_NAME):cpu-avx -f $(IMAGE_DIR)/$(DOCKERFILE) .

# Push the image to the registry
push:
	@echo "Using $(CONTAINER_TOOL) to push the container image..."
	@$(CONTAINER_TOOL) push $(IMAGE)

# Clean up local images
clean:
	@echo "Using $(CONTAINER_TOOL) to remove the local container image..."
	@$(CONTAINER_TOOL) rmi $(IMAGE)

# Test the container image
test:
	@echo "Testing the container image..."
	@$(CONTAINER_TOOL) run --rm $(IMAGE) --help || { echo "Test failed."; exit 1; }
	@echo "Test passed successfully."

.PHONY: help check-deps build-base build push clean test

