# vLLM CPU Benchmarks

This repository provides tools to build and benchmark vLLM CPU-based container images.

## Prerequisites

Ensure the following tools are installed:
- Git
- Docker or Podman
- Python 3.8+ with `pip`

> **Note:** It is recommended to use a Python virtual environment to avoid dependency conflicts:
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> ```

## Building the Images

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/vllm-cpu-bench.git
   cd vllm-cpu-bench
   ```

2. Build the base image and container image:
   ```bash
   make build
   ```

   This will:
   - Build the base image using the `scripts/build_vllm_cpu_image.sh` script.
   - Build the final container image tagged as `quay.io/mtahhan/vllm:cpu`.

3. Push the image to a container registry (optional):
   ```bash
   make push
   ```

## Downloading Models from Hugging Face

Use the provided Python script to download models for offline or vLLM use.

Export your hugging face token as an env var

```
export HF_TOKEN=<hf_token>
```

> **Note:** If you are downloading a private model, ensure your `HF_TOKEN` has the necessary permissions.

Install the required Python package:

```bash
pip install huggingface_hub
```

```bash
python scripts/download_model.py meta-llama/Llama-3.1-8B-Instruct
```

The model will be saved in /tmp/models/Llama-3.1-8B-Instruct.

## Running Benchmarks

Run benchmarks using the built container image. Replace podman with docker if needed.

```bash
podman run --rm \
  --privileged=true \
  --shm-size=16g \
  -p 8000:8000 \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  -e MODE=benchmark-throughput \
  -e EXTRA_ARGS="--dtype=bfloat16 --disable-sliding-window" \
  -e  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_CPU_OMP_THREADS_BIND=auto \
  -e MODEL=/models--meta-llama--Llama-3.1-8B-Instruct \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 \
  -e VLLM_RPC_TIMEOUT=1000000 \
  -e INPUT_LEN=256 \
  -e OUTPUT_LEN=256 \
  -v /tmp/models/Llama-3.1-8B-Instruct:/models--meta-llama--Llama-3.1-8B-Instruct \
  quay.io/mtahhan/vllm:cpu
```

## Troubleshooting

### Common Issues

1. **Docker/Podman not found**:
   Ensure Docker or Podman is installed and accessible in your `PATH`.

2. **Permission Denied**:
   If you encounter permission issues, try running the commands with `sudo` or ensure your user is part of the `docker` group.

3. **Model Download Fails**:
   Verify your `HF_TOKEN` is valid and has access to the specified model.

