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

### Using `bench.sh` for Automated Benchmarking

The `bench.sh` script automates the benchmarking process by running a parameter sweep over various input lengths, output lengths, number of prompts, and concurrency levels.

#### Usage

1. Ensure the script is executable:
   ```bash
   chmod +x bench.sh
   ```

2. Run the script:
   ```bash
   ./bench.sh
   ```

   The script will:
   - Launch containers with the specified configurations.
   - Collect performance metrics such as token throughput and mean latencies.
   - Save the results in a CSV file under the `RESULTS_ROOT` directory.

#### Configuration

You can modify the following parameters in the script:
- `MODEL_PATH`: Path to the model directory.
- `RESULTS_ROOT`: Directory where benchmark results will be saved.
- `INPUT_LENS`, `OUTPUT_LENS`, `NUM_PROMPTS_LIST`, `NUM_CONCURRENT_LIST`: Arrays defining the parameter sweep.

#### Output

The results are saved in a CSV file with the following columns:
- `input_len`: Input sequence length.
- `output_len`: Output sequence length.
- `num_prompts`: Number of prompts.
- `num_concurrent`: Number of concurrent requests.
- `tokens_per_sec`: Total token throughput.
- `output_tokens_per_sec`: Output token throughput.
- `mean_ttft`: Mean time-to-first-token.
- `mean_tpot`: Mean time-per-output-token.

#### Example

```sh
./bench.sh

🚀 [12:33:56] Run 1/30: input=256, output=256, prompts=1, concurrent=1
🕒 Waiting for container 925bffe55c45 to finish...
📊 Extracting results...
✅ [12:36:21] Completed: input=256, output=256, prompts=1, tokens/s=45.30, Output-token/s=22.70

🚀 [12:36:21] Run 2/30: input=256, output=256, prompts=8, concurrent=8
🕒 Waiting for container 8133c0d44af5 to finish...
📊 Extracting results...
✅ [12:38:49] Completed: input=256, output=256, prompts=8, tokens/s=304.33, Output-token/s=152.46

🚀 [12:38:49] Run 3/30: input=256, output=256, prompts=32, concurrent=32
🕒 Waiting for container de5d9889721e to finish...
📊 Extracting results...
✅ [12:41:22] Completed: input=256, output=256, prompts=32, tokens/s=813.97, Output-token/s=407.78

🚀 [12:41:22] Run 4/30: input=256, output=256, prompts=64, concurrent=64
🕒 Waiting for container 9f6189b7a32d to finish...
📊 Extracting results...
✅ [12:44:06] Completed: input=256, output=256, prompts=64, tokens/s=1079.38, Output-token/s=540.74

🚀 [12:44:06] Run 5/30: input=256, output=256, prompts=128, concurrent=128
🕒 Waiting for container caecd1fd14df to finish...
📊 Extracting results...
✅ [12:47:11] Completed: input=256, output=256, prompts=128, tokens/s=1275.71, Output-token/s=639.10

...
```

## Troubleshooting

### Common Issues

1. **Docker/Podman not found**:
   Ensure Docker or Podman is installed and accessible in your `PATH`.

2. **Permission Denied**:
   If you encounter permission issues, try running the commands with `sudo` or ensure your user is part of the `docker` group.

3. **Model Download Fails**:
   Verify your `HF_TOKEN` is valid and has access to the specified model.

