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
   make build-base # Will clone vLLM and build the base cpu image
   make build      # Will build the image that can be used for simple serving or benchmarking
   ```

   This will:
   - Build the base image using the `scripts/build_vllm_cpu_image.sh` script.
   - Build the final container image tagged as `quay.io/mtahhan/vllm:cpu`.

   > Note: TODO update the image name to something more generic


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

The model will be saved in /var/models/Llama-3.1-8B-Instruct.

## Running the container in serving mode

```bash
podman run --rm \
  --privileged=true \
  --shm-size=16g \
  -p 8000:8000 \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  -e MODE=serve\
  -e EXTRA_ARGS="--dtype=bfloat16 --disable-sliding-window" \
  -e  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_CPU_OMP_THREADS_BIND=auto \
  -e MODEL=/models--meta-llama--Llama-3.1-8B-Instruct \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 \
  -e VLLM_RPC_TIMEOUT=1000000 \
  -v /var/models/Llama-3.1-8B-Instruct:/models--meta-llama--Llama-3.1-8B-Instruct \
  quay.io/mtahhan/vllm:cpu
```

Test the endpoint with curl:


```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/models--meta-llama--Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Analyze the main changes for Dijkstra algorithm"}
  ],
  "temperature": 0.7,
  "max_tokens": 50
}'
```

> Note: you will need to stop the serving pod with `podman stop`

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
  -v /var/models/Llama-3.1-8B-Instruct:/models--meta-llama--Llama-3.1-8B-Instruct \
  quay.io/mtahhan/vllm:cpu
```

### Using `bench.sh` for a Benchmark sweep

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

The `bench.sh` script provides several configurable parameters. These can be set as environment variables before running the script or modified directly in the script:

- **Base Configuration**:
  - `MODE`: Mode of operation. Options: `benchmark-serve`, `serve`, `benchmark-throughput`, `benchmark-latency`. Default: `benchmark-serve`.
  - `MODEL`: Model name or Path to the model directory. Default: `${MODEL_PATH}`.
  - `PORT`: Port for the container. Default: `8000`.

  > NOTE: if using the MOOEL name directly ensure you also configure `HF_TOKEN`
    as an env var for the container.

- **Benchmark Configuration**:
  - `INPUT_LEN`: Input sequence length. Default: `256`.
  - `OUTPUT_LEN`: Output sequence length. Default: `256`.
  - `NUM_PROMPTS`: Number of prompts per request. Default: `1000`.
  - `NUM_ROUNDS`: Number of benchmark rounds. Default: `3`.
  - `MAX_BATCH_TOKENS`: Maximum tokens per batch. Default: `8192`.
  - `NUM_CONCURRENT`: Number of concurrent requests. Default: `8`.
  - `BENCHMARK_SUMMARY_MODE`: Format of benchmark summary. Options: `table`, `graph`, `none`. Default: `table`.

- **System-Specific Parallelism Tuning**:
  - `CPUS`: CPU cores to use. Default: `0-103`.
  - `TP`: Tensor parallelism (number of NUMA nodes). Default: `2`.
  - `OMP_NUM_THREADS`: Threads per shard. Default: `26`.
  - `VLLM_CPU_OMP_THREADS_BIND`: CPU thread binding for shards. Default: `0-25|52-77`.
  - `VLLM_CPU_KVCACHE_SPACE`: Percentage of memory allocated for KV cache. Default: `30`.
  - `SWAP_SPACE`: Swap space in GB. Default: `8`.
  - `GOODPUT_PARAMS`: Goodput settings. Default: `--goodput tpot:100 --goodput ttft:1000`.
  - `EXTRA_ARGS`: Additional arguments for the container. Default: `--dtype=bfloat16 --swap-space ${SWAP_SPACE} --no-enable-log-requests --enable_chunked_prefill --distributed-executor-backend mp -tp=${TP}`.

- **Sweep Parameters**:
  - `INPUT_LENS`: Array of input sequence lengths. Default: `(256 1024)`.
  - `OUTPUT_LENS`: Array of output sequence lengths. Default: `(256 1024 2048)`.
  - `NUM_PROMPTS_LIST`: Array of prompt counts. Default: `(1 8 32 64 128)`.
  - `NUM_CONCURRENT_LIST`: Array of concurrency levels. Default: `(1 8 32 64 128)`.

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

