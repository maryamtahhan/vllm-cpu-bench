#!/bin/bash

MODEL_PATH="/models--meta-llama--Llama-3.1-8B-Instruct"
HOST_MODEL_PATH=${HOST_MODEL_PATH:-"/var/models/Llama-3.1-8B-Instruct"}
RESULTS_ROOT=${RESULTS_ROOT:-"/data/benchmarks/sweep_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$RESULTS_ROOT"

# Base configuration with defaults
MODE=${MODE:-"benchmark-serve"} # benchmark-serve or serve or benchmark-throughput or benchmark-latency
MODEL=${MODEL:-"${MODEL_PATH}"}
PORT=${PORT:-8000}

# Benchmark configuration with defaults
INPUT_LEN=${INPUT_LEN:-256}
OUTPUT_LEN=${OUTPUT_LEN:-256}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
NUM_ROUNDS=${NUM_ROUNDS:-3}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-8192}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}
BENCHMARK_SUMMARY_MODE=${BENCHMARK_SUMMARY_MODE:-"table"}  # Options: table, graph, none

# System-specific parallelism tuning (Default Sapphire Rapids dual-socket, 104 physical cores total)
CPUS=${CPUS:-"0-103"}
TP=${TP:-2}  # tensor parallelism = number of NUMA nodes
OMP_NUM_THREADS=${OMP_NUM_THREADS:-26}  # threads per shard (52 physical cores / 2)
VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-"0-25|52-77"}  # one shard per NUMA node
VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-40}  # % of memory for KV cache
SWAP_SPACE=${SWAP_SPACE:-8}  # safe small swap-space to avoid overcommit
GOODPUT_PARAMS=${GOODPUT_PARAMS:-"--goodput tpot:100 --goodput ttft:1000"}  # adjustable goodput settings
EXTRA_ARGS=${EXTRA_ARGS:-"--dtype=bfloat16 --swap-space ${SWAP_SPACE} --no-enable-log-requests --enable_chunked_prefill --distributed-executor-backend mp -tp=${TP}"}

cleanup() {
  echo "🧹 Cleaning up..."
  for c in $(podman ps -aq --filter ancestor=quay.io/mtahhan/vllm:cpu); do
    podman rm -f "$c" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

VLLM_CONTAINER_PARAMS=(
  "--privileged=true"
  "--shm-size=64g"
  "-p" "${PORT}:${PORT}"
  "--cpuset-cpus=${CPUS}"
  "-e" "MODEL=${MODEL_PATH}"
  "-e" "MODE=${MODE}"
  "-e" "VLLM_USE_V1=1"
  "-e" "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  "-e" "VLLM_ENGINE_ITERATION_TIMEOUT_S=600"
  "-e" "VLLM_RPC_TIMEOUT=1000000"
  "-e" "VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE}"
  "-e" "VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND}"
  "-e" "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  "-e" "GOODPUT_PARAMS=${GOODPUT_PARAMS}"
  "-e" "EXTRA_ARGS=${EXTRA_ARGS}"
  "-v" "${HOST_MODEL_PATH}:${MODEL_PATH}"
  "-v" "${RESULTS_ROOT}:${RESULTS_ROOT}"
)

# ---------- SWEEP PARAMETERS ----------
INPUT_LENS=(256 1024)
OUTPUT_LENS=(256 1024 2048)
# NUM_PROMPTS_LIST=(1 8 24 32 36 48 64 96 128)
# NUM_CONCURRENT_LIST=(1 8 24 32 36 48 64 96 128)
# Matching NUM_PROMPTS_LIST and NUM_CONCURRENT_LIST misses important dynamics like:
# startup/queueing effects (small NUM_PROMPTS + high concurrency),
# sustained throughput (large NUM_PROMPTS + high concurrency),
# steady-state vs burst behavior.
NUM_PROMPTS_LIST=(8 32 64 128 256 512)
NUM_CONCURRENT_LIST=(1 4 8 16 32 48 64 96 128)

# ---------- RESULTS CSV ----------
RESULTS_CSV="${RESULTS_ROOT}/benchmark_results.csv"
echo "input_len,output_len,num_prompts,num_concurrent,successful_requests,max_request_concurrency,request_throughput,request_goodput,tokens_per_sec,output_tokens_per_sec,peak_output_tokens_per_sec,peak_concurrent_requests,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,median_itl,p99_itl" > "$RESULTS_CSV"

extract_metrics_from_log() {
  local LOG_FILE=$1

  # Request-level metrics
  SUCCESSFUL_REQUESTS=$(grep "Successful requests" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  MAX_REQUEST_CONCURRENCY=$(grep "Maximum request concurrency" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  REQUEST_THROUGHPUT=$(grep "Request throughput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  REQUEST_GOODPUT=$(grep "Request goodput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  PEAK_OUTPUT_TOKENS=$(grep "Peak output token throughput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  PEAK_CONCURRENT_REQUESTS=$(grep "Peak concurrent requests" "$LOG_FILE" | awk '{print $NF}' | tail -n1)

  # Token-level metrics
  TOKS_PER_SEC=$(grep "Total Token throughput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  OUT_TOKS_PER_SEC=$(grep "Output token throughput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)

  # TTFT / TPOT / ITL
  MEAN_TTFT=$(grep "Mean TTFT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  MEDIAN_TTFT=$(grep "Median TTFT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  P99_TTFT=$(grep "P99 TTFT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  MEAN_TPOT=$(grep "Mean TPOT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  MEDIAN_TPOT=$(grep "Median TPOT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  P99_TPOT=$(grep "P99 TPOT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  MEAN_ITL=$(grep "Mean ITL" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  MEDIAN_ITL=$(grep "Median ITL" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
  P99_ITL=$(grep "P99 ITL" "$LOG_FILE" | awk '{print $NF}' | tail -n1)

  # Apply defaults to avoid empty fields
  SUCCESSFUL_REQUESTS=${SUCCESSFUL_REQUESTS:-0}
  MAX_REQUEST_CONCURRENCY=${MAX_REQUEST_CONCURRENCY:-0}
  REQUEST_THROUGHPUT=${REQUEST_THROUGHPUT:-0}
  REQUEST_GOODPUT=${REQUEST_GOODPUT:-0}
  PEAK_OUTPUT_TOKENS=${PEAK_OUTPUT_TOKENS:-0}
  PEAK_CONCURRENT_REQUESTS=${PEAK_CONCURRENT_REQUESTS:-0}
  TOKS_PER_SEC=${TOKS_PER_SEC:-0}
  OUT_TOKS_PER_SEC=${OUT_TOKS_PER_SEC:-0}
  MEAN_TTFT=${MEAN_TTFT:-0}
  MEDIAN_TTFT=${MEDIAN_TTFT:-0}
  P99_TTFT=${P99_TTFT:-0}
  MEAN_TPOT=${MEAN_TPOT:-0}
  MEDIAN_TPOT=${MEDIAN_TPOT:-0}
  P99_TPOT=${P99_TPOT:-0}
  MEAN_ITL=${MEAN_ITL:-0}
  MEDIAN_ITL=${MEDIAN_ITL:-0}
  P99_ITL=${P99_ITL:-0}
}


# ---------- RUN SWEEP ----------
total_runs=$(( ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * ${#NUM_PROMPTS_LIST[@]} * ${#NUM_CONCURRENT_LIST[@]} ))
run_count=0

for input_len in "${INPUT_LENS[@]}"; do
  for output_len in "${OUTPUT_LENS[@]}"; do
    for num_prompts in "${NUM_PROMPTS_LIST[@]}"; do
      for num_concurrent in "${NUM_CONCURRENT_LIST[@]}"; do
        ((run_count++))
        echo ""
        echo "🚀 [$(date +'%H:%M:%S')] Run ${run_count}/${total_runs}: input=${input_len}, output=${output_len}, prompts=${num_prompts}, concurrent=${num_concurrent}"

        LOG_FILE="${RESULTS_ROOT}/run_${input_len}_${output_len}_${num_prompts}_${num_concurrent}.log"

        # ---------- RUN CONTAINER ----------
        if ! CONTAINER_ID=$(podman run -d "${VLLM_CONTAINER_PARAMS[@]}" \
            -e "INPUT_LEN=${input_len}" \
            -e "OUTPUT_LEN=${output_len}" \
            -e "NUM_PROMPTS=${num_prompts}" \
            -e "NUM_CONCURRENT=${num_concurrent}" \
            quay.io/mtahhan/vllm:cpu 2>>"${LOG_FILE}"); then
            echo "❌ Failed to start container for input=${input_len}, output=${output_len}, prompts=${num_prompts}" | tee -a "$LOG_FILE"
            echo "${input_len},${output_len},${num_prompts},${num_concurrent},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
            continue
        fi

            echo "🕒 Streaming logs for container ${CONTAINER_ID:0:12}..."

            # Stream logs to the file in the background
            podman logs -f "$CONTAINER_ID" > "$LOG_FILE" 2>&1 &
            LOG_PID=$!

            # Wait for the container to finish
            podman wait "$CONTAINER_ID" >/dev/null || true

            # Stop the log stream
            kill "$LOG_PID" >/dev/null 2>&1 || true
            wait "$LOG_PID" 2>/dev/null || true

            # Cleanup
            podman rm "$CONTAINER_ID" >/dev/null 2>&1 || true
            echo "🛑 Container ${CONTAINER_ID:0:12} finished."

        # ---------- EXTRACT RESULTS ----------
        echo "📊 Extracting results..."
        extract_metrics_from_log "$LOG_FILE"

        echo "${input_len},${output_len},${num_prompts},${num_concurrent},${SUCCESSFUL_REQUESTS},${MAX_REQUEST_CONCURRENCY},${REQUEST_THROUGHPUT},${REQUEST_GOODPUT},${TOKS_PER_SEC},${OUT_TOKS_PER_SEC},${PEAK_OUTPUT_TOKENS},${PEAK_CONCURRENT_REQUESTS},${MEAN_TTFT},${MEDIAN_TTFT},${P99_TTFT},${MEAN_TPOT},${MEDIAN_TPOT},${P99_TPOT},${MEAN_ITL},${MEDIAN_ITL},${P99_ITL}" >> "$RESULTS_CSV"
        echo "✅ [$(date +'%H:%M:%S')] Completed: input=${input_len}, output=${output_len}, prompts=${num_prompts}, tokens/s=${TOKS_PER_SEC}, Output-token/s=${OUT_TOKS_PER_SEC}"
        done
    done
  done
done

echo ""
echo "🏁 Sweep complete!"
echo "Results saved to: ${RESULTS_CSV}"