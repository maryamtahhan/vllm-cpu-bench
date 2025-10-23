#!/bin/bash

MODEL_PATH="/models--meta-llama--Llama-3.1-8B-Instruct"
RESULTS_ROOT="/data/benchmarks/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_ROOT"

cleanup() {
  echo "🧹 Cleaning up..."
  for c in $(podman ps -aq --filter ancestor=quay.io/mtahhan/vllm:cpu); do
    podman rm -f "$c" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

VLLM_PARAMS=(
  "--privileged=true"
  "--shm-size=64g"
  "-p" "8000:8000"
  "--cpuset-cpus=0-103"
  "-e" "MODEL=${MODEL_PATH}"
  "-e" "MODE=benchmark-serve"
  "-e" "VLLM_USE_V1=1"
  "-e" "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  "-e" "VLLM_ENGINE_ITERATION_TIMEOUT_S=600"
  "-e" "VLLM_RPC_TIMEOUT=1000000"
  "-e" "VLLM_CPU_KVCACHE_SPACE=30"
  "-e" "VLLM_CPU_OMP_THREADS_BIND=0-25|52-77"
  "-e" "OMP_NUM_THREADS=26"
  "-e" "GOODPUT_PARAMS=--goodput tpot:100 --goodput ttft:1000"
  "-e" "EXTRA_ARGS=--dtype=bfloat16 --swap-space 8 --no-enable-log-requests --enable_chunked_prefill --distributed-executor-backend mp -tp=2"
  "-v" "/home/mtahhan/models/Llama-3.1-8B-Instruct:${MODEL_PATH}"
  "-v" "${RESULTS_ROOT}:/data/benchmarks"
)

# ---------- SWEEP PARAMETERS ----------
INPUT_LENS=(256 1024)
OUTPUT_LENS=(256 1024 2048)
NUM_PROMPTS_LIST=(1 8 32 64 128)
NUM_CONCURRENT_LIST=(1 8 32 64 128)

# ---------- RESULTS CSV ----------
RESULTS_CSV="${RESULTS_ROOT}/benchmark_results.csv"
echo "input_len,output_len,num_prompts,num_concurrent,tokens_per_sec,output_tokens_per_sec,mean_ttft,mean_tpot" > "$RESULTS_CSV"

# ---------- RUN SWEEP ----------
total_runs=$(( ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * ${#NUM_PROMPTS_LIST[@]} ))
run_count=0

for input_len in "${INPUT_LENS[@]}"; do
  for output_len in "${OUTPUT_LENS[@]}"; do
    for ((i=0; i<${#NUM_PROMPTS_LIST[@]}; i++)); do
      ((run_count++))
      num_prompts=${NUM_PROMPTS_LIST[$i]}
      num_concurrent=${NUM_CONCURRENT_LIST[$i]}
      echo ""
      echo "🚀 [$(date +'%H:%M:%S')] Run ${run_count}/${total_runs}: input=${input_len}, output=${output_len}, prompts=${num_prompts}, concurrent=${num_concurrent}"

      LOG_FILE="${RESULTS_ROOT}/run_${input_len}_${output_len}_${num_prompts}.log"

      # ---------- RUN CONTAINER ----------
      if ! CONTAINER_ID=$(podman run -d "${VLLM_PARAMS[@]}" \
        -e "INPUT_LEN=${input_len}" \
        -e "OUTPUT_LEN=${output_len}" \
        -e "NUM_PROMPTS=${num_prompts}" \
        -e "NUM_CONCURRENT=${num_concurrent}" \
        quay.io/mtahhan/vllm:cpu 2>>"$LOG_FILE"); then
        echo "❌ Failed to start container for input=${input_len}, output=${output_len}, prompts=${num_prompts}" | tee -a "$LOG_FILE"
        echo "${input_len},${output_len},${num_prompts},${num_concurrent},0,0,0,0" >> "$RESULTS_CSV"
        continue
      fi

      echo "🕒 Waiting for container ${CONTAINER_ID:0:12} to finish..."
      podman wait "$CONTAINER_ID" >/dev/null || true

      podman logs "$CONTAINER_ID" > "$LOG_FILE" 2>&1 || true
      podman rm "$CONTAINER_ID" >/dev/null 2>&1 || true

      # ---------- EXTRACT RESULTS ----------
      echo "📊 Extracting results..."
      TOKS_PER_SEC=$(grep "Total Token throughput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
      OUT_TOKS_PER_SEC=$(grep "Output token throughput" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
      MEAN_TTFT=$(grep "Mean TTFT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)
      MEAN_TPOT=$(grep "Mean TPOT" "$LOG_FILE" | awk '{print $NF}' | tail -n1)

      TOKS_PER_SEC=${TOKS_PER_SEC:-0}
      OUT_TOKS_PER_SEC=${OUT_TOKS_PER_SEC:-0}
      MEAN_TTFT=${MEAN_TTFT:-0}
      MEAN_TPOT=${MEAN_TPOT:-0}

      echo "${input_len},${output_len},${num_prompts},${num_concurrent},${TOKS_PER_SEC},${OUT_TOKS_PER_SEC},${MEAN_TTFT},${MEAN_TPOT}" >> "$RESULTS_CSV"
      echo "✅ [$(date +'%H:%M:%S')] Completed: input=${input_len}, output=${output_len}, prompts=${num_prompts}, tokens/s=${TOKS_PER_SEC}, Output-token/s=${OUT_TOKS_PER_SEC}"
    done
  done
done

echo ""
echo "🏁 Sweep complete!"
echo "Results saved to: ${RESULTS_CSV}"