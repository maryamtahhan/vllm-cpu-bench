#!/bin/bash
# ===============================================================
# AUTO-TUNE SCRIPT (CPU version)
# Finds best KV cache size (VLLM_CPU_KVCACHE_SPACE) and
# best performance config for vLLM CPU backend.
# ===============================================================

TAG=$(date +"%Y_%m_%d_%H_%M")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE=${BASE:-"$SCRIPT_DIR/../../.."}
MODEL=${MODEL:-"/models--meta-llama--Llama-3.1-8B-Instruct"}
SYSTEM=${SYSTEM:-"CPU"}
TP=${TP:-2}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MIN_CACHE_HIT_PCT=${MIN_CACHE_HIT_PCT:-0}
MAX_LATENCY_ALLOWED_MS=${MAX_LATENCY_ALLOWED_MS:-10000}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-26}
THREAD_BIND=${THREAD_BIND:-"0-25|52-77"}
EXTRA_ARGS=${EXTRA_ARGS:-"--dtype=bfloat16 --swap-space 8 --enable_chunked_prefill --distributed-executor-backend mp"}

NUM_SEQS_LIST=${NUM_SEQS_LIST:-"64 128 256"}
NUM_BATCHED_TOKENS_LIST=${NUM_BATCHED_TOKENS_LIST:-"512 1024 2048 4096"}

LOG_FOLDER="$BASE/auto-benchmark/$TAG"
RESULT="$LOG_FOLDER/result.txt"
PROFILE_PATH="$LOG_FOLDER/profile"

mkdir -p "$LOG_FOLDER" "$PROFILE_PATH"

# ===============================================================
# Validate THREAD_BIND against visible CPUs
# ===============================================================
VISIBLE_CPUS=$(cat /sys/fs/cgroup/cpuset.cpus | tr -d ' \n')
echo "Detected visible CPUs: $VISIBLE_CPUS"

expand_range() {
  local range=$1
  if [[ "$range" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  else
    echo "$range"
  fi
}

# Flatten the visible CPUs into a simple list of numbers
VISIBLE_LIST=()
for seg in ${VISIBLE_CPUS//,/ }; do
  VISIBLE_LIST+=($(expand_range "$seg"))
done

# Now check each thread binding segment
IFS='|' read -ra SEGMENTS <<< "$THREAD_BIND"
for seg in "${SEGMENTS[@]}"; do
  for cpu in $(expand_range "$seg"); do
    if ! printf '%s\n' "${VISIBLE_LIST[@]}" | grep -q -x "$cpu"; then
      echo "❌ THREAD_BIND includes CPU $cpu, which is NOT visible in container cpuset: $VISIBLE_CPUS"
      echo "   Please fix THREAD_BIND or container --cpuset-cpus"
      exit 1
    fi
  done
done
echo "✅ THREAD_BIND validation passed: $THREAD_BIND"


echo "====================== CPU AUTO TUNE ======================"
echo "MODEL=$MODEL"
echo "SYSTEM=$SYSTEM"
echo "TP=$TP"
echo "INPUT_LEN=$INPUT_LEN"
echo "OUTPUT_LEN=$OUTPUT_LEN"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "MAX_LATENCY_ALLOWED_MS=$MAX_LATENCY_ALLOWED_MS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "THREAD_BIND=$THREAD_BIND"
echo "RESULT_FILE=$RESULT"
echo "============================================================"

TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
if (( TOTAL_LEN > MAX_MODEL_LEN )); then
    echo "FAILED: INPUT_LEN($INPUT_LEN)+OUTPUT_LEN($OUTPUT_LEN)=$TOTAL_LEN > MAX_MODEL_LEN($MAX_MODEL_LEN)"
    exit 1
fi

best_throughput=0
best_goodput=0
best_request_rate=0
best_max_num_seqs=0
best_num_batched_tokens=0
best_kvcache_space=0

# ===============================================================
# Helper: Start server with given KV cache %
# ===============================================================
start_server() {
    local kv_cache_pct=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    local vllm_log=$4
    local profile_dir=$5

    pkill -if "vllm serve" || true
    sleep 5
    fuser -k 8004/tcp || true
    sleep 2

    export VLLM_CPU_KVCACHE_SPACE=$kv_cache_pct
    export OMP_NUM_THREADS=$OMP_NUM_THREADS
    export VLLM_CPU_OMP_THREADS_BIND=$THREAD_BIND

    local args=(
        "$MODEL"
        "--port" "8004"
        "--max-num-seqs" "$max_num_seqs"
        "--max-num-batched-tokens" "$max_num_batched_tokens"
        "--tensor-parallel-size" "$TP"
        "--disable-log-requests"
        "--enable-prefix-caching"
        "--max-model-len" "$MAX_MODEL_LEN"
    )

    if [[ -n "$profile_dir" ]]; then
        VLLM_SERVER_DEV_MODE=1 VLLM_TORCH_PROFILER_DIR=$profile_dir \
            vllm serve "${args[@]}" $EXTRA_ARGS > "$vllm_log" 2>&1 &
    else
        VLLM_SERVER_DEV_MODE=1 vllm serve "${args[@]}" $EXTRA_ARGS > "$vllm_log" 2>&1 &
    fi

    local server_pid=$!

    # Wait for up to 5 minutes for readiness
    for i in {1..60}; do
        kill -0 "$server_pid" 2>/dev/null || return 1
        if curl -s "http://0.0.0.0:8004/health" | grep -q "ok"; then
            return 0
        fi
        sleep 5
    done

    if [[ $? -ne 0 ]]; then
        echo "Server failed to start. Last 10 lines of log:"
        tail -n 10 "$vllm_log" || true
        return 1
    fi

    return 1
}

# ===============================================================
# Helper: Run benchmark
# ===============================================================
run_benchmark() {
    local kv_cache_pct=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    local vllm_log="$LOG_FOLDER/vllm_${kv_cache_pct}_${max_num_seqs}_${max_num_batched_tokens}.log"
    local bm_log="$LOG_FOLDER/bm_${kv_cache_pct}_${max_num_seqs}_${max_num_batched_tokens}.log"

    start_server "$kv_cache_pct" "$max_num_seqs" "$max_num_batched_tokens" "$vllm_log" ""
    if [[ $? -ne 0 ]]; then
        echo "Server failed for KV=$kv_cache_pct%" | tee -a "$RESULT"
        return 1
    fi

    prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
    adjusted_input_len=$(( INPUT_LEN - prefix_len ))

    vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --dataset-name random \
        --random-input-len $adjusted_input_len \
        --random-output-len $OUTPUT_LEN \
        --ignore-eos \
        --disable-tqdm \
        --request-rate inf \
        --num-prompts 500 \
        --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
        --port 8004 &> "$bm_log"

    throughput=$(grep "Request throughput" "$bm_log" | awk '{print $NF}')
    e2el=$(grep "P99 E2EL" "$bm_log" | awk '{print $NF}')
    goodput=$(grep "Request goodput" "$bm_log" | awk '{print $NF}')

    throughput=${throughput:-0}
    e2el=${e2el:-999999}
    goodput=${goodput:-0}

    echo "KV=$kv_cache_pct%, seqs=$max_num_seqs, tokens=$max_num_batched_tokens, throughput=$throughput, goodput=$goodput, e2el=$e2el" | tee -a "$RESULT"

    if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
        if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
            best_throughput=$throughput
            best_goodput=$goodput
            best_kvcache_space=$kv_cache_pct
            best_max_num_seqs=$max_num_seqs
            best_num_batched_tokens=$max_num_batched_tokens
        fi
    fi

    pkill -if "vllm serve" || true
    sleep 5
    fuser -k 8004/tcp || true
    sleep 2
}

# ===============================================================
# STEP 1: Binary search for max KV cache space that fits memory
# ===============================================================
echo "Searching for max usable VLLM_CPU_KVCACHE_SPACE..."
low=5
high=90
found_kvcache=0

while (( low <= high )); do
    mid=$(( (low + high) / 2 ))
    echo "Trying KV cache: ${mid}%"
    start_server "$mid" "${NUM_SEQS_LIST##* }" "${NUM_BATCHED_TOKENS_LIST##* }" "$LOG_FOLDER/vllm_kv_${mid}.log" ""
    if [[ $? -eq 0 ]]; then
        found_kvcache=$mid
        low=$(( mid + 2 ))
        pkill -if "vllm serve" || true
        sleep 3
        fuser -k 8004/tcp || true
        sleep 2
    else
        high=$(( mid - 2 ))
    fi
done

if (( found_kvcache == 0 )); then
    echo "❌ Could not start vLLM with KV cache ≥ 5%. Exiting."
    exit 1
fi
echo "✅ Using VLLM_CPU_KVCACHE_SPACE=$found_kvcache%"
echo "best_kvcache_space=$found_kvcache" >> "$RESULT"

# ===============================================================
# STEP 2: Run tuning sweeps for seqs/tokens combos
# ===============================================================
for num_seqs in $NUM_SEQS_LIST; do
    for num_batched_tokens in $NUM_BATCHED_TOKENS_LIST; do
        run_benchmark "$found_kvcache" "$num_seqs" "$num_batched_tokens"
    done
done

echo "Tuning complete."
echo "Best: KV=$best_kvcache_space%, seqs=$best_max_num_seqs, tokens=$best_num_batched_tokens, throughput=$best_throughput" | tee -a "$RESULT"

# ===============================================================
# STEP 3: Profile best configuration
# ===============================================================
if (( $(echo "$best_throughput > 0" | bc -l) )); then
    echo "Profiling best configuration..."
    start_server "$best_kvcache_space" "$best_max_num_seqs" "$best_num_batched_tokens" "$LOG_FOLDER/vllm_best_profile.log" "$PROFILE_PATH"
    vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --ignore-eos \
        --disable-tqdm \
        --num-prompts 100 \
        --port 8004 \
        --profile &> "$LOG_FOLDER/bm_best_profile.log"
    pkill -if "vllm serve" || true
else
    echo "No valid config met latency constraints."
fi

echo ""
echo "================= FINAL RESULTS ================="
echo "Best KV cache: $best_kvcache_space%"
echo "Best seqs: $best_max_num_seqs"
echo "Best batched tokens: $best_num_batched_tokens"
echo "Best throughput: $best_throughput"
echo "Results in: $RESULT"
echo "Profile in: $PROFILE_PATH"
