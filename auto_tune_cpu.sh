#!/bin/bash
# ===============================================================
# AUTO-TUNE SCRIPT (CPU version)
# Finds best KV cache size (VLLM_CPU_KVCACHE_SPACE) and
# best performance config for vLLM CPU backend.
# ===============================================================

TAG=$(date +"%Y_%m_%d_%H_%M")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
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

# Clean and create directories
rm -rf "$LOG_FOLDER"
rm -rf "$PROFILE_PATH"
mkdir -p "$LOG_FOLDER" "$PROFILE_PATH"

# Record git hash for reproducibility
cd "$BASE/vllm" 2>/dev/null && {
    current_hash=$(git rev-parse HEAD)
    echo "hash:$current_hash" >> "$RESULT"
    echo "current_hash: $current_hash"
    cd - > /dev/null
} || echo "Not in git repo, skipping hash"

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


echo "====================== CPU AUTO TUNE PARAMETERS ======================"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "BASE=$BASE"
echo "MODEL=$MODEL"
echo "SYSTEM=$SYSTEM"
echo "TP=$TP"
echo "INPUT_LEN=$INPUT_LEN"
echo "OUTPUT_LEN=$OUTPUT_LEN"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "MIN_CACHE_HIT_PCT=$MIN_CACHE_HIT_PCT"
echo "MAX_LATENCY_ALLOWED_MS=$MAX_LATENCY_ALLOWED_MS"
echo "NUM_SEQS_LIST=$NUM_SEQS_LIST"
echo "NUM_BATCHED_TOKENS_LIST=$NUM_BATCHED_TOKENS_LIST"
echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "THREAD_BIND=$THREAD_BIND"
echo "RESULT_FILE=$RESULT"
echo "======================================================================="

TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
RED='\033[0;31m'
if (( TOTAL_LEN > MAX_MODEL_LEN )); then
    echo -e "${RED}FAILED: INPUT_LEN($INPUT_LEN) + OUTPUT_LEN($OUTPUT_LEN) = $TOTAL_LEN, which is > MAX_MODEL_LEN = $MAX_MODEL_LEN.\033[0m" >&2
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
    if command -v fuser >/dev/null 2>&1; then
        fuser -k 8004/tcp || true
    else
        echo "(fuser not found, skipping port cleanup)"
        pkill -f "0.0.0.0:8004" >/dev/null 2>&1 || true
    fi
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
        if curl -fs -o /dev/null -w "%{http_code}" "http://127.0.0.1:8004/health" | grep -q "200"; then
            echo "✅ vLLM server healthy at KV cache ${kv_cache_pct}%"
            return 0
        fi
        sleep 5
    done

    echo "❌ Timeout waiting for vLLM health check"
    tail -n 10 "$vllm_log" || true
    return 1

}

# ===============================================================
# Helper: Run benchmark
# ===============================================================
run_benchmark() {
    local kv_cache_pct=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    echo "max_num_seq: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    local vllm_log="$LOG_FOLDER/vllm_${kv_cache_pct}_${max_num_seqs}_${max_num_batched_tokens}.log"
    echo "vllm_log: $vllm_log"
    echo

    start_server "$kv_cache_pct" "$max_num_seqs" "$max_num_batched_tokens" "$vllm_log" ""
    if [[ $? -ne 0 ]]; then
        echo "Server failed for KV=$kv_cache_pct%, seqs=$max_num_seqs, tokens=$max_num_batched_tokens" | tee -a "$RESULT"
        return 1
    fi
    echo "server started."
    echo

    prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
    adjusted_input_len=$(( INPUT_LEN - prefix_len ))

    echo "run benchmark test..."
    meet_latency_requirement=0
    # Get a basic qps by using request-rate inf
    bm_log="$LOG_FOLDER/bm_${kv_cache_pct}_${max_num_seqs}_${max_num_batched_tokens}_requestrate_inf.txt"
    vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --dataset-name random \
        --random-input-len $adjusted_input_len \
        --random-output-len $OUTPUT_LEN \
        --random-prefix-len $prefix_len \
        --ignore-eos \
        --disable-tqdm \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
        --num-prompts 1000 \
        --port 8004 &> "$bm_log"

    throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
    e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
    goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')

    if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
        meet_latency_requirement=1
        request_rate=inf
    fi

    # If latency requirement not met at infinite rate, iterate to find optimal rate
    if (( ! meet_latency_requirement )); then
        # Start from request-rate as int(throughput) + 1
        request_rate=$((${throughput%.*} + 1))
        while ((request_rate > 0)); do
            # Clear prefix cache
            curl -X POST http://127.0.0.1:8004/reset_prefix_cache
            sleep 5
            bm_log="$LOG_FOLDER/bm_${kv_cache_pct}_${max_num_seqs}_${max_num_batched_tokens}_requestrate_${request_rate}.txt"
            vllm bench serve \
                --backend vllm \
                --model $MODEL \
                --dataset-name random \
                --random-input-len $adjusted_input_len \
                --random-output-len $OUTPUT_LEN \
                --random-prefix-len $prefix_len \
                --ignore-eos \
                --disable-tqdm \
                --request-rate $request_rate \
                --percentile-metrics ttft,tpot,itl,e2el \
                --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
                --num-prompts 100 \
                --port 8004 &> "$bm_log"
            throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
            e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
            goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
            if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
                meet_latency_requirement=1
                break
            fi
            request_rate=$((request_rate-1))
        done
    fi

    # Write the results and update the best result
    if ((meet_latency_requirement)); then
        echo "KV=$kv_cache_pct%, seqs=$max_num_seqs, tokens=$max_num_batched_tokens, request_rate=$request_rate, e2el=$e2el, throughput=$throughput, goodput=$goodput"
        echo "KV=$kv_cache_pct%, seqs=$max_num_seqs, tokens=$max_num_batched_tokens, request_rate=$request_rate, e2el=$e2el, throughput=$throughput, goodput=$goodput" >> "$RESULT"
        if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
            best_throughput=$throughput
            best_goodput=$goodput
            best_request_rate=$request_rate
            best_kvcache_space=$kv_cache_pct
            best_max_num_seqs=$max_num_seqs
            best_num_batched_tokens=$max_num_batched_tokens
        fi
    else
        echo "KV=$kv_cache_pct%, seqs=$max_num_seqs, tokens=$max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}"
        echo "KV=$kv_cache_pct%, seqs=$max_num_seqs, tokens=$max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}" >> "$RESULT"
    fi

    echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput"

    pkill -if "vllm serve" || true
    sleep 10
    if command -v fuser >/dev/null 2>&1; then
        fuser -k 8004/tcp || true
    else
        echo "(fuser not found, skipping port cleanup)"
        pkill -f "0.0.0.0:8004" >/dev/null 2>&1 || true
    fi
    sleep 2
    echo "===================="
    return 0
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
        if command -v fuser >/dev/null 2>&1; then
            fuser -k 8004/tcp || true
        else
            echo "(fuser not found, skipping port cleanup)"
            pkill -f "0.0.0.0:8004" >/dev/null 2>&1 || true
        fi
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

echo "finish permutations"

# ===============================================================
# STEP 3: Profile best configuration
# ===============================================================
if (( $(echo "$best_throughput > 0" | bc -l) )); then
    echo
    echo "Benchmark tuning finished. Now running profiling on the best configuration found..."
    echo "Best config: KV=$best_kvcache_space%, seqs=$best_max_num_seqs, tokens=$best_num_batched_tokens, throughput=$best_throughput"
    echo

    vllm_log="$LOG_FOLDER/vllm_log_BEST_PROFILE.txt"
    bm_log="$LOG_FOLDER/bm_log_BEST_PROFILE.txt"

    # Start server with the best params and profiling ENABLED
    echo "Starting server for profiling..."
    start_server "$best_kvcache_space" "$best_max_num_seqs" "$best_num_batched_tokens" "$vllm_log" "$PROFILE_PATH"

    # Run benchmark with the best params and the --profile flag
    echo "Running benchmark with profiling..."
    prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
    adjusted_input_len=$(( INPUT_LEN - prefix_len ))
    vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --dataset-name random \
        --random-input-len $adjusted_input_len \
        --random-output-len $OUTPUT_LEN \
        --random-prefix-len $prefix_len \
        --ignore-eos \
        --disable-tqdm \
        --request-rate $best_request_rate \
        --percentile-metrics ttft,tpot,itl,e2el \
        --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
        --num-prompts 100 \
        --port 8004 \
        --profile &> "$bm_log"
else
    echo "No configuration met the latency requirements. Skipping final profiling run."
fi
pkill -if "vllm serve" || true
echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput, profile saved in: $PROFILE_PATH"
echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput, profile saved in: $PROFILE_PATH" >> "$RESULT"
