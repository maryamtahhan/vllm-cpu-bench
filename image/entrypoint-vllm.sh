#!/bin/bash

# Base configuration with defaults
MODE=${MODE:-"serve"}
MODEL=${MODEL:-"Qwen/Qwen1.5-MoE-A2.7B-Chat"}
PORT=${PORT:-8000}

# Benchmark configuration with defaults
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
NUM_ROUNDS=${NUM_ROUNDS:-3}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-8192}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}
BENCHMARK_SUMMARY_MODE=${BENCHMARK_SUMMARY_MODE:-"table"}  # Options: table, graph, none

# Additional args passed directly to vLLM
EXTRA_ARGS=${EXTRA_ARGS:-""}

# Log file location
LOG_PATH="/tmp/vllm.log"

# Validate required environment variables
if [[ -z "$MODEL" ]]; then
  echo "Error: MODEL environment variable is not set."
  exit 1
fi

summarize_logs() {
  local logfile="$1"
  echo -e "\n===== Startup Summary ====="
  awk '
    /Loading weights took/ {
      print " Weight Load Time:    " $(NF-1), "seconds"
    }
    /Model loading took/ {
      print " Model Load Time:     " $(NF-1), "seconds"
    }
    /torch\.compile takes/ {
      for (i=1; i<=NF; i++) {
        if ($i == "takes" && $(i+1) ~ /^[0-9.]+$/ && $(i+2) == "s" && $(i+3) == "in" && $(i+4) == "total") {
          print " Torch Compile Time:  " $(i+1), "seconds"
        }
      }
    }
    /Memory profiling takes/ {
      print " Memory Profile Time: " $(NF-1), "seconds"
    }
    /Graph capturing finished/ {
      for (i=1; i<NF; i++) {
        if ($i == "in" && $(i+1) ~ /^[0-9.]+$/ && $(i+2) == "secs,") {
          print " CUDA Graphs Time:    " $(i+1), "seconds"
        }
      }
    }
    /init engine.*took/ {
      print " Total Startup Time:  " $(NF-1), "seconds"
    }
  ' "$logfile"
  echo "============================="
}

watch_for_startup_complete() {
  local logfile="$1"
  while read -r line; do
    echo "$line" >> "$logfile"
    if echo "$line" | grep -q "Application startup complete"; then
      summarize_logs "$logfile"
      break
    fi
  done
}

extract_compile_time() {
  local logfile="$1"
  local label="$2"
  local time

  time=$(grep "torch.compile takes" "$logfile" | tail -1 | awk '{print $(NF-3)}')

  if [[ -n "$time" ]]; then
    echo " Torch Compile Time (${label}): ${time} seconds"
  fi
}


print_benchmark_summary_table() {
  local dir="$1"

  echo -e "\n===== Benchmark Summary (Table) ====="

  if [[ -f "$dir/throughput.json" ]]; then
    echo -e "\n-- Throughput Results --"
    jq -r '
      def fmt(x): if x == null then "n/a" else x|tostring end;
      ["Metric", "Value"],
      ["Elapsed Time (s)", fmt(.elapsed_time)],
      ["Requests", fmt(.num_requests)],
      ["Total Tokens", fmt(.total_num_tokens)],
      ["Requests/sec", fmt(.requests_per_second)],
      ["Tokens/sec", fmt(.tokens_per_second)]
      | @tsv
    ' "$dir/throughput.json" | column -t -s $'\t'
  fi

  if [[ -f "$dir/latency.json" ]]; then
    echo -e "\n-- Latency Results (in seconds) --"
    jq -r '
      def fmt(x): if x == null then "n/a" else x|tostring end;
      ["Metric", "Value"],
      ["Avg", fmt(.avg_latency)],
      ["P10", fmt(.percentiles["10"])],
      ["P25", fmt(.percentiles["25"])],
      ["P50", fmt(.percentiles["50"])],
      ["P75", fmt(.percentiles["75"])],
      ["P90", fmt(.percentiles["90"])],
      ["P99", fmt(.percentiles["99"])]
      | @tsv
    ' "$dir/latency.json" | column -t -s $'\t'
  fi

  echo "======================================"

  echo -e "\n===== Torch Compile Time Summary ====="
  if [[ -f "$dir/throughput.log" ]]; then
    extract_compile_time "$dir/throughput.log" "throughput"
  fi
  if [[ -f "$dir/latency.log" ]]; then
    extract_compile_time "$dir/latency.log" "latency"
  fi
  echo "======================================"

}

print_benchmark_summary_graph() {
  local dir="$1"
  echo -e "\n===== Throughput Summary ====="
  local file="$dir/throughput.json"
  if [[ -f "$file" ]]; then
    local rps=$(jq '.requests_per_second' "$file")
    local tps=$(jq '.tokens_per_second' "$file")
    local tokens=$(jq '.total_num_tokens' "$file")
    local time=$(jq '.elapsed_time' "$file")
    echo "Requests/sec:     $rps"
    echo "Tokens/sec:       $tps"
    echo "Total tokens:     $tokens"
    echo "Elapsed time (s): $time"
  fi

  local file="$dir/latency.json"
  echo -e "\n===== Latency Distribution (Graph) ====="
  echo -e "Metric | Value (s) | Graph"
  echo -e "-------+------------+-------------------------------"

  if [[ -f "$file" ]]; then
    local max_val
    max_val=$(jq '.percentiles["99"]' "$file")

    draw_bar() {
      local label=$1
      local value=$2
      local max=$3
      local width=30

      # Use bc for float-safe calculations
      local scale_factor
      scale_factor=$(echo "scale=6; $width / $max" | bc)

      local bar_len
      bar_len=$(echo "$value * $scale_factor" | bc | awk '{printf "%d", $1}')

      local bar
      bar=$(printf "%${bar_len}s" | tr ' ' '#')

      printf "%-6s | %10.3f | %s\n" "$label" "$value" "$bar"
    }

    draw_bar "Avg" "$(jq '.avg_latency' "$file")" "$max_val"
    draw_bar "P10" "$(jq '.percentiles["10"]' "$file")" "$max_val"
    draw_bar "P25" "$(jq '.percentiles["25"]' "$file")" "$max_val"
    draw_bar "P50" "$(jq '.percentiles["50"]' "$file")" "$max_val"
    draw_bar "P75" "$(jq '.percentiles["75"]' "$file")" "$max_val"
    draw_bar "P90" "$(jq '.percentiles["90"]' "$file")" "$max_val"
    draw_bar "P99" "$(jq '.percentiles["99"]' "$file")" "$max_val"

    echo "========================================="

  echo -e "\n===== Torch Compile Time Summary ====="
  if [[ -f "$dir/throughput.log" ]]; then
    extract_compile_time "$dir/throughput.log" "throughput"
  fi
  if [[ -f "$dir/latency.log" ]]; then
    extract_compile_time "$dir/latency.log" "latency"
  fi
  echo "======================================"
  fi
}


case $MODE in
  "serve")
    echo "Starting vLLM server on port $PORT with model: $MODEL"
    echo "Additional arguments: $EXTRA_ARGS"

    # Kick off the server, stream stdout and stderr, and monitor output live
    (
      # Run summarizer watcher in background
      tail -F "$LOG_PATH" | while read -r line; do
        echo "$line"
        if [[ "$line" == *"Application startup complete."* ]]; then
          summarize_logs "$LOG_PATH"
          break
        fi
      done
    ) &

    # Start vLLM and tee everything to the log file
     vllm serve \
      --model "$MODEL" \
      --port "$PORT" \
      $EXTRA_ARGS > "$LOG_PATH" 2>&1
    ;;

  "benchmark-throughput")
    echo "Running vLLM throughput benchmark with model: $MODEL"
    echo "Additional arguments: $EXTRA_ARGS"

    # Create timestamped directory for this benchmark run
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCHMARK_DIR="/data/benchmarks/throughput_$TIMESTAMP"
    mkdir -p "$BENCHMARK_DIR"
    THROUGHPUT_LOG="$BENCHMARK_DIR/throughput.log"

    echo "Running throughput benchmark..."
    START_TIME=$(date +%s)
    vllm bench throughput \
      --model "$MODEL" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPTS" \
      --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
      --output-json "$BENCHMARK_DIR/throughput.json" \
      $EXTRA_ARGS 2>&1 | tee "$THROUGHPUT_LOG"
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    echo "Throughput benchmark complete - results saved in $BENCHMARK_DIR/throughput.json"

    echo -e "\n===== Total Benchmark Runtime ====="
    echo " Total time: ${TOTAL_TIME} seconds"
    echo "==================================="

    case "$BENCHMARK_SUMMARY_MODE" in
      "table")
        print_benchmark_summary_table "$BENCHMARK_DIR"
        ;;
      "graph")
        print_benchmark_summary_graph "$BENCHMARK_DIR"
        ;;
      "none")
        echo "Benchmark summary display disabled."
        ;;
      *)
        echo "Unknown BENCHMARK_SUMMARY_MODE: $BENCHMARK_SUMMARY_MODE"
        ;;
    esac

    echo "All results have been saved to $BENCHMARK_DIR"
    ;;

  "benchmark-latency")
    echo "Running vLLM latency benchmark with model: $MODEL"
    echo "Additional arguments: $EXTRA_ARGS"

    # Create timestamped directory for this benchmark run
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCHMARK_DIR="/data/benchmarks/latency_$TIMESTAMP"
    mkdir -p "$BENCHMARK_DIR"
    LATENCY_LOG="$BENCHMARK_DIR/latency.log"

    echo "Running latency benchmark..."
    START_TIME=$(date +%s)
    vllm bench latency \
      --model "$MODEL" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --output-json "$BENCHMARK_DIR/latency.json" \
      $EXTRA_ARGS 2>&1 | tee "$LATENCY_LOG"
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))

    echo "Latency benchmark complete - results saved in $BENCHMARK_DIR/latency.json"

    echo -e "\n===== Total Benchmark Runtime ====="
    echo " Total time: ${TOTAL_TIME} seconds"
    echo "==================================="

    case "$BENCHMARK_SUMMARY_MODE" in
      "table")
        print_benchmark_summary_table "$BENCHMARK_DIR"
        ;;
      "graph")
        print_benchmark_summary_graph "$BENCHMARK_DIR"
        ;;
      "none")
        echo "Benchmark summary display disabled."
        ;;
      *)
        echo "Unknown BENCHMARK_SUMMARY_MODE: $BENCHMARK_SUMMARY_MODE"
        ;;
    esac

    echo "All results have been saved to $BENCHMARK_DIR"
    ;;

  "benchmark-serve")
    echo "Starting vLLM server + running online serve benchmark with model: $MODEL"
    echo "Additional arguments: $EXTRA_ARGS"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCHMARK_DIR="/data/benchmarks/serve_$TIMESTAMP"
    mkdir -p "$BENCHMARK_DIR"
    SERVE_LOG="$BENCHMARK_DIR/server.log"
    BENCH_LOG="$BENCHMARK_DIR/serve_benchmark.log"

    # Launch the vLLM server
    echo "Launching vLLM server on port $PORT..."
    vllm serve \
      --model "$MODEL" \
      --port "$PORT" \
      $EXTRA_ARGS > "$SERVE_LOG" 2>&1 &
    SERVER_PID=$!

    # Wait for startup
    echo "Waiting for vLLM server to finish loading..."
    until grep -q "Application startup complete" "$SERVE_LOG" 2>/dev/null; do
      sleep 2
      echo -n "."
    done
    echo -e "\n✅ Server is ready!"

    # Run the benchmark
    echo "Running serve benchmark..."
    START_TIME=$(date +%s)
    vllm bench serve \
      --backend vllm \
      --model "$MODEL" \
      --host localhost \
      --port "$PORT" \
      --endpoint /v1/completions \
      --dataset-name random \
      --random-input-len 512 \
      --random-output-len 256 \
      --max-concurrency "$NUM_CONCURRENT" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate inf \
      --goodput tpot:100 \
      --goodput ttft:2000 \
      --result-dir "$BENCHMARK_DIR" \
      --result-filename serve.json \
      --save-result \
    2>&1 | tee "$BENCH_LOG"

    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))

    # Stop the server
    echo "Stopping vLLM server..."
    kill "$SERVER_PID" >/dev/null 2>&1

    echo -e "\n===== Total Benchmark Runtime ====="
    echo " Total time: ${TOTAL_TIME} seconds"
    echo " Results saved in: $BENCHMARK_DIR"
    echo "==================================="

    # Summarize results
    case "$BENCHMARK_SUMMARY_MODE" in
      "table") print_benchmark_summary_table "$BENCHMARK_DIR" ;;
      "graph") print_benchmark_summary_graph "$BENCHMARK_DIR" ;;
      "none")  echo "Benchmark summary display disabled." ;;
      *)       echo "Unknown BENCHMARK_SUMMARY_MODE: $BENCHMARK_SUMMARY_MODE" ;;
    esac
    ;;
  *)
    echo "Error: Unknown mode: $MODE"
    echo "Supported modes: serve, benchmark-throughput, benchmark-latency, benchmark-serve"
    exit 1
    ;;
esac

