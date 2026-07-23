#!/bin/bash
cd "$(dirname "$0")"

# Set MODEL_PATH to the directory of your merged model checkpoint
MODEL_PATH="${MODEL_PATH:-/path/to/your/merged_checkpoint}"
VLLM_PORT="${VLLM_PORT:-8000}"
WEB_PORT="${WEB_PORT:-8101}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

if [ "$MODEL_PATH" = "/path/to/your/merged_checkpoint" ]; then
    echo "ERROR: set MODEL_PATH to a local checkpoint or Hugging Face model identifier."
    exit 2
fi

echo "Starting vLLM server on port ${VLLM_PORT}..."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
vllm serve "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID (waiting for server ready...)"

for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming ready."
        wait "$VLLM_PID"
        exit $?
    fi
    if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM did not become ready within 240 seconds."
        kill "$VLLM_PID" 2>/dev/null
        exit 1
    fi
    sleep 2
done

echo "Starting GenSyntax web server on port ${WEB_PORT}..."
export VLLM_API_URL="http://localhost:${VLLM_PORT}"
export MODEL_NAME="$MODEL_PATH"
python -m uvicorn app:app --host 0.0.0.0 --port "$WEB_PORT"
