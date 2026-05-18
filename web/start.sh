#!/bin/bash
cd "$(dirname "$0")"

# Set MODEL_PATH to the directory of your merged model checkpoint
MODEL_PATH="${MODEL_PATH:-/path/to/your/merged_checkpoint}"
PORT=8000

echo "Starting vLLM server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 131072 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID (waiting for server ready...)"

for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    sleep 2
done

echo "Starting GenSyntax web server on port 8101..."
export VLLM_API_URL="http://localhost:${PORT}"
export MODEL_NAME="$MODEL_PATH"
python app.py
