#!/bin/bash

# This script launches a local vLLM server to serve a specified LLM model.
# It enables fast inference using vLLM's optimized engine, which supports features like paged attention, continuous batching, and memory-efficient serving.

# Check if the model name was passed as an argument
if [ -z "$1" ]; then
  echo "No model name provided."
  exit 1
fi

MODEL_NAME=$1 # The Hugging Face model ID or local path (e.g., meta-llama/Llama-3.1-8B-Instruct)

vllm serve \
  "$MODEL_NAME" \  # Load the specified model for serving
  --gpu-memory-utilization 0.9 \  # Use 90% of GPU memory to balance performance and avoid OOM
  --trust-remote-code \  # Allow loading models with custom code from Hugging Face (required for some instruction-tuned models)
  --max-model-len 8192 \  # Maximum sequence length (context + generation); adjust based on model capability
  --port 8000  # Serve the API on port 8000 (default OpenAI-compatible endpoint: http://localhost:8000/v1)
