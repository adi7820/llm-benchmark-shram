#!/bin/bash

if [ -z "$1" ]; then
  echo "No model name provided."
  exit 1
fi

MODEL_NAME=$1

vllm serve \
  "$MODEL_NAME" \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-model-len 8192 \
  --port 8000
