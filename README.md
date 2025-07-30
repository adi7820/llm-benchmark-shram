# llm-benchmark-shram

## Problem Statement
### Local LLM setup
We need to precisely determine whether it is beneficial running a local model. We want to create a benchmarking code to determine what open-sourced models can the machine run and with what approximate latency (tpm) - Llama 3.1 8B, Qwen 2.5, Gemma 2B. This benchmarking code preferable should be efficient and not take too much computing resources and should give as accurate answers as possible.

## Details and Results

Based on the problem statement, I first set up the environment using uv, a Rust-based Python project management tool. It's become a part of my daily workflow due to its speed and simplicity. I used it to create a virtual environment for this project. Anyone who wants to replicate the same setup can simply run uv sync to install all dependencies from the uv.lock file.

Once the environment was ready, I focused on hosting the model locally and benchmarking it. For this project, I used an NVIDIA L4 GPU with 24 GB of VRAM.

To serve the models, I used vLLM, a high-throughput and low-latency LLM inference engine. vLLM leverages PagedAttention for efficient KV cache management. I created a shell script named local-deploy-llm.sh to deploy models like LLaMA 3.1 8B, Qwen 2.5, and Gemma 2B. The script is generalized to support different model families, and comments inside the script explain each vLLM parameter. A step-by-step guide for running the script is provided below.

Before proceeding further, it's important to clarify the GPU capacity constraints. Models up to 11B parameters can fit into the 24GB VRAM, calculated as:
 ```bash
 11B * (16-bit / 8) = ~22 GB
 ```

 This calculation only accounts for model weights. The remaining VRAM is used for KV cache during request processing. For tight fits like 11B models, quantization becomes essential. However, in this project, we use models in the 2B, 3B, 7B, and 8B range, which comfortably fit within the GPU’s capacity even at bfloat16 (16-bit) precision.