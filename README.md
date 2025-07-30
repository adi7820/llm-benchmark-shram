# llm-benchmark-shram

## Problem Statement
### Local LLM setup
We need to precisely determine whether it is beneficial running a local model. We want to create a benchmarking code to determine what open-sourced models can the machine run and with what approximate latency (tpm) - Llama 3.1 8B, Qwen 2.5, Gemma 2B. This benchmarking code preferable should be efficient and not take too much computing resources and should give as accurate answers as possible.

## Details and Results

Based on the problem statement, I first set up the environment using **uv**, a Rust-based Python project management tool. It's become a part of my daily workflow due to its speed and simplicity. I used it to create a virtual environment for this project. Anyone who wants to replicate the same setup can simply run **uv sync** to install all dependencies from the uv.lock file.

Once the environment was ready, I focused on hosting the model locally and benchmarking it. For this project, I used an **NVIDIA L4** GPU with 24 GB of VRAM.

To serve the models, I used **vLLM**, a high-throughput and low-latency LLM inference engine. vLLM leverages PagedAttention for efficient KV cache management. I created a shell script named **local-deploy-llm.sh** to deploy models like **LLaMA 3.1 8B, Qwen 2.5, and Gemma 2B**. The script is generalized to support different model families, and comments inside the script explain each vLLM parameter. A step-by-step guide for running the script is provided below.

Before proceeding further, it's important to clarify the GPU capacity constraints. Models up to 11B parameters can fit into the 24GB VRAM, calculated as:
 ```bash
 11B * (16-bit / 8) = ~22 GB
 ```

 This calculation only accounts for model weights. The remaining VRAM is used for KV cache during request processing. For tight fits like 11B models, quantization becomes essential. However, in this problem solution, we use models in the 2B, 7B, and 8B range, which comfortably fit within the GPU’s capacity even at bfloat16 (16-bit) precision.

 After deploying the model, the next step is benchmarking. I created a script **(main.py)** that evaluates the following key metrics:
 - Input token count
 - Output token count
 - Tokens per minute (TPM)
 - Tokens per second (TPS)
 - Time to first token (TTFT)
 - Inter-token latency

 The script is explained in detail in main.py.

 To simplify usage, I exposed the benchmarking functionality via a FastAPI service **(app.py)**. This allows users to send a prompt and receive benchmarking data from the locally hosted model via an API. The API implementation closely mirrors the logic in main.py.

 Lastly, I created **test.py** to interact with the FastAPI endpoint. It allows you to send test prompts and gives the benchmarking results.

 ### Steps to create the solution:
 
 1. Install uv
 ```bash
 pip install uv
 ```

 2. Initialize Project
 ```bash
 uv init --python=python3.12
 ```

 3. Create a virtual enviorment
 ```bash
 uv venv
 ```

 4. Make sure to list your dependencies

 5. Make the LLM Deployment Script Executable
 ```bash
 chmod +x local-deploy-llm.sh
 ```

 6. Run the Script to Deploy the Model
 ```bash
 ./local-deploy-llm.sh google/gemma-2-2b-it
 ```
 This will deploy the specified model using vLLM.

 7. Start the Benchamrking FastAPI Server
 ```bash
 uv run app.py
 ```

 8. Run the Test Script
 ```bash
 uv run test.py
 ```

 ### Results

 To evaluate the models, I'm using three types of prompts—SMALL, MEDIUM, and LARGE—based on their input length. These prompts are defined in the prompts.py file. 
 Below are the results for the models:
 **meta-llama/Llama-3.1-8B-Instruct**
 
 **SMALL Prompt:**
 ```bash
 {
  "Input Token": 8, "Output Token": 9,  "TPM": 950.99, "TPS": 15.85, "TTFT": 0.07, "ITL": 0.06, "latency": 0.57
 }
```
**MEDIUM Prompt:**
```bash
{
  "Input Token": 29, "Output Token": 97, "TPM": 942.71, "TPS": 15.71, "TTFT": 0.08, "ITL": 0.06, "latency": 6.17
}
```

**LARGE Prompt:**
```bash
{
  "Input Token": 80, "Output Token": 101, "TPM": 975.69, "TPS": 16.26, "TTFT": 0.08, "ITL": 0.06, "latency": 6.21
}
```

**google/gemma-2-2b-it**

**SMALL Prompt:**
```bash
{
  "Input Token": 8, "Output Token": 10, "TPM": 2036.38, "TPS": 33.94, "TTFT": 0.03, "ITL": 0.02, "latency": 0.29
}
```

**MEDIUM Prompt**
```bash
{
  "Input Token": 30, "Output Token": 93, "TPM": 2349.07, "TPS": 39.15, "TTFT": 0.04, "ITL": 0.03, "latency": 2.38
}
```

**LARGE Prompt**
```bash
{
  "Input Token": 83, "Output Token": 97, "TPM": 2438.95, "TPS": 40.65, "TTFT": 0.04, "ITL": 0.02, "latency": 2.39
}
```

**Qwen/Qwen2.5-7B-Instruct**

**SMALL Prompt:**
```bash
{
    'Input Token': 7, 'Output Token': 8, 'TPM': 879.0, 'TPS': 14.65, 'TTFT': 0.07, 'ITL': 0.06, 'latency': 0.55
}
```

**MEDIUM Prompt**
```bash
{
    'Input Token': 30, 'Output Token': 97, 'TPM': 978.11, 'TPS': 16.3, 'TTFT': 0.08, 'ITL': 0.06, 'latency': 5.95
}
```
**LARGE Prompt**
```bash
{
    'Input Token': 79, 'Output Token': 98, 'TPM': 988.33, 'TPS': 16.47, 'TTFT': 0.08, 'ITL': 0.06, 'latency': 5.95
}
```

Based on the results, **google/gemma-2-2b-it** consistently demonstrates the fastest latency and highest throughput (TPM and TPS) across all prompt sizes, making it the most efficient among the three. meta-llama/Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct show comparable performance, but Qwen has slightly higher latency and lower throughput, especially on smaller prompts. Overall, Gemma-2B is the best choice for low-latency, high-speed local inference.