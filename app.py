
from openai import OpenAI
import time
from transformers import AutoTokenizer
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

models = client.models.list()
model = models.data[0].id

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

def count_input_tokens_tokenizer(tokenizer, messages):
    full_text = "".join([m["content"] for m in messages])
    return len(tokenizer.encode(full_text))

def count_output_tokens(tokenizer, text):
    return len(tokenizer.encode(text))

class PromptRequest(BaseModel):
    prompt: str

@app.post("/benchmark")
async def benchmarking(prompt_request: PromptRequest):
    prompt = prompt_request.prompt

    messages = [
        {"role": "user", "content": prompt},
    ]

    
    # print("Tokenizer loaded in: ", time.time() - tt, " seconds")
    input_token_count = count_input_tokens_tokenizer(tokenizer, messages)
    first_token_time = None
    token_times = []
    full_output = ""
    start_time = time.time()
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.6,
        max_tokens=100,
        stream=True,
    )
    

    for chunk in chat_completion:
        now = time.time()
        if first_token_time is None:
            first_token_time = now
        delta = chunk.choices[0].delta
        content = delta.content or ""
        if content.strip():
            token_times.append(now)
            full_output += content
            # print(content, end="", flush=True)

    end_time = time.time()
    
    op_tokens = count_output_tokens(tokenizer, full_output)
    ttft = first_token_time - start_time
    inter_token_latencies = [t2 - t1 for t1, t2 in zip(token_times, token_times[1:])]
    avg_inter_token_latency = sum(inter_token_latencies) / len(inter_token_latencies) 
    
    total_duration = end_time - start_time

    tps = op_tokens / total_duration
    tpm = (op_tokens / total_duration) * 60

    
    result_dict = {
        "Input Token": input_token_count,
        "Output Token": op_tokens,
        "TPM": round(tpm, 2),
        "TPS": round(tps, 2),
        "TTFT": round(ttft, 2),
        "ITL": round(avg_inter_token_latency, 2),
        "latency": round(total_duration, 2),

    }
    
    return result_dict

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)