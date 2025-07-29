
from openai import OpenAI
import time
from transformers import AutoTokenizer

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
]

def count_input_tokens(client, model, messages):
        try:
            usage = client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=1,
                temperature=0,
                stream=False,
            ).usage
            return usage.prompt_tokens
        except Exception as e:
            print(f"Token count estimation failed: {e}")
            return 0
def count_input_tokens_tokenizer(tokenizer, messages):
    full_text = "".join([m["content"] for m in messages])
    return len(tokenizer.encode(full_text))

def count_output_tokens(tokenizer, text):
    return len(tokenizer.encode(text))

def main():
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    print("Input Tokens: ", count_input_tokens(client, model, messages))
    print("Input Tokens (Tokenizer): ", count_input_tokens_tokenizer(tokenizer, messages))
    # Chat Completion API
    start_time = time.time()
    first_token_time = None
    token_times = []
    tokens_received = 0
    full_output = ""
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.6,
        max_tokens=100,
        stream=True,
    )
    
    # for chunk in chat_completion:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")
    
    for chunk in chat_completion:
        now = time.time()
        if first_token_time is None:
            first_token_time = now
        delta = chunk.choices[0].delta
        content = delta.content or ""
        if content.strip():
            tokens_received += 1
            token_times.append(now)
            full_output += content
            # print(content, end="", flush=True)

    end_time = time.time()
    
    print("Output token count: ", count_output_tokens(tokenizer, full_output))
    
    print("TTFT: ", first_token_time - start_time)
    inter_token_latencies = [t2 - t1 for t1, t2 in zip(token_times, token_times[1:])]
    avg_inter_token_latency = sum(inter_token_latencies) / len(inter_token_latencies) 
    
    print("Average Inter-Token Latency: ", avg_inter_token_latency)
    total_duration = end_time - start_time
    print("Total Latency: ", total_duration)
    tpm = (tokens_received / total_duration) * 60
    
    print("Tokens Per Minute (TPM): ", tpm)

if __name__ == "__main__":
    main()