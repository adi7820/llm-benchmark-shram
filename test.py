import requests
import pandas as pd
from prompts import SMALL, MEDIUM, LARGE

url = "http://localhost:8080/benchmark"
data = {
    "prompt": LARGE
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print("Error:", response.status_code)
