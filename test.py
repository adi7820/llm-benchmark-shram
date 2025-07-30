import requests
import pandas as pd

url = "http://localhost:8080/benchmark"
data = {
    "prompt": "Who won the world series in 2020?"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print("Error:", response.status_code)
