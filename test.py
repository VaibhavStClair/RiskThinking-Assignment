import json
import requests

url = "http://127.0.0.1:8000/volume_prediction"

input_data = {
    'moving_average': 200.9,
    'rolling_median': 23.14
}

input_json = json.dumps(input_data)
response = requests.post(url, data = input_json)
print(response.text)