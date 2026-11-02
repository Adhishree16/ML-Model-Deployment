
import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Input data in JSON format
data = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Send POST request
response = requests.post(
    url,
    json=data,
    timeout=10
)

# Display response
print("Status Code:", response.status_code)
print("Prediction Result:", response.json())