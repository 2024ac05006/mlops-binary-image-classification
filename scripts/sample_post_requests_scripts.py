import requests

url = "http://127.0.0.1:8000/predict"
# Replace with paths to any local test images
image_paths = ["tests/sample_cat.jpg", "tests/sample_dog.jpg"]

for img_path in image_paths * 3:  # Sends 6 requests
    with open(img_path, "rb") as f:
        response = requests.post(url, files={"file": f})
        print(response.json())