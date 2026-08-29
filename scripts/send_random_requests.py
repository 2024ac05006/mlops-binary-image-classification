import os
import glob
import random
import time
import requests

# API Endpoint URL (Update port if using K8s service or port-forwarding)
API_URL = "http://localhost:8000/predict"

# Path to test data
TEST_DIR = os.path.join("data", "processed", "test")

def get_all_test_images():
    # Gather image paths from both cats and dogs subfolders
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(TEST_DIR, "**", ext), recursive=True))
        
    return image_paths

def send_random_requests(num_requests=30):
    image_paths = get_all_test_images()
    
    if not image_paths:
        print(f"Error: No images found in '{TEST_DIR}'. Verify folder paths.")
        return

    print(f"Found {len(image_paths)} images in test directory.")
    print(f"Sending {num_requests} random prediction requests to {API_URL}...\n")

    successful_requests = 0

    for i in range(1, num_requests + 1):
        # Pick a random image file
        selected_img_path = random.choice(image_paths)
        actual_label = os.path.basename(os.path.dirname(selected_img_path)) # 'cats' or 'dogs'
        img_name = os.path.basename(selected_img_path)

        try:
            with open(selected_img_path, "rb") as img_file:
                files = {"file": (img_name, img_file, "image/jpeg")}
                start_time = time.time()
                response = requests.post(API_URL, files=files)
                latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                successful_requests += 1
                res_json = response.json()
                prediction = res_json.get("prediction", res_json)
                print(f"[{i:02d}/{num_requests}] Actual: {actual_label:<4} | Pred: {str(prediction):<4} | Status: 200 | Latency: {latency:.2f}ms")
            else:
                print(f"[{i:02d}/{num_requests}] Failed with status code: {response.status_code}")

        except Exception as e:
            print(f"[{i:02d}/{num_requests}] Connection error: {e}")

        # Small sleep delay to space out request time-series metrics for Prometheus/Grafana
        time.sleep(random.uniform(0.2, 0.8))

    print(f"\nCompleted! {successful_requests}/{num_requests} requests executed successfully.")

if __name__ == "__main__":
    send_random_requests(num_requests=30)