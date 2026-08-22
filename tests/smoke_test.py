import os
import sys
import time
import requests
from io import BytesIO
from PIL import Image

# Use host.docker.internal so the Jenkins container can reach the host's K8s cluster
BASE_URL = os.getenv("SMOKE_TEST_URL", "http://host.docker.internal:30080")

def run_smoke_tests():
    print(f"Running smoke tests against {BASE_URL}...")
    
    # 1. Health Check
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print("✅ Health check passed.")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)
        
    # 2. Prediction Check
    try:
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('dummy.jpg', img_bytes, 'image/jpeg')}
        res = requests.post(f"{BASE_URL}/predict", files=files)
        res.raise_for_status()
        print(f"✅ Prediction check passed: {res.json()}")
    except Exception as e:
        print(f"❌ Prediction check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    time.sleep(5) # Allow pod routing to initialize
    run_smoke_tests()