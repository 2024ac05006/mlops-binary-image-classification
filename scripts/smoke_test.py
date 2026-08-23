import os
import sys
import time
import requests
import socket
from io import BytesIO
from PIL import Image

# Force IPv4 resolution to prevent Docker Desktop IPv6 'Errno 101'
try:
    host_ip = socket.gethostbyname('host.docker.internal')
except socket.gaierror:
    host_ip = '127.0.0.1'

BASE_URL = os.getenv("SMOKE_TEST_URL", f"http://{host_ip}:30080")

def run_smoke_tests():
    print(f"Running smoke tests against {BASE_URL}...")
    
    # 1. Health Check Endpoint
    try:
        res = requests.get(f"{BASE_URL}/")
        res.raise_for_status()
        print("✅ Health check passed.")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

    # 2. Prediction Endpoint Test
    try:
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        files = {'file': ('smoke_test.jpg', img_bytes, 'image/jpeg')}
        res = requests.post(f"{BASE_URL}/predict", files=files)
        res.raise_for_status()
        print(f"Prediction check passed: {res.json()}")
    except Exception as e:
        print(f"Prediction check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    time.sleep(5) # Allow ingress routing to settle
    run_smoke_tests()