import io
import uuid
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn
import time
import json
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import csv
import os
from prometheus_client import Counter

# Define counter metric with prediction_result label
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['prediction_result']
)

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0")
PRED_LOG_PATH = "data/predictions.csv"

# Setup structured logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_service")

# 1. Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

# --- 1. Model Definition (Must match trained architecture) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load model weights on startup
MODEL_PATH = "models/baseline_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")

# Class labels matching ImageFolder order (0: cats, 1: dogs)
CLASSES = ["cat", "dog"]

# Preprocessing transforms (Must match validation pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def log_prediction(request_id: str, prediction: str, confidence: float):
    """Appends prediction details to a CSV file for Evidently AI analysis."""
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(PRED_LOG_PATH)

    with open(PRED_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["request_id", "prediction", "confidence"])
        writer.writerow([request_id, prediction, confidence])
        
# --- 2. Endpoints ---
# 2. Structured Request/Response Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    
    log_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": duration_ms,
    }
    logger.info(json.dumps(log_data))
    return response

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Cats and Dogs ML API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model artifact failed to load")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint returning class label and probabilities, and logging output."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image"
        )

    # Generate a unique request ID for tracking
    request_id = str(uuid.uuid4())[:8]

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        predicted_label = CLASSES[predicted_idx.item()]
        conf_val = round(confidence.item(), 4)

        # Log prediction to CSV for model monitoring (Step 3)
        log_prediction(request_id, predicted_label, conf_val)

        PREDICTION_COUNTER.labels(prediction_result=predicted_label).inc()

        return {
            "request_id": request_id,
            "label": predicted_label,
            "confidence": conf_val,
            "probabilities": {
                CLASSES[i]: round(probabilities[i].item(), 4)
                for i in range(len(CLASSES))
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        )