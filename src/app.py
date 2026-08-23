import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0")

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

# --- 2. Endpoints ---

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
    """Prediction endpoint returning class label and probabilities."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        return {
            "label": CLASSES[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                CLASSES[i]: round(probabilities[i].item(), 4) for i in range(len(CLASSES))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")