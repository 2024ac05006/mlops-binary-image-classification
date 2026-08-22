import json
import csv
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import SimpleCNN, transform

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load model architecture
    model = SimpleCNN().to(device)
    
    # Load weights if available
    model_path = os.path.join("models", "model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print("Warning: Model weights not found. Evaluating with initialized weights.")
        
    model.eval()

    # 2. Target processed test directory from image structure
    data_dir = os.path.join("data", "processed", "test")
    
    actual_labels = []
    predicted_labels = []
    correct = 0
    total = 0
    avg_loss = 0.0

    if os.path.exists(data_dir):
        # Load dataset from data/processed/test/
        test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        class_names = test_dataset.classes  # ['cats', 'dogs']
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                actual_labels.extend(labels.cpu().numpy().tolist())
                predicted_labels.extend(preds.cpu().numpy().tolist())

        avg_loss = round(total_loss / total, 4) if total > 0 else 0.0
        accuracy = round(correct / total, 4) if total > 0 else 0.0
    else:
        raise FileNotFoundError(f"Expected directory '{data_dir}' not found.")

    # 3. Export real metrics JSON
    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        "total_samples": total
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 4. Map class indices to actual folder names ('cats', 'dogs') for DVC plot
    with open("confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "predicted"])
        for act, pred in zip(actual_labels, predicted_labels):
            writer.writerow([class_names[act], class_names[pred]])

    print(f"Evaluation complete on {total} test samples: Accuracy={accuracy}, Loss={avg_loss}")

if __name__ == "__main__":
    evaluate()