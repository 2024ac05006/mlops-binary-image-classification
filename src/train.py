import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Configurations & Hyperparameters ---
PROCESSED_DIR = "data/processed"
MODEL_SAVE_PATH = "models/baseline_cnn.pt"
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Baseline CNN Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 16 x 112 x 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 32 x 56 x 56
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Binary output: [Cat, Dog]
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- 2. Helper Functions for Artifacts ---
def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

# --- 3. Training & Evaluation Pipeline ---
def train_and_evaluate():
    # Data Augmentation & Normalization
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Randomly flips images horizontally so the model learns features regardless of orientation
        transforms.RandomRotation(10), # Slightly rotates images up to 10 degrees to increase variation and prevent overfitting.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # MLflow Setup
    mlflow.set_experiment("Cats_vs_Dogs_Classification")

    with mlflow.start_run(run_name="baseline_cnn_run_1"):
        # Log Parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("model_type", "SimpleCNN")

        train_losses, val_losses = [], []

        for epoch in range(EPOCHS):
            # Training Phase
            model.train()
            running_train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)
            train_losses.append(epoch_train_loss)

            # Validation Phase
            model.eval()
            running_val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_val_loss = running_val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Log Metrics per epoch
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Generate & Log Artifacts (Plots & Saved Model)
        plot_loss_curve(train_losses, val_losses)
        plot_confusion_matrix(all_labels, all_preds, train_dataset.classes)
        
        mlflow.log_artifact("loss_curve.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Save serialized PyTorch model artifact
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        mlflow.log_artifact(MODEL_SAVE_PATH)
        
        # Clean up temporary plot files
        os.remove("loss_curve.png")
        os.remove("confusion_matrix.png")

        print(f"Model and artifacts successfully logged to MLflow and saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()