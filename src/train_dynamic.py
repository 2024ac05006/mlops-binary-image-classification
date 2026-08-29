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

PROCESSED_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def plot_loss_curve(train_losses, val_losses, epochs, run_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves ({run_name})")
    plt.legend()
    plt.grid(True)
    filename = f"loss_curve_{run_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_confusion_matrix(y_true, y_pred, class_names, run_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix ({run_name})")
    filename = f"confusion_matrix_{run_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def train_and_evaluate(epochs=5, batch_size=32, learning_rate=0.001, run_name="baseline_cnn_run_1"):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Cats_vs_Dogs_Classification")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("model_type", "SimpleCNN")

        train_losses, val_losses = [], []

        for epoch in range(epochs):
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

            print(f"[{run_name}] Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Generate & Log Artifacts
        loss_file = plot_loss_curve(train_losses, val_losses, epochs, run_name)
        cm_file = plot_confusion_matrix(all_labels, all_preds, train_dataset.classes, run_name)
        
        mlflow.log_artifact(loss_file)
        mlflow.log_artifact(cm_file)

        os.makedirs("models", exist_ok=True)
        model_save_path = f"models/{run_name}.pt"
        torch.save(model.state_dict(), model_save_path)
        mlflow.log_artifact(model_save_path)
        
        # Cleanup temp plots
        if os.path.exists(loss_file): os.remove(loss_file)
        if os.path.exists(cm_file): os.remove(cm_file)

        print(f"Successfully logged artifacts for {run_name}")

if __name__ == "__main__":
    train_and_evaluate()