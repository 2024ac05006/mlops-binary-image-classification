import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set dataset path
DATA_DIR = os.path.join("data", "processed", "train")

def extract_features():
    records = []
    classes = {"cats": 0, "dogs": 1}
    
    for class_name, label in classes.items():
        folder_path = os.path.join(DATA_DIR, class_name)
        image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(folder_path, "*.png")) + \
                      glob.glob(os.path.join(folder_path, "*.jpeg"))
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w, c = img.shape
            aspect_ratio = h / float(w)
            
            # OpenCV loads as BGR
            mean_blue = np.mean(img[:, :, 0]) / 255.0
            mean_green = np.mean(img[:, :, 1]) / 255.0
            mean_red = np.mean(img[:, :, 2]) / 255.0
            
            # Calculate Sobel edge density (intensity gradients)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_density = np.mean(grad_mag) / 255.0
            
            records.append({
                "aspect_ratio": aspect_ratio,
                "mean_red": mean_red,
                "mean_green": mean_green,
                "mean_blue": mean_blue,
                "edge_density": edge_density,
                "label": label
            })
            
    return pd.DataFrame(records)

def generate_plots():
    df = extract_features()
    if df.empty:
        print("Error: No images found in data/processed/train. Verify data path.")
        return

    # Set visual theme
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # Plot 1: EDA Feature Distribution & Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("EDA Feature Distribution & Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=300)
    plt.close()
    print("Saved: eda_correlation_heatmap.png")

    # -------------------------------------------------------------
    # Plot 2: Feature Histograms and Intensity Gradients
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Feature Histograms and Intensity Gradients (Cats vs Dogs)", fontsize=16)

    features = ["aspect_ratio", "mean_red", "mean_green", "mean_blue", "edge_density"]
    titles = ["Aspect Ratio (H/W)", "Mean Red Intensity", "Mean Green Intensity", "Mean Blue Intensity", "Edge Density (Sobel)"]

    for idx, (feat, title) in enumerate(zip(features, titles)):
        ax = axes[idx // 3, idx % 3]
        sns.histplot(data=df, x=feat, hue="label", kde=True, palette=["#1f77b4", "#ff7f0e"], ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Hide unused 6th subplot
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("feature_histograms_gradients.png", dpi=300)
    plt.close()
    print("Saved: feature_histograms_gradients.png")

if __name__ == "__main__":
    generate_plots()