import os
import shutil
import kagglehub

# Configuration
DATASET_NAME = "salader/dogs-vs-cats"  # Replace if using a different Kaggle URL
RAW_DIR = "data/raw"

def download_and_extract_data():
    """Authenticates via kagglehub using the access_token, downloads, and moves data."""
    
    print("Authenticating with Kaggle and downloading dataset...")
    # kagglehub automatically detects ~/.kaggle/access_token
    # It downloads the dataset to a local cache and returns the path to that cache
    cache_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset cached successfully at: {cache_path}")
    
    print(f"Copying dataset to project directory: {RAW_DIR}...")
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Copy the contents from the kagglehub cache to our local data/raw directory
    shutil.copytree(cache_path, RAW_DIR, dirs_exist_ok=True)
    
    print("Download and setup complete!")

if __name__ == "__main__":
    download_and_extract_data()