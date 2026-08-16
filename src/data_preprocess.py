import os
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_TRAIN_DIR = "data/raw/dogs-vs-cats"
PROCESSED_DIR = "data/processed"
IMG_SIZE = (224, 224)

def process_and_split():
    # Retrieve files from raw/train
    files = [f for f in os.listdir(RAW_TRAIN_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    filepaths = []
    labels = []
    for f in files:
        filepaths.append(os.path.join(RAW_TRAIN_DIR, f))
        if f.startswith("cat"):
            labels.append("cats")
        elif f.startswith("dog"):
            labels.append("dogs")

    # Split: 80% Train, 20% Temp (Val + Test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    # Split Temp into 10% Validation, 10% Test
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test)
    }

    # Process and save images into structured subdirectories
    for split_name, (paths, split_labels) in splits.items():
        for path, label in zip(paths, split_labels):
            save_dir = os.path.join(PROCESSED_DIR, split_name, label)
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(os.path.join(save_dir, os.path.basename(path)))
            except Exception as e:
                print(f"Skipping {path}: {e}")

if __name__ == "__main__":
    process_and_split()
    print("Dataset successfully processed into 80/10/10 splits.")