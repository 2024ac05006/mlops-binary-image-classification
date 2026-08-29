import sys
import os

# Append project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_dynamic import train_and_evaluate

# Define experiment grid
experiments = [
    {"learning_rate": 0.001, "batch_size": 32, "epochs": 3, "run_name": "run_lr0.001_bs32"},
    {"learning_rate": 0.0005, "batch_size": 16, "epochs": 3, "run_name": "run_lr0.0005_bs16"},
    {"learning_rate": 0.0001, "batch_size": 32, "epochs": 3, "run_name": "run_lr0.0001_bs32"}
]

for exp in experiments:
    print(f"\n--- Running Experiment: {exp['run_name']} ---")
    train_and_evaluate(
        epochs=exp["epochs"],
        batch_size=exp["batch_size"],
        learning_rate=exp["learning_rate"],
        run_name=exp["run_name"]
    )

print("\nAll experiments completed successfully!")