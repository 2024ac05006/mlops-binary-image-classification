import pandas as pd
from evidently import ColumnMapping
from evidently.presets import ClassificationPreset
from evidently import Report


def generate_report():
    # 1. Load logged predictions
    preds_df = pd.read_csv("data/predictions.csv")

    # 2. Add simulated ground-truth targets for post-deployment evaluation
    preds_df["target"] = [
        "cat" if i % 2 == 0 else "dog" for i in range(len(preds_df))
    ]

    # 3. Explicitly map dataset columns for Evidently AI
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"

    # 4. Build and run classification report
    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=None,
        current_data=preds_df,
        column_mapping=column_mapping,
    )

    # 5. Save HTML output
    report.save_html("evidently_report.html")
    print("✅ Evidently performance report generated: evidently_report.html")


if __name__ == "__main__":
    generate_report()