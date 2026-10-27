"""Plot accuracy and loss from the recorded training history."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"

history = pd.read_csv(RESULTS_DIR / "training_history.csv")

required = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
missing = set(required) - set(history.columns)

if missing:
    raise ValueError(f"Missing history columns: {sorted(missing)}")

if history.empty or not np.isfinite(history[required].to_numpy()).all():
    raise ValueError("Training history is empty or contains invalid values.")

if not history["epoch"].is_unique:
    raise ValueError("Repeated epoch numbers found.")

history = history.sort_values("epoch")
epochs = history["epoch"] + 1
best_row = history.loc[history["val_loss"].idxmin()]
best_epoch = int(best_row["epoch"]) + 1

for metric, title, filename in [
    ("accuracy", "Training and Validation Accuracy", "accuracy_curve.png"),
    ("loss", "Training and Validation Loss", "loss_curve.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, history[metric], marker="o", label="Training")
    ax.plot(
        epochs,
        history[f"val_{metric}"],
        marker="o",
        label="Validation",
    )
    ax.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label=f"Best validation-loss checkpoint: epoch {best_epoch}",
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = RESULTS_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)

print("Recorded epochs:", len(history))
print("Best validation-loss epoch:", best_epoch)