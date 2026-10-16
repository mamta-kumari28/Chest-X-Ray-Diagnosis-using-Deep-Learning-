"""Train the custom CNN on a validated manifest; keep the test split untouched."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from covid_xray.config import CLASSES, IMAGE_SIZE
from covid_xray.data import load_manifest, make_dataset
from covid_xray.model import build_model


def file_sha256(path: Path) -> str:
    """Hash files in chunks so a large model does not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_history(history: dict, output: Path) -> None:
    """Save the real epoch values and a reusable training-curve figure."""
    with (output / "history.json").open("w") as file:
        json.dump(history, file, indent=2, allow_nan=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], label="Training")
        ax.plot(epochs, history[f"val_{metric}"], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def train(
    manifest: Path,
    output: Path,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
    learning_rate: float = 1e-3,
    overwrite: bool = False,
) -> dict:
    """Fit on train/validation data and return metadata for the saved best model."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    manifest_hash = file_sha256(Path(manifest))
    frame = load_manifest(manifest)
    if frame["source"].str.lower().str.startswith("synthetic").any():
        raise ValueError("Synthetic fixtures cannot be used for a trained portfolio model.")

    train_frame = frame.loc[frame["split"] == "train"].copy()
    val_frame = frame.loc[frame["split"] == "val"].copy()
    for name, part in (("train", train_frame), ("val", val_frame)):
        missing = set(CLASSES) - set(part["label"])
        if missing:
            raise ValueError(f"The {name} split is missing classes: {sorted(missing)}")

    output = Path(output)
    model_path = output / "model.keras"
    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"{model_path} already exists. Choose another --output or use --overwrite."
        )
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate a previous run before training. A failed run must never leave
    # metadata that presents an old or partially replaced model as current.
    (output / "model_metadata.json").write_text(
        json.dumps({"trained": False, "status": "training_in_progress"}, indent=2)
    )
    if overwrite:
        model_path.unlink(missing_ok=True)

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Older TensorFlow builds may not expose this optional reproducibility aid.
        pass
    gpu_names = [device.name for device in tf.config.list_physical_devices("GPU")]
    if not gpu_names:
        print("No GPU detected. Full training is intended for a Google Colab GPU runtime.")

    train_data = make_dataset(train_frame, batch_size=batch_size, training=True, seed=seed)
    val_data = make_dataset(val_frame, batch_size=batch_size, training=False, seed=seed)
    model = build_model(seed=seed, learning_rate=learning_rate)
    model.summary()

    # Class weighting reduces the effect of imbalance without duplicating images.
    counts = train_frame["label"].value_counts()
    weights = {
        index: len(train_frame) / (len(CLASSES) * int(counts[label]))
        for index, label in enumerate(CLASSES)
    }
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(str(output / "training_log.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    result = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=weights,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in result.history.items()
    }
    if not model_path.is_file() or not all(
        np.isfinite(values).all() for values in history.values()
    ):
        raise RuntimeError("Training did not produce a finite, valid best checkpoint.")
    if file_sha256(Path(manifest)) != manifest_hash:
        raise RuntimeError("The manifest changed during training; restore it and rerun.")
    save_history(history, output)
    (output / "class_names.json").write_text(json.dumps(list(CLASSES), indent=2))
    best_epoch = int(np.argmin(history["val_loss"]))
    metadata = {
        "trained": True,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "custom_cnn",
        "class_names": list(CLASSES),
        "input_size": list(IMAGE_SIZE),
        "input_channels": 3,
        "input_range": [0, 255],
        "normalization": "embedded_rescaling_1_over_255",
        "augmentation": "embedded_training_only",
        "sha256": file_sha256(model_path),
        "manifest_sha256": manifest_hash,
        "seed": seed,
        "epochs_completed": len(history["loss"]),
        "best_epoch": best_epoch + 1,
        "best_validation_loss": history["val_loss"][best_epoch],
        "validation_accuracy_at_best_epoch": history["val_accuracy"][best_epoch],
        "train_counts": {label: int(counts[label]) for label in CLASSES},
        "validation_counts": {
            label: int((val_frame["label"] == label).sum()) for label in CLASSES
        },
        "class_weights": {CLASSES[index]: value for index, value in weights.items()},
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "gpu_devices": gpu_names,
        "test_evaluated": False,
        "purpose": "Experimental research; not a medical diagnosis.",
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    print(f"Best model saved to {model_path}. Test images have not been evaluated.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
