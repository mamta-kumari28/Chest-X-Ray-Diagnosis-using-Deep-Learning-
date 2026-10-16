"""Evaluate a trained model once on held-out patients and save honest metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from covid_xray.config import CLASSES, IMAGE_SIZE
from covid_xray.data import load_manifest, make_dataset
from covid_xray.train import file_sha256


def json_ready(value):
    """Use JSON null for undefined metrics; never serialize nonstandard NaN."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_error_images(predictions, output: Path) -> None:
    """Show the most confident incorrect predictions for qualitative review."""
    errors = predictions.loc[
        predictions["label"] != predictions["predicted_label"]
    ].sort_values("confidence", ascending=False).head(12)
    if errors.empty:
        (output / "false_predictions.png").unlink(missing_ok=True)
        return
    cols = min(4, len(errors))
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.itertuples()):
        with Image.open(row.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            ax.imshow(image)
        ax.set_title(
            f"Actual: {row.label}\nPredicted: {row.predicted_label} ({row.confidence:.1%})",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output / "false_predictions.png", dpi=150)
    plt.close(fig)


def evaluate(
    manifest: Path,
    model: Path,
    output: Path,
    batch_size: int = 32,
) -> dict:
    """Write metrics, per-image predictions, ROC curves and FP/FN examples."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model = Path(model)
    metadata_path = model.parent / "model_metadata.json"
    if not model.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("A trained model and its model_metadata.json are required.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("trained") is not True:
        raise ValueError("The supplied artifact is not marked as successfully trained.")
    if metadata.get("class_names") != list(CLASSES):
        raise ValueError("Model class order does not match the configured classes.")
    if metadata.get("input_size") != list(IMAGE_SIZE):
        raise ValueError("Model image size does not match the configured size.")
    if metadata.get("sha256") != file_sha256(model):
        raise ValueError("Model checksum does not match its training metadata.")

    frame = load_manifest(manifest)
    if frame["source"].str.lower().str.startswith("synthetic").any():
        raise ValueError("Synthetic fixtures cannot produce portfolio evaluation results.")
    # A different split manifest can place training patients in the test set.
    if metadata.get("manifest_sha256") != file_sha256(Path(manifest)):
        raise ValueError("Evaluate with the exact manifest used during training.")
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if test_frame.empty:
        raise ValueError("The manifest contains no held-out test images.")
    dataset = make_dataset(test_frame, batch_size=batch_size, training=False)
    network = tf.keras.models.load_model(model, compile=False, safe_mode=True)
    probabilities = np.asarray(network.predict(dataset, verbose=0))
    if (
        probabilities.shape != (len(test_frame), len(CLASSES))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError("Model output is not a valid three-class probability matrix.")
    class_ids = {label: index for index, label in enumerate(CLASSES)}
    actual = test_frame["label"].map(class_ids).to_numpy()
    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    report = classification_report(
        actual, predicted, labels=labels, target_names=list(CLASSES),
        output_dict=True, zero_division=np.nan,
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Each one-vs-rest AUC needs both positive and negative test examples.
    aucs = {}
    skipped_auc = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(CLASSES):
        binary = (actual == index).astype(int)
        if len(np.unique(binary)) != 2:
            aucs[label] = None
            skipped_auc[label] = "Requires positive and negative test examples."
            continue
        aucs[label] = float(roc_auc_score(binary, probabilities[:, index]))
        false_positive_rate, true_positive_rate, _ = roc_curve(binary, probabilities[:, index])
        ax.plot(false_positive_rate, true_positive_rate, label=f"{label}: AUC {aucs[label]:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves")
    if any(value is not None for value in aucs.values()):
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "roc_curves.png", dpi=160)
    else:
        (output / "roc_curves.png").unlink(missing_ok=True)
    plt.close(fig)

    undefined = []
    for index, label in enumerate(CLASSES):
        if not (predicted == index).any():
            undefined.append(f"{label}: precision undefined because no images were predicted as this class.")
        if not (actual == index).any():
            undefined.append(f"{label}: recall undefined because there are no true test examples.")
        if not (predicted == index).any() and not (actual == index).any():
            undefined.append(f"{label}: F1 undefined because actual and predicted counts are both zero.")
    summary = {}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=labels, average=average, zero_division=np.nan
        )
        summary[average] = {"precision": precision, "recall": recall, "f1": f1}
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        **summary,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASSES),
        "roc_auc_ovr": aucs,
        "roc_auc_macro_ovr": (
            float(roc_auc_score(actual, probabilities, labels=labels, multi_class="ovr", average="macro"))
            if set(actual) == set(labels) else None
        ),
        "skipped_auc": skipped_auc,
        "undefined_metrics": undefined,
        "metric_notes": (
            "Undefined values are null. Macro and weighted averages exclude undefined values. "
            "Confidence is a softmax score, not a calibrated probability of disease. "
            "These are image-level metrics on held-out patients."
        ),
        "test_images": len(test_frame),
        "test_patients": int(test_frame["patient_id"].nunique()),
        "model_sha256": metadata["sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
    }
    metrics = json_ready(metrics)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (output / "classification_report.json").write_text(
        json.dumps(json_ready(report), indent=2, allow_nan=False)
    )
    (output / "classification_report.txt").write_text(
        classification_report(
            actual, predicted, labels=labels, target_names=list(CLASSES),
            zero_division=np.nan,
        )
        + "\nUndefined metrics appear as nan; see metrics.json for reasons.\n"
    )

    predictions = test_frame.copy()
    predictions["predicted_label"] = [CLASSES[index] for index in predicted]
    predictions["confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(CLASSES):
        predictions[f"score_{label}"] = probabilities[:, index]
    predictions.to_csv(output / "predictions.csv", index=False)
    error_analysis = {}
    error_lines = [
        "# Held-out test error analysis",
        "",
        "False positives and false negatives below use one class versus all other classes.",
        "Scores describe a research benchmark; they do not measure clinical safety.",
        "",
        "| Class | Actual images | False negatives | False negative rate | False positives | False positive rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, label in enumerate(CLASSES):
        false_negatives = predictions.loc[(actual == index) & (predicted != index)]
        false_positives = predictions.loc[(actual != index) & (predicted == index)]
        positives = int((actual == index).sum())
        negatives = int((actual != index).sum())
        fn_rate = len(false_negatives) / positives if positives else None
        fp_rate = len(false_positives) / negatives if negatives else None
        columns = ["path", "label", "patient_id", "predicted_label", "confidence"]
        error_analysis[label] = {
            "actual_images": positives,
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
            "false_negative_rate": fn_rate,
            "false_positive_rate": fp_rate,
            "false_negative_examples": false_negatives.sort_values(
                "confidence", ascending=False
            )[columns].head(20).to_dict("records"),
            "false_positive_examples": false_positives.sort_values(
                "confidence", ascending=False
            )[columns].head(20).to_dict("records"),
        }
        fn_text = f"{fn_rate:.1%}" if fn_rate is not None else "undefined"
        fp_text = f"{fp_rate:.1%}" if fp_rate is not None else "undefined"
        error_lines.append(
            f"| {label} | {positives} | {len(false_negatives)} | {fn_text} | "
            f"{len(false_positives)} | {fp_text} |"
        )
    (output / "error_analysis.json").write_text(
        json.dumps(json_ready(error_analysis), indent=2, allow_nan=False)
    )
    confusions = [
        (int(matrix[i, j]), CLASSES[i], CLASSES[j])
        for i in labels for j in labels if i != j and matrix[i, j] > 0
    ]
    error_lines.extend(["", "## Observed class confusions", ""])
    if confusions:
        for count, true_label, predicted_label in sorted(confusions, reverse=True):
            error_lines.append(
                f"- {count} actual {true_label} images were predicted as {predicted_label}."
            )
    else:
        error_lines.append("No misclassifications were observed in this specific test sample.")
    error_lines.extend([
        "",
        "The highest-confidence incorrect predictions are saved in false_predictions.png "
        "when errors exist. Per-class examples are in error_analysis.json.",
        "",
        "Inspect these examples and their source distributions before attributing causes. "
        "Acquisition equipment, annotations, views, patient cohorts and source-specific "
        "artifacts can confound performance. These outputs alone cannot establish the cause "
        "of an error. A separate external and subgroup audit is required before discussing "
        "generalization. COVID-19 false negatives deserve particular review, and an "
        "apparently confident prediction cannot rule disease in or out.",
        "",
        "Patient metadata and paths in generated reports must remain private unless their "
        "release is specifically permitted by the source terms.",
    ])
    (output / "error_analysis.md").write_text("\n".join(error_lines) + "\n")
    save_error_images(predictions, output)
    print(f"Test accuracy: {metrics['accuracy']:.4f}; results saved to {output}")
    if undefined or skipped_auc:
        print("Some metrics are undefined; see metrics.json for explicit reasons.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.keras"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
