"""Save descriptive tables and figures from an actual validated image manifest."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps

from covid_xray.config import CLASSES
from covid_xray.data import load_manifest


def create_report(manifest, output):
    frame = load_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Tables are saved alongside plots so reviewers can inspect exact counts.
    counts = pd.crosstab(frame.label, frame.split).reindex(CLASSES)
    counts.to_csv(output / "class_counts.csv")
    frame.groupby("split").patient_id.nunique().to_csv(output / "patient_counts.csv")
    frame.groupby(["source", "label", "split"]).size().rename("images").to_csv(output / "source_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=frame, x="label", order=CLASSES, hue="label", legend=False, ax=axes[0])
    axes[0].set(title="Images by class", xlabel="Class", ylabel="Images")
    sns.countplot(data=frame, x="label", order=CLASSES, hue="split", hue_order=["train", "val", "test"], ax=axes[1])
    axes[1].set(title="Patient-safe split distribution", xlabel="Class", ylabel="Images")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Only training examples are previewed; these are local analysis artifacts.
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, label in enumerate(CLASSES):
        examples = frame[(frame.split == "train") & (frame.label == label)].sample(frac=1, random_state=42).head(3)
        for column in range(3):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(examples):
                with Image.open(examples.iloc[column].path) as image:
                    axis.imshow(ImageOps.exif_transpose(image).convert("L"), cmap="gray")
                axis.set_title(f"{label} — training sample")
    fig.tight_layout()
    fig.savefig(output / "training_samples_PRIVATE.png", dpi=140)
    plt.close(fig)

    # Inspect dimensions on a reproducible training subset without loading all pixels.
    sizes = []
    train = frame[frame.split == "train"].sample(frac=1, random_state=42).head(1000)
    for row in train.itertuples():
        with Image.open(row.path) as image:
            sizes.append({"width": image.width, "height": image.height, "label": row.label})
    size_frame = pd.DataFrame(sizes)
    size_frame.to_csv(output / "sample_dimensions.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=size_frame, x="width", y="height", hue="label", ax=axis, alpha=0.6)
    axis.set_title("Original dimensions: up to 1,000 training images")
    fig.tight_layout()
    fig.savefig(output / "image_dimensions.png", dpi=160)
    plt.close(fig)
    summary = frame.groupby("split").agg(images=("path", "size"), patients=("patient_id", "nunique"))
    summary.to_csv(output / "split_summary.csv")
    frame.groupby("split").agg(
        images=("path", "size"), unique_pixel_hashes=("sha256", "nunique")
    ).assign(duplicate_rows=lambda counts: counts.images - counts.unique_pixel_hashes).to_csv(
        output / "exact_duplicate_summary.csv"
    )
    print(summary.to_string())
    print(f"Saved EDA to {output}. Sample images may be license-restricted; do not publish them.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="artifacts/results/eda")
    args = parser.parse_args()
    create_report(args.manifest, args.output)


if __name__ == "__main__":
    main()
