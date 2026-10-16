"""Save descriptive tables and figures from an actual validated image manifest."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps

from covid_xray.config import CLASSES
from covid_xray.data import audit_metadata_file, load_manifest


def create_dataset_report(frame, quality_report, output):
    """Save Phase 2 EDA from images that were actually decoded locally."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    audited = frame.copy()
    audited["file_exists"] = audited.file_exists.astype(bool)
    audited["readable"] = audited.readable.astype(bool)

    summary = (
        audited.groupby(["original_partition", "label"], dropna=False)
        .agg(
            metadata_records=("image_path", "size"),
            records_with_accessible_images=("file_exists", "sum"),
            valid_image_records=("readable", "sum"),
            unique_patients=("patient_id", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(output / "dataset_summary.csv", index=False)
    (output / "data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2, allow_nan=False)
    )

    # Distribution plots contain only images decoded successfully in this run.
    valid = audited[audited.readable].copy()
    partitions = ["train", "test"]
    combinations = pd.MultiIndex.from_product(
        [CLASSES, partitions], names=["label", "original_partition"]
    )
    plot_data = (
        valid.groupby(["label", "original_partition"])
        .size()
        .reindex(combinations, fill_value=0)
        .rename("count")
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_data,
        x="label",
        y="count",
        hue="original_partition",
        order=CLASSES,
        hue_order=partitions,
        ax=axis,
    )
    axis.set(
        title="Locally validated readable images by class",
        xlabel="Class",
        ylabel="Count",
    )
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    if not valid.empty:
        # Display only locally decoded images; no placeholder pixels are generated.
        fig, axes = plt.subplots(len(CLASSES), 3, figsize=(9, 9), squeeze=False)
        for row_number, label in enumerate(CLASSES):
            examples = valid[valid.label.eq(label)].sample(frac=1, random_state=42).head(3)
            for column_number, axis in enumerate(axes[row_number]):
                axis.axis("off")
                if column_number < len(examples):
                    example = examples.iloc[column_number]
                    with Image.open(example.resolved_path) as image:
                        axis.imshow(ImageOps.exif_transpose(image).convert("L"), cmap="gray")
                    axis.set_title(f"{label} — local validated sample")
        fig.tight_layout()
        fig.savefig(output / "sample_images.png", dpi=140)
        plt.close(fig)

        dimensions = valid[["label", "width", "height"]].copy()
        dimensions[["width", "height"]] = dimensions[["width", "height"]].astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(
            data=dimensions,
            x="width",
            y="height",
            hue="label",
            hue_order=CLASSES,
            alpha=0.6,
            ax=axes[0],
        )
        axes[0].set_title("Original image dimensions")
        pixel_counts = np.log10(dimensions.width * dimensions.height)
        sns.histplot(pixel_counts, bins=25, ax=axes[1])
        axes[1].set(
            title="Image-area distribution",
            xlabel="log10(width × height)",
            ylabel="Images",
        )
        fig.tight_layout()
        fig.savefig(output / "image_dimensions.png", dpi=160)
        plt.close(fig)
    print(
        f"Saved Phase 2 EDA to {output}. "
        f"Validated {quality_report['valid_images']} of "
        f"{quality_report['metadata_records']} metadata records locally."
    )
    return summary


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--metadata")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--validated-output", default="data/processed/validated_metadata.csv"
    )
    parser.add_argument("--output", default="artifacts/results/eda")
    args = parser.parse_args()
    if args.metadata:
        if not args.data_dir:
            parser.error("--data-dir is required with --metadata")
        report_path = Path(args.output) / "data_quality_report.json"
        frame, report = audit_metadata_file(
            args.metadata,
            args.data_dir,
            args.validated_output,
            report_path,
        )
        create_dataset_report(frame, report, args.output)
    else:
        create_report(args.manifest, args.output)


if __name__ == "__main__":
    main()
