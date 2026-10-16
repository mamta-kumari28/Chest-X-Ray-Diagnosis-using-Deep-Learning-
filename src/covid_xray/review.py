"""Private, reproducible review tools for duplicate-image candidates.

Review CSV files and contact sheets contain paths and patient identifiers. Keep
them in ignored local output directories and never publish them without permission.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


REVIEW_STATUSES = {"accepted", "rejected", "unresolved"}
REVIEW_COLUMNS = {
    "image_path",
    "label",
    "patient_id",
    "original_partition",
    "sha256",
    "perceptual_hash",
}


def _canonical_review_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept audited metadata or an image index without inventing identities."""
    data = frame.copy()
    aliases = {"path": "image_path", "original_split": "original_partition"}
    for old, new in aliases.items():
        if new not in data and old in data:
            data = data.rename(columns={old: new})
    missing = REVIEW_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Duplicate review requires columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Duplicate review input is empty.")
    for column in REVIEW_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    if not set(data.original_partition).issubset({"train", "test"}):
        raise ValueError("Duplicate review requires official train/test partitions.")
    return data


def _decision_map(decisions: pd.DataFrame | None) -> dict[str, tuple[str, str]]:
    """Validate one consistent human decision per candidate group."""
    if decisions is None:
        return {}
    required = {"group_id", "review_status"}
    if not required.issubset(decisions):
        raise ValueError(f"Review decisions require columns: {sorted(required)}")
    data = decisions.copy()
    data["group_id"] = data.group_id.fillna("").astype(str).str.strip()
    data["review_status"] = data.review_status.fillna("").astype(str).str.strip().str.lower()
    if "review_notes" not in data:
        data["review_notes"] = ""
    data["review_notes"] = data.review_notes.fillna("").astype(str)
    if data.group_id.eq("").any():
        raise ValueError("Review decisions contain an empty group_id.")
    unknown = set(data.review_status) - REVIEW_STATUSES
    if unknown:
        raise ValueError(f"Unknown review statuses: {sorted(unknown)}")
    if data.groupby("group_id").review_status.nunique().gt(1).any():
        raise ValueError("A duplicate group has conflicting review decisions.")
    result = {}
    for group_id, group in data.groupby("group_id", sort=False):
        notes = " | ".join(note for note in group.review_notes.unique() if note)
        result[group_id] = (group.review_status.iloc[0], notes)
    return result


def build_duplicate_review(
    frame: pd.DataFrame, decisions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Create a private row-level review table for exact and perceptual groups.

    Exact decoded-pixel matches are confirmed computational evidence and always
    remain linked by the splitter. Perceptual matches remain candidates until a
    human marks their group accepted or rejected.
    """
    data = _canonical_review_input(frame)
    decision_map = _decision_map(decisions)
    rows = []
    specifications = (
        ("exact", "sha256", "confirmed_exact"),
        ("perceptual", "perceptual_hash", "candidate"),
    )
    known_groups = set()
    for evidence_type, hash_column, evidence_status in specifications:
        usable = data[data[hash_column].ne("")]
        counts = usable[hash_column].value_counts()
        for value in sorted(counts[counts.gt(1)].index):
            group = usable[usable[hash_column].eq(value)].copy()
            group_id = f"{evidence_type}:{value}"
            known_groups.add(group_id)
            review_status, notes = decision_map.get(group_id, ("unresolved", ""))
            common = {
                "group_id": group_id,
                "evidence_type": evidence_type,
                "evidence_status": evidence_status,
                "group_size": int(len(group)),
                "crosses_official_partition": group.original_partition.nunique() > 1,
                "conflicting_labels": group.label.nunique() > 1,
                "differing_patient_ids": group.patient_id.nunique() > 1,
                "review_status": review_status,
                "review_notes": notes,
            }
            for record in group.to_dict("records"):
                rows.append({**common, **record})
    unknown_groups = set(decision_map) - known_groups
    if unknown_groups:
        raise ValueError(
            "Review decisions reference unknown candidate groups: "
            f"{sorted(unknown_groups)[:5]}"
        )
    columns = [
        "group_id",
        "evidence_type",
        "evidence_status",
        "group_size",
        "crosses_official_partition",
        "conflicting_labels",
        "differing_patient_ids",
        "review_status",
        "review_notes",
        *data.columns,
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_duplicate_review(review: pd.DataFrame) -> dict[str, int]:
    """Return JSON-safe group counts without paths or patient identifiers."""
    if review.empty:
        return {
            "exact_groups": 0,
            "perceptual_candidate_groups": 0,
            "cross_partition_exact_groups": 0,
            "cross_partition_perceptual_groups": 0,
            "unresolved_cross_partition_perceptual_groups": 0,
            "accepted_cross_partition_perceptual_groups": 0,
            "rejected_cross_partition_perceptual_groups": 0,
            "accepted_perceptual_groups": 0,
            "rejected_perceptual_groups": 0,
            "unresolved_perceptual_groups": 0,
        }
    groups = review.drop_duplicates("group_id")
    exact = groups.evidence_type.eq("exact")
    perceptual = groups.evidence_type.eq("perceptual")
    cross = groups.crosses_official_partition.astype(bool)
    return {
        "exact_groups": int(exact.sum()),
        "perceptual_candidate_groups": int(perceptual.sum()),
        "cross_partition_exact_groups": int((exact & cross).sum()),
        "cross_partition_perceptual_groups": int((perceptual & cross).sum()),
        "unresolved_cross_partition_perceptual_groups": int(
            (perceptual & cross & groups.review_status.eq("unresolved")).sum()
        ),
        "accepted_cross_partition_perceptual_groups": int(
            (perceptual & cross & groups.review_status.eq("accepted")).sum()
        ),
        "rejected_cross_partition_perceptual_groups": int(
            (perceptual & cross & groups.review_status.eq("rejected")).sum()
        ),
        "accepted_perceptual_groups": int(
            (perceptual & groups.review_status.eq("accepted")).sum()
        ),
        "rejected_perceptual_groups": int(
            (perceptual & groups.review_status.eq("rejected")).sum()
        ),
        "unresolved_perceptual_groups": int(
            (perceptual & groups.review_status.eq("unresolved")).sum()
        ),
    }


def accepted_perceptual_groups(review: pd.DataFrame) -> set[str]:
    """Return only human-accepted perceptual group IDs."""
    if review.empty:
        return set()
    groups = review.drop_duplicates("group_id")
    is_perceptual = (
        groups.evidence_type.eq("perceptual")
        if "evidence_type" in groups
        else groups.group_id.str.startswith("perceptual:")
    )
    return set(
        groups.loc[
            is_perceptual & groups.review_status.eq("accepted"),
            "group_id",
        ]
    )


def enforce_official_duplicate_boundary(review: pd.DataFrame) -> None:
    """Fail closed for exact, accepted, or unresolved official-boundary links."""
    if review.empty:
        return
    groups = review.drop_duplicates("group_id")
    crossing = groups[groups.crosses_official_partition.astype(bool)]
    exact = crossing[crossing.evidence_type.eq("exact")]
    if not exact.empty:
        raise ValueError(
            "Exact duplicate images cross the official train/test boundary; "
            "document an exclusion policy before training."
        )
    candidates = crossing[crossing.evidence_type.eq("perceptual")]
    unresolved = candidates[candidates.review_status.eq("unresolved")]
    if not unresolved.empty:
        raise ValueError(
            "Unresolved perceptual duplicate candidates cross the official "
            "train/test boundary; complete human review before splitting."
        )
    accepted = candidates[candidates.review_status.eq("accepted")]
    if not accepted.empty:
        raise ValueError(
            "Human-accepted duplicate groups cross the official train/test "
            "boundary; document an exclusion policy before training."
        )


def create_contact_sheets(
    review: pd.DataFrame,
    output: str | Path,
    max_groups: int | None = None,
    data_dir: str | Path | None = None,
) -> int:
    """Write private contact sheets for authorized human review."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_dir).expanduser().resolve() if data_dir else None
    groups = list(review.groupby("group_id", sort=True))
    if max_groups is not None:
        if max_groups < 1:
            raise ValueError("max_groups must be positive when provided.")
        groups = groups[:max_groups]
    written = 0
    for group_id, group in groups:
        cards = []
        for row in group.itertuples():
            path = Path(getattr(row, "resolved_path", ""))
            if not path.is_file() and data_root:
                path = (data_root / row.image_path).resolve()
                if not path.is_relative_to(data_root):
                    raise ValueError("Review image path leaves the configured data directory.")
            if not path.is_file():
                path = Path(row.image_path)
            if not path.is_file():
                continue
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((224, 224))
                card = Image.new("RGB", (260, 310), "white")
                card.paste(image, ((260 - image.width) // 2, 8))
            draw = ImageDraw.Draw(card)
            text = (
                f"{Path(row.image_path).name[:36]}\n"
                f"{row.label} | {row.original_partition}\n"
                f"patient: {row.patient_id[:30]}"
            )
            draw.multiline_text((8, 240), text, fill="black", spacing=3)
            cards.append(card)
        if not cards:
            continue
        sheet = Image.new("RGB", (260 * len(cards), 310), "white")
        for index, card in enumerate(cards):
            sheet.paste(card, (260 * index, 0))
        safe_name = group_id.replace(":", "-")[:80]
        sheet.save(output / f"{safe_name}.jpg", quality=90)
        written += 1
    return written


def write_duplicate_review(
    validated_metadata: str | Path,
    output: str | Path,
    decisions_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    contact_sheets: str | Path | None = None,
    max_contact_sheets: int | None = None,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create private review artifacts from a locally validated metadata CSV."""
    frame = pd.read_csv(validated_metadata, dtype=str, keep_default_na=False)
    decisions = (
        pd.read_csv(decisions_path, dtype=str, keep_default_na=False)
        if decisions_path
        else None
    )
    review = build_duplicate_review(frame, decisions)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output, index=False)
    summary = summarize_duplicate_review(review)
    if summary_path:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    if contact_sheets:
        summary["contact_sheets_written"] = create_contact_sheets(
            review, contact_sheets, max_contact_sheets, data_dir
        )
    return review, summary
