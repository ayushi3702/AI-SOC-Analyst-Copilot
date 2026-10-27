import argparse
import json
from pathlib import Path

import pandas as pd

from ml.data import load_cfg, load_frame
from ml.splits import get_split, train_validation_split


PHASE_COLUMNS = {
    "train": ("train_rows", "train_attack_rows", "train_labels"),
    "validation": (
        "validation_rows",
        "validation_attack_rows",
        "validation_labels",
    ),
    "test": ("test_rows", "test_attack_rows", "test_labels"),
}


def build_coverage_report(df, cfg, split="unseen"):
    """Summarize attack-family and raw-label coverage in the actual model splits."""
    required = {"label", "is_attack"}
    if split == "unseen":
        required.add("day")
    if split in {"block", "unseen"}:
        required.add("block")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coverage input is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Coverage input must not be empty")
    if df.index.has_duplicates:
        raise ValueError("Coverage input index values must be unique")
    attack_values = pd.to_numeric(df["is_attack"], errors="coerce")
    if attack_values.isna().any() or not attack_values.isin([0, 1]).all():
        raise ValueError("is_attack values must be binary")

    working = df.copy()
    working["is_attack"] = attack_values.astype(bool)
    working["family"] = working["label"].map(cfg["families"]).fillna("Unmapped")
    initial_train, test = get_split(split, working, cfg)
    train, validation = train_validation_split(
        working, initial_train, seed=cfg["seed"]
    )
    index_sets = {
        "train": pd.Index(train),
        "validation": pd.Index(validation),
        "test": pd.Index(test),
    }

    combined = pd.concat(
        [
            working.loc[indices, ["label", "family", "is_attack"]].assign(
                phase=phase
            )
            for phase, indices in index_sets.items()
        ],
        ignore_index=True,
    )
    phases = list(PHASE_COLUMNS)
    families = sorted(combined["family"].astype(str).unique())
    report = pd.DataFrame({"family": families})

    for phase in phases:
        rows_column, attacks_column, labels_column = PHASE_COLUMNS[phase]
        phase_data = combined.loc[combined["phase"] == phase]
        counts = phase_data.groupby("family", sort=True).agg(
            **{
                rows_column: ("label", "size"),
                attacks_column: ("is_attack", "sum"),
            }
        )
        raw_labels = (
            phase_data.groupby("family", sort=True)["label"]
            .agg(lambda labels: json.dumps(sorted(labels.astype(str).unique())))
            .rename(labels_column)
        )
        report = report.merge(counts, on="family", how="left")
        report = report.merge(raw_labels, on="family", how="left")
        report[rows_column] = report[rows_column].fillna(0).astype("int64")
        report[attacks_column] = report[attacks_column].fillna(0).astype("int64")
        report[labels_column] = report[labels_column].fillna("[]")

    report["seen_in_training"] = report["train_rows"] > 0
    report["missing_from_training"] = (
        (report["test_attack_rows"] > 0) & ~report["seen_in_training"]
    )
    report["unmapped_labels_in_test"] = (
        (report["family"] == "Unmapped") & (report["test_rows"] > 0)
    )
    return report.sort_values("family").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Report label/family coverage in fit, validation, and test splits."
    )
    parser.add_argument("--split", choices=["block", "unseen"], default="unseen")
    parser.add_argument(
        "--output",
        default="data/processed/coverage_unseen.csv",
        help="Path for the family coverage CSV report.",
    )
    args = parser.parse_args()

    cfg = load_cfg()
    cfg["sample_frac"] = 1.0
    frame = load_frame(cfg)
    report = build_coverage_report(frame, cfg, split=args.split)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f"Coverage report for '{args.split}' split:")
    print(report.to_string(index=False))
    missing_families = report.loc[
        report["missing_from_training"], "family"
    ].tolist()
    if missing_families:
        print(
            "\nTest attack families with no training examples: "
            + ", ".join(missing_families)
        )
    else:
        print("\nEvery mapped test attack family appears in training.")
    unmapped_labels = json.loads(
        report.loc[report["unmapped_labels_in_test"], "test_labels"]
        .iloc[0]
    ) if report["unmapped_labels_in_test"].any() else []
    if unmapped_labels:
        print("Unmapped raw test labels: " + ", ".join(unmapped_labels))
    print(f"\nSaved report to {output}")


if __name__ == "__main__":
    main()
