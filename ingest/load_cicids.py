"""
Load CICIDS2017 CSVs one at a time, clean them, save as Parquet.
One file at a time keeps RAM low (safe for an 8 GB laptop).

Run from the repo root:
    python -m ingest.load_cicids
    python -m ingest.load_cicids --force        # redo files already converted
"""
import argparse
import gc
from pathlib import Path

import pandas as pd

from ingest.clean import clean_dataframe, to_snake

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
OUT_DIR = ROOT / "data" / "interim"


def out_name(csv_path: Path) -> str:
    stem = csv_path.name.replace(".pcap_ISCX.csv", "").replace(".csv", "")
    return to_snake(stem) + ".parquet"


def main(raw_dir: Path, out_dir: Path, force: bool) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for csv in csv_files:
        target = out_dir / out_name(csv)
        if target.exists() and not force:
            print(f"[skip] {target.name} already exists (use --force to redo)")
            continue

        print(f"[read] {csv.name}")
        # latin-1 never fails on odd bytes; low_memory=False avoids mixed-type warnings
        df = pd.read_csv(csv, encoding="latin-1", low_memory=False)
        df, stats = clean_dataframe(df)
        df["source_file"] = csv.name
        df["day"] = csv.name.split("-")[0].capitalize()

        df.to_parquet(target, index=False)
        stats["file"] = csv.name
        stats["parquet"] = target.name
        report.append(stats)
        print(
            f"[done] {target.name}: {stats['rows_raw']:,} -> {stats['rows_clean']:,} rows "
            f"(dup {stats['dropped_duplicates']:,}, inf/NaN {stats['dropped_inf_nan']:,})"
        )
        del df
        gc.collect()  # free RAM before the next file

    if report:
        rep = pd.DataFrame(report)
        rep.to_csv(out_dir / "_cleaning_report.csv", index=False)
        print("\nCleaning report saved to", out_dir / "_cleaning_report.csv")

    # Quick look at the label counts across all Parquet files (reads only 1 column)
    parts = [pd.read_parquet(p, columns=["label"]) for p in sorted(out_dir.glob("*.parquet"))]
    if parts:
        print("\nLabel counts (all files):")
        print(pd.concat(parts)["label"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.raw, a.out, a.force)
