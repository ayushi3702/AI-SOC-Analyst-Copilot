"""
Cleaning functions for CICIDS2017 (the TrafficLabelling / GeneratedLabelledFlows version).

Problems this fixes:
  1. Column names have stray spaces (" Label", " Source IP") -> snake_case names.
  2. One column name appears twice ("Fwd Header Length") -> made unique.
  3. Flow Bytes/s and Flow Packets/s contain Infinity and NaN -> removed.
  4. Fully duplicated rows -> removed (a major cause of fake 99% accuracy).
  5. Label text has a broken dash ("Web Attack \x96 Brute Force") -> "Web Attack - Brute Force".
  6. Timestamps are 12-hour with no AM/PM ("7/7/2017 1:05" really means 13:05).
  7. Everything is float64/int64 -> downsized to save RAM and disk.

Requires pandas >= 2.0.
"""
import re

import numpy as np
import pandas as pd


def to_snake(name: str) -> str:
    """' Flow Bytes/s ' -> 'flow_bytes_s'"""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    return name.strip("_").lower()


def make_unique(names: list[str]) -> list[str]:
    """['a', 'a', 'b'] -> ['a', 'a_2', 'b']"""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            out.append(n)
    return out


def clean_label(series: pd.Series) -> pd.Series:
    """Replace any non-ASCII junk (the broken dash) with '-', tidy spaces."""
    s = series.astype(str)
    s = s.str.replace(r"[^\x20-\x7E]+", "-", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def fix_timestamps(series: pd.Series) -> pd.Series:
    """
    CICIDS2017 timestamps have no AM/PM. Capture ran roughly 08:00-17:00,
    so an hour between 0 and 7 must be afternoon (add 12 hours).
    """
    ts = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    is_afternoon = ts.dt.hour < 8
    return ts.where(~is_afternoon, ts + pd.Timedelta(hours=12))


def downsize(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> smallest int that fits. Roughly halves RAM."""
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean one CICIDS2017 file. Returns (clean_df, stats_dict)."""
    stats = {"rows_raw": len(df)}

    # 1-2. column names
    df.columns = make_unique([to_snake(c) for c in df.columns])

    required = {"flow_id", "source_ip", "destination_ip", "timestamp", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)}. You probably downloaded the "
            "'MachineLearningCVE' version. You need 'TrafficLabelling' / "
            "'GeneratedLabelledFlows' (it has IPs and timestamps)."
        )

    # empty rows (some files end with blank lines)
    df = df.dropna(how="all")
    stats["rows_after_empty"] = len(df)

    # 5. labels
    df["label"] = clean_label(df["label"])

    # 3. inf / NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    stats["dropped_inf_nan"] = before - len(df)

    # 6. timestamps
    df["timestamp"] = fix_timestamps(df["timestamp"])
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    stats["dropped_bad_timestamp"] = before - len(df)

    # 4. exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    stats["dropped_duplicates"] = before - len(df)

    # extra helper columns
    df["is_attack"] = (df["label"].str.upper() != "BENIGN").astype("int8")

    # 7. memory
    df = downsize(df).reset_index(drop=True)
    stats["rows_clean"] = len(df)
    stats["attack_rows"] = int(df["is_attack"].sum())
    return df, stats
