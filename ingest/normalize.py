"""
Map cleaned CICIDS2017 data to the common schema (configs/schema.yaml).

One cleaned file goes in, THREE tables come out:

  events         what the AI agents may query (no labels!)
  labels         ground truth, used only for ML training and evaluation
  flow_features  the numeric columns the ML models learn from

Keeping labels out of `events` matters: if the log-hunter agent could read the
`label` column, it would just look up the answer and your evaluation would be fake.
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "schema.yaml"


def load_schema(path: Path = SCHEMA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give every numeric column ONE fixed type (int64 / float64), whatever the file.
    clean.py shrinks types per file (int32 in one file, int64 in another), and DuckDB
    fixes a column's type when the table is first created, so later files would not fit.
    """
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int64")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float64")
    return df


def normalize(df: pd.DataFrame, file_stem: str, schema: dict | None = None) -> dict[str, pd.DataFrame]:
    schema = schema or load_schema()
    df = df.reset_index(drop=True)

    mapping: dict[str, str] = schema["events"]
    needed = set(mapping.values()) | {"label", "is_attack"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Cleaned data is missing columns: {sorted(missing)}")

    # Flow ID is NOT unique in CICIDS2017, so we make our own unique id.
    event_id = file_stem + ":" + df.index.astype(str).str.zfill(7)

    # ---- events -------------------------------------------------------
    events = pd.DataFrame({"event_id": event_id})
    events["event_type"] = schema["event_type"]
    events["source_dataset"] = schema["source_dataset"]
    for new_name, old_name in mapping.items():
        events[new_name] = df[old_name].values
    proto = {int(k): v for k, v in schema["protocol_names"].items()}
    events["protocol"] = (
        df[mapping["protocol"]].map(proto).fillna("other").values
    )
    events = events.sort_values("timestamp").reset_index(drop=True)

    # ---- labels (ground truth, hidden from agents) ----------------------
    label_cols = ["label", "is_attack"] + [c for c in ("day", "source_file") if c in df.columns]
    labels = pd.concat([pd.Series(event_id, name="event_id"), df[label_cols]], axis=1)

    # ---- ML features ----------------------------------------------------
    num_cols = [c for c in df.select_dtypes("number").columns if c != "is_attack"]
    features = pd.concat([pd.Series(event_id, name="event_id"), df[num_cols]], axis=1)

    return {
        "events": standardize_dtypes(events),
        "labels": labels,
        "flow_features": standardize_dtypes(features),
    }
