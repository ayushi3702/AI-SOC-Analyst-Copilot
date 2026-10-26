import duckdb, yaml, pandas as pd

DAY_MAP = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
           "thu": "Thursday", "fri": "Friday"}

def load_cfg(path="configs/settings.yaml"):
    return yaml.safe_load(open(path))

def load_frame(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_cfg()
    con = duckdb.connect(cfg["db_path"], read_only=True)
    df = con.execute("""
        SELECT e.event_id, e.timestamp, e.dst_port,
               l.label, l.is_attack, l.day, l.source_file,
               f.* EXCLUDE (event_id, source_port, destination_port)
        FROM events e
        JOIN labels l USING (event_id)
        JOIN flow_features f USING (event_id)
    """).df()
    con.close()

    # shrink memory: 2.8M rows x ~80 cols
    fcols = df.select_dtypes("float64").columns
    df[fcols] = df[fcols].astype("float32")

    if cfg["sample_frac"] < 1.0:
        df = df.sample(frac=cfg["sample_frac"], random_state=cfg["seed"])
    df = df.reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # normalise day names whether stored as 'tue' or 'Tuesday'
    df["day"] = df["day"].astype(str).str.strip().str[:3].str.lower().map(DAY_MAP)
    df["is_attack"] = df["is_attack"].astype("int8")
    df["family"] = df["label"].map(cfg["families"])
    df["block"] = df["day"] + "_" + df["timestamp"].dt.floor(f'{cfg["block_minutes"]}min').astype(str)
    return df
