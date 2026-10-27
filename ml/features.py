import numpy as np, pandas as pd, yaml


def select_features(
    df, use_dst_port=None, cfg_path="configs/features.yaml", feature_names=None
):
    if feature_names is not None:
        missing = set(feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Input data is missing model features: {sorted(missing)}")
        X = df.loc[:, feature_names].copy()
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    cfg = yaml.safe_load(open(cfg_path))
    use_port = cfg["use_dst_port"] if use_dst_port is None else use_dst_port
    drop = set(cfg["drop_columns"])
    cols = [c for c in df.select_dtypes("number").columns if c not in drop]
    if not use_port and "dst_port" in cols:
        cols.remove("dst_port")

    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.loc[:, X.nunique() > 1]                       # constant columns
    sample_size = min(len(X), 200_000)
    if sample_size and "event_id" in df.columns:
        row_hashes = pd.util.hash_pandas_object(
            df["event_id"], index=False
        ).to_numpy(dtype=np.uint64)
        sample_positions = np.argpartition(row_hashes, sample_size - 1)[
            :sample_size
        ]
        s = X.iloc[sample_positions]
    else:
        s = X.sample(sample_size, random_state=0)
    X = X.loc[:, ~s.T.duplicated()]
    return X
