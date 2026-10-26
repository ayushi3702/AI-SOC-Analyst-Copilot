import numpy as np, pandas as pd, yaml

def select_features(df, use_dst_port=None, cfg_path="configs/features.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    use_port = cfg["use_dst_port"] if use_dst_port is None else use_dst_port
    drop = set(cfg["drop_columns"])
    cols = [c for c in df.select_dtypes("number").columns if c not in drop]
    if not use_port and "dst_port" in cols:
        cols.remove("dst_port")

    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.loc[:, X.nunique() > 1]                       # constant columns
    s = X.sample(min(len(X), 200_000), random_state=0)  # duplicate columns, on a sample
    X = X.loc[:, ~s.T.duplicated()]
    return X
