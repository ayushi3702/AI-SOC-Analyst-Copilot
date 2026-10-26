import numpy as np, pandas as pd, mlflow
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from ml.data import load_cfg, load_frame
from ml.splits import holdout_family_split
from ml.features import select_features

cfg = load_cfg()
df = load_frame(cfg)
X = select_features(df, use_dst_port=False)
y = df["is_attack"]
families = [f for f in df["family"].unique() if f != "Benign"]

rows = []
mlflow.set_experiment("leave_one_family_out")
for fam in families:
    tr, te = holdout_family_split(df, fam, cfg["seed"])
    Xtr, ytr, Xte, yte = X.loc[tr], y.loc[tr], X.loc[te], y.loc[te]
    if yte.sum() == 0:
        print(f"skip {fam}: no attack rows in test"); continue

    # supervised: has seen every OTHER attack family
    xgb = XGBClassifier(n_estimators=300, max_depth=8, tree_method="hist",
                        n_jobs=-1, random_state=cfg["seed"])
    xgb.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    p_sup = xgb.predict_proba(Xte)[:, 1]

    # anomaly: trained on benign only, so it has no concept of any attack
    iso = IsolationForest(n_estimators=200, random_state=cfg["seed"], n_jobs=-1)
    benign = Xtr[ytr == 0]
    iso.fit(benign.sample(min(len(benign), 200_000), random_state=0))
    p_ano = -iso.score_samples(Xte)

    # recall of the held-out family at a fixed 1% false-positive budget on benign
    def recall_at_fpr(score, fpr=0.01):
        thr = np.quantile(score[yte.values == 0], 1 - fpr)
        return recall_score(yte, (score > thr).astype(int))

    r = {"held_out": fam, "n_attack_test": int(yte.sum()),
         "attack_rate": float(yte.mean()),
         "xgb_pr_auc": average_precision_score(yte, p_sup),
         "xgb_roc_auc": roc_auc_score(yte, p_sup),
         "xgb_recall_at_1pct_FPR": recall_at_fpr(p_sup),
         "iso_pr_auc": average_precision_score(yte, p_ano),
         "iso_roc_auc": roc_auc_score(yte, p_ano),
         "iso_recall_at_1pct_FPR": recall_at_fpr(p_ano)}
    rows.append(r); print(r)
    with mlflow.start_run(run_name=f"holdout-{fam}"):
        mlflow.log_params({"held_out": fam})
        mlflow.log_metrics({k: v for k, v in r.items() if k != "held_out"})

out = pd.DataFrame(rows)
Path("evaluation/results").mkdir(parents=True, exist_ok=True)
out.to_csv("evaluation/results/leave_one_family_out.csv", index=False)
print(out.round(3).to_string(index=False))
