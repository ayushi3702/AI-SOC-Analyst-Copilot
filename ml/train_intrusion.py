import argparse, mlflow, numpy as np
from sklearn.metrics import classification_report, f1_score, average_precision_score, roc_auc_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from ml.data import load_cfg, load_frame
from ml.splits import get_split
from ml.features import select_features

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["random", "block", "unseen"], required=True)
p.add_argument("--task", choices=["binary", "family"], default="binary")
p.add_argument("--with-port", action="store_true")
a = p.parse_args()

cfg = load_cfg()
df = load_frame(cfg)
X = select_features(df, use_dst_port=a.with_port)
target = "is_attack" if a.task == "binary" else "family"
y = df[target]
if a.task == "family":
    le = LabelEncoder(); y = le.fit_transform(y)
    names = list(le.classes_)
else:
    names = ["benign", "attack"]

tr, te = get_split(a.split, df, cfg)
Xtr, Xte = X.loc[tr], X.loc[te]
ytr = y[df.index.get_indexer(tr)] if a.task == "family" else y.loc[tr]
yte = y[df.index.get_indexer(te)] if a.task == "family" else y.loc[te]

model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                      tree_method="hist", n_jobs=-1, random_state=cfg["seed"])
model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
pred = model.predict(Xte)

proba = model.predict_proba(Xte)[:, 1] if a.task == "binary" else None
mlflow.set_experiment("intrusion_baseline")
with mlflow.start_run(run_name=f"{a.split}-{a.task}-port{a.with_port}"):
    mlflow.log_params({"split": a.split, "task": a.task, "with_port": a.with_port,
                       "n_features": X.shape[1], "n_train": len(tr), "n_test": len(te)})
    mlflow.log_metric("macro_f1", f1_score(yte, pred, average="macro"))
    if proba is not None:
        mlflow.log_metric("pr_auc", average_precision_score(yte, proba))
        mlflow.log_metric("roc_auc", roc_auc_score(yte, proba))
        mlflow.log_metric("attack_recall", recall_score(yte, pred))
        print(f"PR-AUC {average_precision_score(yte, proba):.4f}  "
              f"ROC-AUC {roc_auc_score(yte, proba):.4f}")
    print(classification_report(yte, pred, zero_division=0, digits=4))
