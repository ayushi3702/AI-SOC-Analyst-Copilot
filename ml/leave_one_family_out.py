import pandas as pd, mlflow
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from ml.data import load_cfg, load_frame
from ml.evaluate import (
    false_positive_rate_at_threshold,
    recall_at_threshold,
    threshold_for_fpr,
)
from ml.splits import holdout_family_split, train_validation_split
from ml.features import select_features


def main():
    cfg = load_cfg()
    df = load_frame(cfg)
    X = select_features(df, use_dst_port=False)
    y = df["is_attack"]
    families = [f for f in df["family"].unique() if f != "Benign"]

    rows = []
    mlflow.set_experiment("leave_one_family_out")
    for fam in families:
        family_train, test = holdout_family_split(df, fam, cfg["seed"])
        train, validation = train_validation_split(
            df, family_train, seed=cfg["seed"]
        )
        Xtr, ytr = X.loc[train], y.loc[train]
        Xval, yval = X.loc[validation], y.loc[validation]
        Xte, yte = X.loc[test], y.loc[test]
        if yte.sum() == 0:
            print(f"skip {fam}: no attack rows in test")
            continue

        # Supervised model sees every attack family except the held-out family.
        xgb = XGBClassifier(n_estimators=300, max_depth=8, tree_method="hist",
                            n_jobs=-1, random_state=cfg["seed"])
        xgb.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        p_sup_val = xgb.predict_proba(Xval)[:, 1]
        p_sup = xgb.predict_proba(Xte)[:, 1]

        # The anomaly model is fit only on benign training rows.
        iso = IsolationForest(n_estimators=200, random_state=cfg["seed"], n_jobs=-1)
        benign = Xtr[ytr == 0]
        iso.fit(benign.sample(min(len(benign), 200_000), random_state=0))
        p_ano_val = -iso.score_samples(Xval)
        p_ano = -iso.score_samples(Xte)

        # Calibrate on validation benign rows; the test set is used only for scoring.
        sup_threshold = threshold_for_fpr(yval, p_sup_val)
        iso_threshold = threshold_for_fpr(yval, p_ano_val)
        r = {"held_out": fam, "n_attack_test": int(yte.sum()),
             "attack_rate": float(yte.mean()),
             "xgb_pr_auc": average_precision_score(yte, p_sup),
             "xgb_roc_auc": roc_auc_score(yte, p_sup),
             "xgb_recall_at_validation_1pct_fpr": recall_at_threshold(
                 yte, p_sup, sup_threshold
             ),
             "xgb_test_fpr": false_positive_rate_at_threshold(
                 yte, p_sup, sup_threshold
             ),
             "iso_pr_auc": average_precision_score(yte, p_ano),
             "iso_roc_auc": roc_auc_score(yte, p_ano),
             "iso_recall_at_validation_1pct_fpr": recall_at_threshold(
                 yte, p_ano, iso_threshold
             ),
             "iso_test_fpr": false_positive_rate_at_threshold(
                 yte, p_ano, iso_threshold
             )}
        rows.append(r)
        print(r)
        with mlflow.start_run(run_name=f"holdout-{fam}"):
            mlflow.log_params({
                "held_out": fam,
                "validation_size": 0.2,
                "threshold_calibration_fpr": 0.01,
            })
            mlflow.log_metrics({k: v for k, v in r.items() if k != "held_out"})

    out = pd.DataFrame(rows)
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    out.to_csv("evaluation/results/leave_one_family_out.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
