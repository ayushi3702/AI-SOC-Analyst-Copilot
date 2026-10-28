import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evaluate import threshold_for_fpr


LOGIN_FEATURES = (
    "event_count",
    "success_count",
    "failure_count",
    "source_computer_events",
    "destination_computer_events",
    "kerberos_events",
    "ntlm_events",
    "negotiate_events",
    "network_events",
    "interactive_events",
    "service_events",
    "batch_events",
    "logon_events",
    "logoff_events",
    "previous_24h_events",
    "previous_24h_failures",
    "previous_24h_successes",
    "hours_since_previous_activity",
    "hour_sin",
    "hour_cos",
)
COUNT_FEATURES = LOGIN_FEATURES[:-3]
ARTIFACT_VERSION = 1


def transform_login_features(frame):
    missing = set(LOGIN_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Login data is missing features: {sorted(missing)}")
    features = frame.loc[:, list(LOGIN_FEATURES)].astype("float32").copy()
    if (features[list(COUNT_FEATURES)] < 0).any().any():
        raise ValueError("Login count features must be non-negative")
    features["hours_since_previous_activity"] = (
        features["hours_since_previous_activity"].fillna(24 * 58)
    )
    features.loc[:, list(COUNT_FEATURES)] = np.log1p(
        features.loc[:, list(COUNT_FEATURES)]
    )
    features["hours_since_previous_activity"] = np.log1p(
        features["hours_since_previous_activity"].clip(lower=0)
    )
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Login feature transformation produced non-finite values")
    return features


def _metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores > threshold
    attacks = labels == 1
    benign = ~attacks
    true_positives = int(np.sum(predicted & attacks))
    false_positives = int(np.sum(predicted & benign))
    return {
        "threshold": float(threshold),
        "rows": int(len(labels)),
        "redteam_user_hours": int(attacks.sum()),
        "benign_user_hours": int(benign.sum()),
        "alerts": int(predicted.sum()),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall": (
            true_positives / int(attacks.sum()) if attacks.any() else None
        ),
        "false_positive_rate": (
            false_positives / int(benign.sum()) if benign.any() else None
        ),
        "precision": (
            true_positives / int(predicted.sum()) if predicted.any() else None
        ),
        "pr_auc": (
            float(average_precision_score(labels, scores))
            if attacks.any() and benign.any()
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(labels, scores))
            if attacks.any() and benign.any()
            else None
        ),
    }


def _fit_isolation_forest(train, n_estimators, seed):
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(256, len(train)),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(transform_login_features(train))
    return model


def _build_report_and_artifact(
    model,
    validation_labels,
    validation_scores,
    test_labels,
    test_scores,
    fit_end_hour,
    validation_end_hour,
    target_fpr,
    training_rows_available,
    training_rows_used,
    seed,
):
    if not np.any(validation_labels == 0):
        raise ValueError("Validation period must include benign user-hour rows")
    if not np.any(test_labels == 0) or not np.any(test_labels == 1):
        raise ValueError(
            "Test period must include both benign and red-team user-hour rows"
        )
    threshold = threshold_for_fpr(
        validation_labels, validation_scores, fpr=target_fpr
    )
    max_samples = min(256, training_rows_used)
    report = {
        "report_name": "LANL login anomaly detector evaluation",
        "method": {
            "detector": "IsolationForest trained on red-team-unlabeled user-hour rows",
            "feature_unit": "user-hour aggregate",
            "time_feature_note": (
                "LANL timestamps are relative epochs; cyclic hour features "
                "preserve daily periodicity but are not local wall-clock time."
            ),
            "fit_end_hour_exclusive": int(fit_end_hour),
            "validation_end_hour_exclusive": int(validation_end_hour),
            "split": "chronological; validation/test occur after fit",
            "fit_redteam_rows_excluded": True,
            "target_validation_false_positive_rate": float(target_fpr),
            "threshold_policy": (
                "Validation benign anomaly-score quantile, fixed on test"
            ),
            "training_rows_available": int(training_rows_available),
            "training_rows_used": int(training_rows_used),
            "validation_rows": int(len(validation_labels)),
            "test_rows": int(len(test_labels)),
            "validation_redteam_user_hours": int(validation_labels.sum()),
            "test_redteam_user_hours": int(test_labels.sum()),
            "seed": int(seed),
            "n_estimators": int(model.n_estimators),
            "max_samples": int(max_samples),
            "feature_names": list(LOGIN_FEATURES),
            "limitations": [
                "Only red-team compromise events present in the supplied LANL labels are positives.",
                "User-hour aggregation can dilute a single malicious authentication event.",
                "LANL relative timestamps do not reveal local wall-clock hours.",
                "Detection performance on this historical de-identified dataset is not production performance.",
            ],
        },
        "validation_metrics": _metrics(
            validation_labels, validation_scores, threshold
        ),
        "test_metrics": _metrics(test_labels, test_scores, threshold),
    }
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "task": "login_anomaly",
        "model": model,
        "feature_names": list(LOGIN_FEATURES),
        "threshold": float(threshold),
        "threshold_comparison": ">",
        "target_fpr": float(target_fpr),
        "fit_end_hour_exclusive": int(fit_end_hour),
        "validation_end_hour_exclusive": int(validation_end_hour),
        "seed": int(seed),
    }
    return report, artifact


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_training_sample(frame, mask, max_rows):
    positions = np.flatnonzero(mask.to_numpy())
    sample_size = min(len(positions), max_rows)
    if not sample_size:
        return frame.iloc[positions]
    keys = frame.iloc[positions][["user_id", "hour_index"]]
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(
        dtype=np.uint64
    )
    selected = np.argpartition(hashes, sample_size - 1)[:sample_size]
    selected = selected[np.argsort(hashes[selected], kind="stable")]
    return frame.iloc[positions[selected]]


def train_login_anomaly_detector(
    frame,
    fit_end_hour=9 * 24,
    validation_end_hour=15 * 24,
    target_fpr=0.01,
    max_train_rows=200_000,
    n_estimators=200,
    seed=42,
):
    required = {
        "user_id",
        "hour_index",
        "redteam_event_count",
        *LOGIN_FEATURES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Login feature data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Login feature data is empty")
    if not 0 < fit_end_hour < validation_end_hour:
        raise ValueError("Time split boundaries must satisfy 0 < fit < validation")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    if max_train_rows <= 0 or n_estimators <= 0:
        raise ValueError("Training sample size and estimator count must be positive")

    hours = pd.to_numeric(frame["hour_index"], errors="coerce")
    redteam_counts = pd.to_numeric(
        frame["redteam_event_count"], errors="coerce"
    )
    if hours.isna().any() or redteam_counts.isna().any():
        raise ValueError("Hour indices and red-team counts must be numeric")
    if (redteam_counts < 0).any():
        raise ValueError("Red-team event counts must be non-negative")
    labels = (redteam_counts > 0).astype("int8")

    fit_mask = hours < fit_end_hour
    validation_mask = (hours >= fit_end_hour) & (hours < validation_end_hour)
    test_mask = hours >= validation_end_hour
    train_mask = fit_mask & (labels == 0)
    if not train_mask.any():
        raise ValueError("Fit period contains no benign user-hour rows")
    if not validation_mask.any() or not test_mask.any():
        raise ValueError("Validation and test periods must both contain rows")

    validation_labels = labels.loc[validation_mask].to_numpy()
    if not np.any(validation_labels == 0):
        raise ValueError("Validation period must include benign user-hour rows")
    test_labels = labels.loc[test_mask].to_numpy()
    if not np.any(test_labels == 0) or not np.any(test_labels == 1):
        raise ValueError(
            "Test period must include both benign and red-team user-hour rows"
        )

    train = _stable_training_sample(
        frame,
        train_mask,
        max_train_rows,
    )
    model = _fit_isolation_forest(train, n_estimators, seed)

    validation = frame.loc[validation_mask]
    validation_scores = -model.score_samples(
        transform_login_features(validation)
    )
    test = frame.loc[test_mask]
    test_scores = -model.score_samples(transform_login_features(test))
    return _build_report_and_artifact(
        model,
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
        fit_end_hour,
        validation_end_hour,
        target_fpr,
        int(train_mask.sum()),
        len(train),
        seed,
    )


def train_login_anomaly_from_parquet(
    input_path,
    fit_end_hour=9 * 24,
    validation_end_hour=15 * 24,
    target_fpr=0.01,
    max_train_rows=200_000,
    n_estimators=200,
    seed=42,
    batch_size=100_000,
    memory_limit="4GB",
):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Feature dataset does not exist: {input_path}")
    if not 0 < fit_end_hour < validation_end_hour:
        raise ValueError("Time split boundaries must satisfy 0 < fit < validation")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    if max_train_rows <= 0 or n_estimators <= 0 or batch_size <= 0:
        raise ValueError("Training rows, estimator count, and batch size must be positive")

    required = {
        "user_id",
        "hour_index",
        "redteam_event_count",
        *LOGIN_FEATURES,
    }
    with TemporaryDirectory(prefix="lanl_login_detector_") as temporary_directory:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET memory_limit = ?", [memory_limit])
            connection.execute("SET threads = 2")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(
                "SET temp_directory = ?",
                [str(Path(temporary_directory).resolve())],
            )
            connection.execute("SET max_temp_directory_size = '100GB'")

            columns = {
                row[0]
                for row in connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(input_path)],
                ).fetchall()
            }
            missing = required - columns
            if missing:
                raise ValueError(
                    f"Login feature data is missing columns: {sorted(missing)}"
                )

            counts = connection.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE hour_index < ? AND redteam_event_count = 0
                    ),
                    COUNT(*) FILTER (WHERE hour_index < ?),
                    COUNT(*) FILTER (
                        WHERE hour_index >= ? AND hour_index < ?
                    ),
                    COUNT(*) FILTER (
                        WHERE hour_index >= ? AND hour_index < ?
                          AND redteam_event_count > 0
                    ),
                    COUNT(*) FILTER (
                        WHERE hour_index >= ? AND hour_index < ?
                          AND redteam_event_count = 0
                    ),
                    COUNT(*) FILTER (WHERE hour_index >= ?),
                    COUNT(*) FILTER (
                        WHERE hour_index >= ? AND redteam_event_count > 0
                    ),
                    COUNT(*) FILTER (
                        WHERE hour_index >= ? AND redteam_event_count = 0
                    )
                FROM read_parquet(?)
                """,
                [
                    fit_end_hour,
                    fit_end_hour,
                    fit_end_hour,
                    validation_end_hour,
                    fit_end_hour,
                    validation_end_hour,
                    fit_end_hour,
                    validation_end_hour,
                    validation_end_hour,
                    validation_end_hour,
                    validation_end_hour,
                    str(input_path),
                ],
            ).fetchone()
            (
                training_rows_available,
                fit_rows,
                validation_rows,
                validation_positive,
                validation_benign,
                test_rows,
                test_positive,
                test_benign,
            ) = map(int, counts)
            if not training_rows_available:
                raise ValueError("Fit period contains no benign user-hour rows")
            if not validation_rows or not validation_benign:
                raise ValueError("Validation period must include benign user-hour rows")
            if not test_benign or not test_positive:
                raise ValueError(
                    "Test period must include both benign and red-team user-hour rows"
                )

            feature_sql = ", ".join(f'"{name}"' for name in LOGIN_FEATURES)
            training = connection.execute(
                f"""
                SELECT {feature_sql}
                FROM read_parquet(?)
                WHERE hour_index < ? AND redteam_event_count = 0
                ORDER BY hash(user_id, hour_index)
                LIMIT ?
                """,
                [str(input_path), fit_end_hour, max_train_rows],
            ).fetch_df()
            model = _fit_isolation_forest(training, n_estimators, seed)

            def score_period(where_clause, parameters):
                result = connection.execute(
                    f"""
                    SELECT {feature_sql}, redteam_event_count
                    FROM read_parquet(?)
                    WHERE {where_clause}
                    """,
                    [str(input_path), *parameters],
                )
                labels_parts = []
                scores_parts = []
                while True:
                    batch = result.fetch_df_chunk(batch_size)
                    if batch.empty:
                        break
                    labels_parts.append(
                        (batch["redteam_event_count"].to_numpy() > 0).astype(
                            np.int8
                        )
                    )
                    scores_parts.append(
                        -model.score_samples(transform_login_features(batch))
                    )
                if not labels_parts:
                    return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float64)
                return np.concatenate(labels_parts), np.concatenate(scores_parts)

            validation_labels, validation_scores = score_period(
                "hour_index >= ? AND hour_index < ?",
                [fit_end_hour, validation_end_hour],
            )
            test_labels, test_scores = score_period(
                "hour_index >= ?",
                [validation_end_hour],
            )
        finally:
            connection.close()

    report, artifact = _build_report_and_artifact(
        model,
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
        fit_end_hour,
        validation_end_hour,
        target_fpr,
        training_rows_available,
        len(training),
        seed,
    )
    report["method"]["fit_rows_in_period"] = int(fit_rows)
    report["method"]["validation_rows_in_period"] = validation_rows
    report["method"]["test_rows_in_period"] = test_rows
    report["method"]["validation_redteam_user_hours"] = validation_positive
    report["method"]["test_redteam_user_hours"] = test_positive
    report["method"]["evaluation_batch_size"] = int(batch_size)
    return report, artifact


def score_login_features(frame, artifact):
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported login model artifact version")
    if artifact.get("task") != "login_anomaly":
        raise ValueError("Model artifact is not a login anomaly detector")
    if artifact.get("feature_names") != list(LOGIN_FEATURES):
        raise ValueError("Login model artifact feature schema does not match")
    if not hasattr(artifact.get("model"), "score_samples"):
        raise ValueError("Login model artifact does not support anomaly scoring")
    threshold = artifact.get("threshold")
    if not isinstance(threshold, (float, int, np.floating, np.integer)) or not np.isfinite(
        threshold
    ):
        raise ValueError("Login model artifact threshold must be finite")
    features = transform_login_features(frame)
    scores = -artifact["model"].score_samples(features)
    if not np.isfinite(scores).all():
        raise ValueError("Login anomaly detector produced non-finite scores")
    return pd.DataFrame({
        "anomaly_score": scores,
        "is_alert": scores > threshold,
    }, index=frame.index)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train an Isolation Forest login anomaly detector using "
            "chronological LANL user-hour partitions."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/lanl_login_features.parquet",
        help="Per-user-hour feature Parquet produced by ml.prepare_login_data.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/lanl_login_features_manifest.json",
        help="Preprocessing manifest used to verify feature data and label coverage.",
    )
    parser.add_argument(
        "--model-out",
        default="ml/models/lanl_login_anomaly.joblib",
        help="Saved model artifact path.",
    )
    parser.add_argument(
        "--report-out",
        default="evaluation/results/lanl_login_anomaly.json",
        help="JSON evaluation report path.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    if not 0 <= args.target_fpr < 1:
        parser.error("--target-fpr must be in [0, 1)")

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Feature dataset does not exist: {input_path}")
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        parser.error(f"Preprocessing manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_hash = _sha256_file(input_path)
    if manifest.get("output_sha256", manifest.get("feature_file_sha256")) != feature_hash:
        parser.error("Feature Parquet hash does not match the preprocessing manifest")
    report, artifact = train_login_anomaly_from_parquet(
        input_path,
        target_fpr=args.target_fpr,
        max_train_rows=args.max_train_rows,
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
    )
    report["source"] = {
        "feature_file": str(input_path),
        "feature_file_sha256": feature_hash,
        "preprocessing_manifest": str(manifest_path),
        "preprocessing_manifest_sha256": _sha256_file(manifest_path),
        "auth_rows": manifest.get(
            "raw_auth_rows_processed", manifest.get("auth_rows_validated")
        ),
        "redteam_rows": manifest.get("redteam_rows"),
        "unique_redteam_labels": manifest.get("unique_redteam_labels"),
        "matched_redteam_auth_events": manifest.get(
            "matched_redteam_auth_events"
        ),
        "matched_unique_redteam_labels": manifest.get(
            "matched_unique_redteam_labels"
        ),
        "unmatched_unique_redteam_labels": manifest.get(
            "unmatched_unique_redteam_labels"
        ),
    }
    model_path = Path(args.model_out)
    report_path = Path(args.report_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report["source"]["model_file"] = str(model_path)
    report["source"]["model_file_sha256"] = _sha256_file(model_path)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved login detector to {model_path}")
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
