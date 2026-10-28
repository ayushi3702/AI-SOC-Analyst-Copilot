import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import duckdb
import numpy as np
import pandas as pd


AUTH_COLUMNS = {
    "epoch": "BIGINT",
    "source_user": "VARCHAR",
    "destination_user": "VARCHAR",
    "source_computer": "VARCHAR",
    "destination_computer": "VARCHAR",
    "auth_type": "VARCHAR",
    "logon_type": "VARCHAR",
    "orientation": "VARCHAR",
    "outcome": "VARCHAR",
}
REDTEAM_COLUMNS = {
    "epoch": "BIGINT",
    "source_user": "VARCHAR",
    "source_computer": "VARCHAR",
    "destination_computer": "VARCHAR",
}
GROUP_COLUMNS = [
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
    "redteam_event_count",
]


def _sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_login_features(
    auth_path,
    redteam_path,
    output_path,
    memory_limit="4GB",
    threads=2,
):
    """Aggregate LANL authentication events into labeled user-hour features."""
    auth_path = Path(auth_path)
    redteam_path = Path(redteam_path)
    output_path = Path(output_path)
    for path in (auth_path, redteam_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input telemetry file does not exist: {path}")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if threads <= 0:
        raise ValueError("threads must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    redteam = pd.read_csv(
        redteam_path,
        header=None,
        names=list(REDTEAM_COLUMNS),
        dtype={
            "epoch": "int64",
            "source_user": "string",
            "source_computer": "string",
            "destination_computer": "string",
        },
        keep_default_na=False,
    ).drop_duplicates()
    if redteam.empty:
        raise ValueError("Red-team file contains no valid event labels")
    label_keys = list(REDTEAM_COLUMNS)
    redteam_index = pd.MultiIndex.from_frame(redteam[label_keys])
    redteam_epoch_values = set(redteam["epoch"].unique())
    matched_labels = set()
    matched_events = 0
    processed_auth_events = 0
    missing_user_events = 0
    chunk_paths = []
    started_at = time.monotonic()

    with TemporaryDirectory(
        prefix="lanl_login_aggregate_", dir=str(output_path.parent)
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        reader = pd.read_csv(
            auth_path,
            header=None,
            names=list(AUTH_COLUMNS),
            usecols=list(AUTH_COLUMNS),
            dtype={
                "epoch": "int64",
                "source_user": "string",
                "destination_user": "string",
                "source_computer": "string",
                "destination_computer": "string",
                "auth_type": "string",
                "logon_type": "string",
                "orientation": "string",
                "outcome": "string",
            },
            keep_default_na=False,
            chunksize=2_000_000,
            engine="c",
            on_bad_lines="error",
        )
        for chunk_number, chunk in enumerate(reader):
            processed_auth_events += len(chunk)
            candidate_mask = chunk["epoch"].isin(redteam_epoch_values).to_numpy()
            candidate_positions = np.flatnonzero(candidate_mask)
            is_redteam = np.zeros(len(chunk), dtype=np.int64)
            if len(candidate_positions):
                candidates = chunk.iloc[candidate_positions][label_keys]
                candidate_index = pd.MultiIndex.from_frame(candidates)
                matched_mask = candidate_index.isin(redteam_index)
                matched_positions = candidate_positions[matched_mask]
                is_redteam[matched_positions] = 1
                matched_events += int(matched_mask.sum())
                matched_labels.update(
                    map(
                        tuple,
                        candidates.loc[matched_mask]
                        .drop_duplicates()
                        .itertuples(index=False, name=None),
                    )
                )

            chunk["hour_index"] = chunk["epoch"] // 3600
            chunk["event_count"] = 1
            chunk["success_count"] = (chunk["outcome"] == "Success").astype("int64")
            chunk["failure_count"] = (chunk["outcome"] == "Fail").astype("int64")
            chunk["source_computer_events"] = (
                chunk["source_computer"] != "?"
            ).astype("int64")
            chunk["destination_computer_events"] = (
                chunk["destination_computer"] != "?"
            ).astype("int64")
            chunk["kerberos_events"] = (chunk["auth_type"] == "Kerberos").astype(
                "int64"
            )
            chunk["ntlm_events"] = (chunk["auth_type"] == "NTLM").astype("int64")
            chunk["negotiate_events"] = (
                chunk["auth_type"] == "Negotiate"
            ).astype("int64")
            chunk["network_events"] = (chunk["logon_type"] == "Network").astype(
                "int64"
            )
            chunk["interactive_events"] = chunk["logon_type"].isin(
                ["Interactive", "RemoteInteractive"]
            ).astype("int64")
            chunk["service_events"] = (chunk["logon_type"] == "Service").astype(
                "int64"
            )
            chunk["batch_events"] = (chunk["logon_type"] == "Batch").astype(
                "int64"
            )
            chunk["logon_events"] = (chunk["orientation"] == "LogOn").astype(
                "int64"
            )
            chunk["logoff_events"] = (chunk["orientation"] == "LogOff").astype(
                "int64"
            )
            chunk["redteam_event_count"] = is_redteam

            missing_user_events += int((chunk["source_user"] == "?").sum())
            valid_chunk = chunk.loc[chunk["source_user"] != "?"]
            partial = (
                valid_chunk.groupby(
                    ["source_user", "hour_index"],
                    sort=False,
                    dropna=False,
                    observed=True,
                )[GROUP_COLUMNS]
                .sum()
                .reset_index()
                .rename(columns={"source_user": "user_id"})
            )
            part_path = temporary_path / f"part-{chunk_number:05d}.parquet"
            partial.to_parquet(part_path, index=False, compression="zstd")
            chunk_paths.append(str(part_path))
            if (chunk_number + 1) % 25 == 0:
                elapsed = time.monotonic() - started_at
                print(
                    f"Processed {processed_auth_events:,} authentication "
                    f"events in {elapsed / 60:.1f} minutes.",
                    flush=True,
                )

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET memory_limit = ?", [memory_limit])
            connection.execute("SET threads = ?", [threads])
            connection.execute("SET preserve_insertion_order = false")
            spill_path = temporary_path / "duckdb_spill"
            spill_path.mkdir()
            connection.execute(
                "SET temp_directory = ?", [str(spill_path.resolve())]
            )
            connection.execute("SET max_temp_directory_size = '100GB'")
            connection.read_parquet(chunk_paths).create_view("partial_hourly")
            connection.execute("""
                CREATE TEMP VIEW hourly AS
                SELECT user_id, hour_index,
                       SUM(event_count)::BIGINT AS event_count,
                       SUM(success_count)::BIGINT AS success_count,
                       SUM(failure_count)::BIGINT AS failure_count,
                       SUM(source_computer_events)::BIGINT
                           AS source_computer_events,
                       SUM(destination_computer_events)::BIGINT
                           AS destination_computer_events,
                       SUM(kerberos_events)::BIGINT AS kerberos_events,
                       SUM(ntlm_events)::BIGINT AS ntlm_events,
                       SUM(negotiate_events)::BIGINT AS negotiate_events,
                       SUM(network_events)::BIGINT AS network_events,
                       SUM(interactive_events)::BIGINT AS interactive_events,
                       SUM(service_events)::BIGINT AS service_events,
                       SUM(batch_events)::BIGINT AS batch_events,
                       SUM(logon_events)::BIGINT AS logon_events,
                       SUM(logoff_events)::BIGINT AS logoff_events,
                       SUM(redteam_event_count)::BIGINT AS redteam_event_count
                FROM partial_hourly
                GROUP BY user_id, hour_index
            """)

            query = """
                WITH context AS (
                    SELECT
                        *,
                        SUM(event_count) OVER prior_24_hours
                            AS previous_24h_events,
                        SUM(failure_count) OVER prior_24_hours
                            AS previous_24h_failures,
                        SUM(success_count) OVER prior_24_hours
                            AS previous_24h_successes,
                        LAG(hour_index) OVER (
                            PARTITION BY user_id ORDER BY hour_index
                        ) AS previous_active_hour
                    FROM hourly
                    WINDOW prior_24_hours AS (
                        PARTITION BY user_id
                        ORDER BY hour_index
                        RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
                    )
                )
                SELECT
                    user_id,
                    hour_index,
                    event_count,
                    success_count,
                    failure_count,
                    source_computer_events,
                    destination_computer_events,
                    kerberos_events,
                    ntlm_events,
                    negotiate_events,
                    network_events,
                    interactive_events,
                    service_events,
                    batch_events,
                    logon_events,
                    logoff_events,
                    COALESCE(previous_24h_events, 0)::BIGINT
                        AS previous_24h_events,
                    COALESCE(previous_24h_failures, 0)::BIGINT
                        AS previous_24h_failures,
                    COALESCE(previous_24h_successes, 0)::BIGINT
                        AS previous_24h_successes,
                    CASE
                        WHEN previous_active_hour IS NULL THEN NULL
                        ELSE hour_index - previous_active_hour
                    END AS hours_since_previous_activity,
                    hour_index % 24 AS relative_hour_in_day,
                    SIN(2 * PI() * (hour_index % 24) / 24.0) AS hour_sin,
                    COS(2 * PI() * (hour_index % 24) / 24.0) AS hour_cos,
                    redteam_event_count
                FROM context
            """
            copy_sql = (
                f"COPY ({query}) TO {_sql_string(output_path)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.execute(copy_sql)
            row_count, event_count, matched_output_events = connection.execute(
                f"""
                SELECT COUNT(*), SUM(event_count), SUM(redteam_event_count)
                FROM read_parquet({_sql_string(output_path)})
                """
            ).fetchone()
            if event_count != processed_auth_events - missing_user_events:
                raise ValueError(
                    "Aggregated event count does not match valid auth rows"
                )
            if matched_output_events != matched_events:
                raise ValueError(
                    "Aggregated red-team count does not match exact auth matches"
                )
        except Exception:
            if output_path.exists():
                output_path.unlink()
            raise
        finally:
            connection.close()

    unique_redteam_labels = len(redteam_index)
    return {
        "auth_file": str(auth_path),
        "auth_bytes": auth_path.stat().st_size,
        "redteam_file": str(redteam_path),
        "redteam_rows": int(len(pd.read_csv(redteam_path, header=None))),
        "unique_redteam_labels": int(unique_redteam_labels),
        "raw_auth_rows_processed": int(processed_auth_events),
        "matched_redteam_auth_events": int(matched_events),
        "matched_unique_redteam_labels": int(len(matched_labels)),
        "unmatched_unique_redteam_labels": int(
            unique_redteam_labels - len(matched_labels)
        ),
        "auth_events_aggregated": int(event_count or 0),
        "auth_events_without_source_user": int(missing_user_events),
        "user_hour_rows": int(row_count),
        "output_file": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256_file(output_path),
        "label_match_key": (
            "epoch, source_user, source_computer, destination_computer"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate LANL authentication telemetry into per-user hourly "
            "login features with exact red-team labels."
        )
    )
    parser.add_argument(
        "--auth",
        default="data/raw/lanl/auth.txt",
        help="Path to the LANL nine-column auth.txt file.",
    )
    parser.add_argument(
        "--redteam",
        default="data/raw/lanl/redteam.txt",
        help="Path to the LANL four-column redteam.txt file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lanl_login_features.parquet",
        help="Output Parquet path; existing files are not overwritten.",
    )
    parser.add_argument(
        "--summary-out",
        default="data/processed/lanl_login_features_manifest.json",
        help="JSON file recording data validation and aggregation counts.",
    )
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    summary = prepare_login_features(
        args.auth,
        args.redteam,
        args.output,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved preprocessing summary to {summary_path}")


if __name__ == "__main__":
    main()
