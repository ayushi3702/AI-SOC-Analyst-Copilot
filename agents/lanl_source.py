"""Read-only search over generated LANL user-hour authentication features."""

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from agents.log_hunter import MAX_RESULTS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "data" / "processed" / "lanl_login_features.parquet"


@dataclass(frozen=True)
class LoginIncidentContext:
    incident_id: str
    user_id: str
    start_hour: int
    end_hour: int

    def __post_init__(self):
        if not isinstance(self.incident_id, str) or not self.incident_id.strip():
            raise ValueError("incident_id must be a non-empty string")
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        for field in ("start_hour", "end_hour"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer hour index")
        if self.start_hour > self.end_hour:
            raise ValueError("start_hour must not be after end_hour")


@dataclass(frozen=True)
class LoginEvidenceRecord:
    evidence_id: str
    user_id: str
    hour_index: int
    event_count: int
    success_count: int
    failure_count: int
    previous_24h_events: int
    previous_24h_failures: int
    previous_24h_successes: int
    hours_since_previous_activity: float | None
    source: str = "lanl_user_hour_features"


class LanlFeatureHunter:
    """Search only precomputed per-user/hour rows; never parses raw auth.txt."""

    def __init__(self, feature_path=DEFAULT_FEATURES):
        self.feature_path = Path(feature_path)
        if not self.feature_path.is_file():
            raise FileNotFoundError(
                f"LANL feature Parquet not found: {self.feature_path}. "
                "Run `python -m ml.prepare_login_data` first."
            )

    def search(
        self, incident: LoginIncidentContext, limit: int = 100
    ) -> list[LoginEvidenceRecord]:
        if not isinstance(incident, LoginIncidentContext):
            raise TypeError("incident must be a LoginIncidentContext")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

        with duckdb.connect(":memory:") as connection:
            rows = connection.execute(
                """
                SELECT user_id, hour_index, event_count, success_count,
                       failure_count, previous_24h_events,
                       previous_24h_failures, previous_24h_successes,
                       hours_since_previous_activity
                FROM read_parquet(?)
                WHERE user_id = ?
                  AND hour_index BETWEEN ? AND ?
                ORDER BY hour_index
                LIMIT ?
                """,
                [
                    str(self.feature_path.resolve()),
                    incident.user_id,
                    incident.start_hour,
                    incident.end_hour,
                    limit,
                ],
            ).fetchall()
        return [
            LoginEvidenceRecord(
                evidence_id=f"lanl:{row[0]}:{row[1]}",
                user_id=row[0],
                hour_index=row[1],
                event_count=row[2],
                success_count=row[3],
                failure_count=row[4],
                previous_24h_events=row[5],
                previous_24h_failures=row[6],
                previous_24h_successes=row[7],
                hours_since_previous_activity=row[8],
            )
            for row in rows
        ]


def serialize_login_evidence(records: list[LoginEvidenceRecord]) -> list[dict]:
    return [asdict(record) for record in records]
