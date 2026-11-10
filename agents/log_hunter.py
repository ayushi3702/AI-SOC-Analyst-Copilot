"""Read-only, bounded searches over normalized network-flow events."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "soc.duckdb"
MAX_RESULTS = 500


@dataclass(frozen=True)
class EventSearchFilters:
    src_ip: str | None = None
    dst_ip: str | None = None
    source_port: int | None = None
    destination_port: int | None = None
    protocol: str | None = None

    def __post_init__(self):
        for field_name in ("src_ip", "dst_ip", "protocol"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("source_port", "destination_port"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 65535
            ):
                raise ValueError(f"{field_name} must be an integer from 0 to 65535")


@dataclass(frozen=True)
class IncidentContext:
    """Minimal incident query context using the source's recorded timestamps."""

    incident_id: str
    entity: str
    start_time: datetime
    end_time: datetime

    def __post_init__(self):
        if not isinstance(self.incident_id, str):
            raise TypeError("incident_id must be a string")
        if not isinstance(self.entity, str):
            raise TypeError("entity must be a string")
        if not self.incident_id.strip():
            raise ValueError("incident_id must not be empty")
        if not self.entity.strip():
            raise ValueError("entity must not be empty")
        for field_name in ("start_time", "end_time"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime):
                raise TypeError(f"{field_name} must be a datetime")
            if value.tzinfo is not None and value.utcoffset() is not None:
                raise ValueError(
                    f"{field_name} must be timezone-naive to match the "
                    "CICIDS2017 timestamp column"
                )
        if self.start_time > self.end_time:
            raise ValueError("start_time must not be later than end_time")


@dataclass(frozen=True)
class EvidenceRecord:
    """A returned event with a stable citation ID and source fields."""

    evidence_id: str
    event_id: str
    observed_at: str
    source_dataset: str
    event_type: str
    src_ip: str
    src_port: int | None
    dst_ip: str
    dst_port: int | None
    protocol: str
    duration_us: int | None
    fwd_packets: int | None
    bwd_packets: int | None
    fwd_bytes: float | None
    bwd_bytes: float | None


class DuckDBLogHunter:
    """Search only the normalized ``events`` table through a read-only handle."""

    def __init__(self, database_path=DEFAULT_DATABASE):
        self.database_path = Path(database_path)
        if not self.database_path.is_file():
            raise FileNotFoundError(
                f"DuckDB database does not exist: {self.database_path}"
            )

    def search(
        self,
        incident: IncidentContext,
        limit: int = 100,
        filters: EventSearchFilters | None = None,
    ) -> list[EvidenceRecord]:
        """Find events involving an exact IP in the inclusive time window."""
        if not isinstance(incident, IncidentContext):
            raise TypeError("incident must be an IncidentContext")
        if filters is None:
            filters = EventSearchFilters()
        if not isinstance(filters, EventSearchFilters):
            raise TypeError("filters must be EventSearchFilters")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 1 or limit > MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

        predicates = [
            "timestamp >= ?",
            "timestamp <= ?",
            "(src_ip = ? OR dst_ip = ?)",
        ]
        parameters = [
            incident.start_time,
            incident.end_time,
            incident.entity,
            incident.entity,
        ]
        for column, value in (
            ("src_ip", filters.src_ip),
            ("dst_ip", filters.dst_ip),
            ("src_port", filters.source_port),
            ("dst_port", filters.destination_port),
            ("protocol", filters.protocol),
        ):
            if value is not None:
                predicates.append(f"{column} = ?")
                parameters.append(value)
        parameters.append(limit)

        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            rows = connection.execute(
                f"""
                SELECT event_id, timestamp, source_dataset, event_type,
                       src_ip, src_port, dst_ip, dst_port, protocol,
                       duration_us, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes
                FROM events
                WHERE {" AND ".join(predicates)}
                ORDER BY timestamp, event_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [
            EvidenceRecord(
                evidence_id=f"event:{row[0]}",
                event_id=row[0],
                observed_at=row[1].isoformat(sep=" "),
                source_dataset=row[2],
                event_type=row[3],
                src_ip=row[4],
                src_port=row[5],
                dst_ip=row[6],
                dst_port=row[7],
                protocol=row[8],
                duration_us=row[9],
                fwd_packets=row[10],
                bwd_packets=row[11],
                fwd_bytes=row[12],
                bwd_bytes=row[13],
            )
            for row in rows
        ]


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "timestamp must be ISO 8601, for example 2017-03-07T08:55:58"
        ) from error


def main():
    parser = argparse.ArgumentParser(
        description="Search normalized events read-only for an incident entity."
    )
    parser.add_argument("--incident-id", required=True)
    parser.add_argument(
        "--entity",
        required=True,
        help="Exact IP address to match against src_ip or dst_ip.",
    )
    parser.add_argument("--start", required=True, type=_parse_timestamp)
    parser.add_argument("--end", required=True, type=_parse_timestamp)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--src-ip")
    parser.add_argument("--dst-ip")
    parser.add_argument("--source-port", type=int)
    parser.add_argument("--destination-port", type=int)
    parser.add_argument("--protocol")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()

    incident = IncidentContext(
        incident_id=args.incident_id,
        entity=args.entity,
        start_time=args.start,
        end_time=args.end,
    )
    filters = EventSearchFilters(
        src_ip=args.src_ip,
        dst_ip=args.dst_ip,
        source_port=args.source_port,
        destination_port=args.destination_port,
        protocol=args.protocol,
    )
    evidence = DuckDBLogHunter(args.db).search(
        incident, limit=args.limit, filters=filters
    )
    print(json.dumps({
        "incident": {
            "incident_id": incident.incident_id,
            "entity": incident.entity,
            "start_time": incident.start_time.isoformat(sep=" "),
            "end_time": incident.end_time.isoformat(sep=" "),
        },
        "evidence": [asdict(record) for record in evidence],
        "evidence_count": len(evidence),
        "maximum_results": args.limit,
    }, indent=2))


if __name__ == "__main__":
    main()
