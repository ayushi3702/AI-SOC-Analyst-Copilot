"""
Load all cleaned Parquet files into one DuckDB database (data/soc.duckdb).
Processes one file at a time to keep RAM low.

Run from the repo root:
    python -m ingest.to_duckdb
    python -m ingest.to_duckdb --force     # delete and rebuild the database
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

from ingest.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
DB_PATH = ROOT / "data" / "soc.duckdb"


def chronological(files: list[Path]) -> list[Path]:
    """Order files by their earliest timestamp, so the events table is roughly time-sorted."""
    starts = {f: pd.read_parquet(f, columns=["timestamp"])["timestamp"].min() for f in files}
    return sorted(files, key=lambda f: starts[f])


def main(interim: Path, db_path: Path, force: bool) -> None:
    files = sorted(interim.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files in {interim}. Run: python -m ingest.load_cicids")
    if force and db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='2GB'")  # safe for an 8 GB laptop
    existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if existing:
        raise SystemExit(f"{db_path.name} already has tables {sorted(existing)}. Use --force to rebuild.")

    first = True
    for f in chronological(files):
        print(f"[load] {f.name}")
        tables = normalize(pd.read_parquet(f), file_stem=f.stem)
        # One transaction per file: if anything fails, nothing from this file is kept,
        # so the three tables can never end up out of sync.
        con.execute("BEGIN TRANSACTION")
        try:
            for name, tdf in tables.items():
                con.register("tmp_df", tdf)
                if first:
                    con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
                else:
                    con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM tmp_df")
                con.unregister("tmp_df")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        first = False

    print("\nRow counts:")
    for t in ("events", "labels", "flow_features"):
        print(f"  {t:14s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:,}")

    print("\nAttacks per day:")
    print(con.execute("""
        SELECT day, COUNT(*) AS flows, SUM(is_attack) AS attacks
        FROM labels GROUP BY day ORDER BY MIN(source_file)
    """).df().to_string(index=False))

    print("\nTop talkers (the kind of query the log-hunter agent will run):")
    print(con.execute("""
        SELECT src_ip, COUNT(*) AS flows FROM events
        GROUP BY src_ip ORDER BY flows DESC LIMIT 5
    """).df().to_string(index=False))
    con.close()
    print(f"\nSaved {db_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", type=Path, default=INTERIM)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.interim, a.db, a.force)
