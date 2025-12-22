import os
from datetime import datetime

import psycopg2
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator


def _db_dsn() -> str:
    """Return a psycopg2-compatible DSN from env."""
    url = os.getenv("AEROSTREAM_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Missing AEROSTREAM_DATABASE_URL (or DATABASE_URL) env var")

    # Accept SQLAlchemy-style URLs like: postgresql+psycopg2://...
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql://", 1)
        
        print("url")
    return url


def _connect_db():
    return psycopg2.connect(_db_dsn())


def ingest_tweets_from_api() -> dict:
    api_base = (os.getenv("AEROSTREAM_FASTAPI_URL") or os.getenv("FASTAPI_URL") or "http://fastapi:8000").rstrip("/")
    batch_size = int(os.getenv("AEROSTREAM_BATCH_SIZE", "10"))

    resp = requests.get(
        f"{api_base}/db/conn/create",
        params={"batch_size": batch_size},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def compute_and_persist_kpis() -> None:
    with _connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kpi_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    total_tweets BIGINT NOT NULL,
                    total_airlines BIGINT NOT NULL,
                    negative_pct DOUBLE PRECISION NOT NULL
                );
                """
            )

            cur.execute("SELECT COUNT(*) FROM tweets;")
            total_tweets = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(DISTINCT airline) FROM tweets;")
            total_airlines = int(cur.fetchone()[0])

            if total_tweets == 0:
                negative_pct = 0.0
            else:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(CASE WHEN prediction = 'negative' THEN 1 ELSE 0 END), 0)
                    FROM tweets;
                    """
                )
                negative_count = int(cur.fetchone()[0])
                negative_pct = (negative_count / total_tweets) * 100.0

            cur.execute(
                """
                INSERT INTO kpi_snapshots (total_tweets, total_airlines, negative_pct)
                VALUES (%s, %s, %s);
                """,
                (total_tweets, total_airlines, float(negative_pct)),
            )


default_args = {
    "owner": "aerostream",
    "retries": 0,
}


with DAG(
    dag_id="aerostream_minutely_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="*/1 * * * *",
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    tags=["aerostream"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_tweets_from_fastapi",
        python_callable=ingest_tweets_from_api,
    )

    kpis = PythonOperator(
        task_id="compute_kpis",
        python_callable=compute_and_persist_kpis,
    )

    ingest >> kpis
