import os
from typing import Optional

from sqlalchemy import create_engine, text

DATABASE_URL = (
    f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'nids_admin')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'change_me')}@"
    f"{os.environ.get('POSTGRES_HOST', 'postgres')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'network_intrusion_detection')}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)

FLOW_COLUMNS = """
    flow_id, dataset_source, split, source_day, protocol, duration,
    bytes_sent, bytes_received, is_attack_actual, attack_category_actual,
    predicted_label, attack_probability, rule_flags, raw_features, scored_at
"""


def fetch_flows(
    limit: int = 50,
    offset: int = 0,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    predicted_label: Optional[int] = None,
    attack_category: Optional[str] = None,
):
    clauses = []
    params = {"limit": limit, "offset": offset}
    if dataset:
        clauses.append("dataset_source = :dataset")
        params["dataset"] = dataset
    if split:
        clauses.append("split = :split")
        params["split"] = split
    if predicted_label is not None:
        clauses.append("predicted_label = :predicted_label")
        params["predicted_label"] = predicted_label
    if attack_category:
        clauses.append("attack_category_actual = :attack_category")
        params["attack_category"] = attack_category

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = text(
        f"""
        SELECT {FLOW_COLUMNS}
        FROM flows_scored
        {where_sql}
        ORDER BY scored_at DESC, flow_id
        LIMIT :limit OFFSET :offset
        """
    )
    count_query = text(f"SELECT count(*) FROM flows_scored {where_sql}")

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query, params)]
        total = conn.execute(count_query, params).scalar_one()
    return rows, total


def fetch_flow_by_id(flow_id: str):
    query = text(f"SELECT {FLOW_COLUMNS} FROM flows_scored WHERE flow_id = :flow_id LIMIT 1")
    with engine.connect() as conn:
        row = conn.execute(query, {"flow_id": flow_id}).first()
    return dict(row._mapping) if row else None


def fetch_summary_stats(dataset: str):
    query = text(
        """
        SELECT
            count(*) AS total_flows,
            sum(is_attack_actual) AS actual_attack_count,
            sum(predicted_label) AS predicted_attack_count,
            avg(attack_probability) AS avg_attack_probability
        FROM flows_scored
        WHERE dataset_source = :dataset
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"dataset": dataset}).first()
    return dict(row._mapping) if row else {}


def fetch_dataset_stats():
    query = text(
        """
        SELECT dataset_source, split, total_flows, actual_attack_count,
               predicted_attack_count, avg_attack_probability
        FROM dataset_stats
        ORDER BY dataset_source, split
        """
    )
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query)]
    return rows


def fetch_category_breakdown(dataset: Optional[str] = None):
    where_sql = "WHERE dataset_source = :dataset" if dataset else ""
    query = text(
        f"""
        SELECT attack_category_actual AS attack_category,
               count(*) AS total_flows,
               sum(predicted_label) AS predicted_attack_count
        FROM flows_scored
        {where_sql}
        GROUP BY attack_category_actual
        ORDER BY predicted_attack_count DESC
        """
    )
    params = {"dataset": dataset} if dataset else {}
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(query, params)]
    return rows
