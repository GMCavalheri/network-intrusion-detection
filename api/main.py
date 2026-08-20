import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import db
import inference
from constants import (
    CICIDS2017_FORM_FIELDS, DATASETS, NSL_KDD_FLAGS, NSL_KDD_FORM_FIELDS,
    NSL_KDD_PROTOCOL_TYPES, NSL_KDD_SERVICES, UNIFIED_CATEGORIES,
)
from logging_config import RequestLoggingMiddleware, configure_logging
from schemas import (
    CategoryBreakdown, DatasetStat, Flow, FlowsPage, ScoreRequest, ScoreResponse, SummaryStats,
)

PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "/opt/data/processed")
METRICS_PATHS = {
    "nsl_kdd": os.path.join(PROCESSED_DATA_DIR, "nsl_kdd_metrics.json"),
    "cicids2017": os.path.join(PROCESSED_DATA_DIR, "cicids2017_metrics.json"),
}
DQ_REPORT_PATHS = {
    "nsl_kdd": os.path.join(PROCESSED_DATA_DIR, "nsl_kdd_data_quality_report.json"),
    "cicids2017": os.path.join(PROCESSED_DATA_DIR, "cicids2017_data_quality_report.json"),
}

logger = configure_logging()

app = FastAPI(
    title="Network Intrusion Detection API",
    description="Serves Spark-cleaned, MLlib-scored NSL-KDD/CICIDS2017 flows and live intrusion scoring.",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware, logger=logger)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_dataset(dataset: str):
    if dataset not in DATASETS:
        raise HTTPException(status_code=422, detail=f"dataset must be one of {DATASETS}")


def _load_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return {
        "datasets": DATASETS,
        "attack_categories": UNIFIED_CATEGORIES,
        "nsl_kdd": {
            "form_fields": NSL_KDD_FORM_FIELDS,
            "protocol_types": NSL_KDD_PROTOCOL_TYPES,
            "services": NSL_KDD_SERVICES,
            "flags": NSL_KDD_FLAGS,
        },
        "cicids2017": {"form_fields": CICIDS2017_FORM_FIELDS},
    }


@app.get("/flows", response_model=FlowsPage)
def list_flows(
    limit: int = Query(50, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    predicted_label: Optional[int] = Query(None, ge=0, le=1),
    attack_category: Optional[str] = None,
):
    if dataset:
        _validate_dataset(dataset)
    rows, total = db.fetch_flows(
        limit=limit, offset=offset, dataset=dataset, split=split,
        predicted_label=predicted_label, attack_category=attack_category,
    )
    return {"total": total, "limit": limit, "offset": offset, "items": rows}


@app.get("/flows/{flow_id}", response_model=Flow)
def get_flow(flow_id: str):
    row = db.fetch_flow_by_id(flow_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Flow not found")
    return row


@app.get("/stats/summary", response_model=SummaryStats)
def stats_summary(dataset: str):
    _validate_dataset(dataset)
    stats = db.fetch_summary_stats(dataset)
    if not stats or not stats.get("total_flows"):
        raise HTTPException(status_code=404, detail="No scored data yet for this dataset - has the Spark pipeline run?")
    stats["dataset_source"] = dataset
    stats["model_metrics"] = _load_json(METRICS_PATHS[dataset])
    return stats


@app.get("/stats/datasets", response_model=list[DatasetStat])
def stats_datasets():
    return db.fetch_dataset_stats()


@app.get("/stats/categories", response_model=list[CategoryBreakdown])
def stats_categories(dataset: Optional[str] = None):
    if dataset:
        _validate_dataset(dataset)
    return db.fetch_category_breakdown(dataset)


@app.get("/reports/data-quality")
def data_quality_report(dataset: str):
    _validate_dataset(dataset)
    report = _load_json(DQ_REPORT_PATHS[dataset])
    if report is None:
        raise HTTPException(status_code=404, detail="Data quality report not found - has the etl_clean step run?")
    return report


@app.post("/score", response_model=ScoreResponse)
def score_flow(req: ScoreRequest):
    try:
        return inference.score(req.dataset, req.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # model/spark not ready, etc.
        logger.exception("Scoring failed for dataset=%s", req.dataset)
        raise HTTPException(status_code=503, detail=f"Scoring unavailable: {exc}")
