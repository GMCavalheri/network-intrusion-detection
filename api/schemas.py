from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Flow(BaseModel):
    flow_id: str
    dataset_source: str
    split: str
    source_day: Optional[str] = None
    protocol: Optional[str] = None
    duration: Optional[float] = None
    bytes_sent: Optional[float] = None
    bytes_received: Optional[float] = None
    is_attack_actual: int
    attack_category_actual: str
    predicted_label: int
    attack_probability: float
    rule_flags: Optional[str] = None
    raw_features: Optional[dict] = None
    scored_at: Optional[datetime] = None


class FlowsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[Flow]


class DatasetStat(BaseModel):
    dataset_source: str
    split: str
    total_flows: int
    actual_attack_count: int
    predicted_attack_count: int
    avg_attack_probability: Optional[float]


class SummaryStats(BaseModel):
    dataset_source: str
    total_flows: int
    actual_attack_count: int
    predicted_attack_count: int
    avg_attack_probability: Optional[float]
    model_metrics: Optional[dict] = None


class CategoryBreakdown(BaseModel):
    attack_category: str
    total_flows: int
    predicted_attack_count: int


class ScoreRequest(BaseModel):
    dataset: str = Field(..., description="'nsl_kdd' or 'cicids2017' - selects which trained PipelineModel scores this flow")
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="A subset of that dataset's form fields (see /meta) - anything omitted defaults to a "
        "typical/unremarkable baseline value, same idea as leaving a form field blank.",
    )


class ScoreResponse(BaseModel):
    dataset: str
    attack_probability: float
    predicted_label: int
    rule_flags: List[str]
    features_used: dict
