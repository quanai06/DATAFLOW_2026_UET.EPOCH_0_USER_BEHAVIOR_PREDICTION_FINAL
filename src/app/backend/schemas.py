from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionFrequencyItem(BaseModel):
    action: str
    count: int


class InputSummary(BaseModel):
    sequence_length: int
    unique_action_count: int
    anchor_action: str
    rollback_3_count: int
    rollback_4_count: int
    action_frequency_top: list[ActionFrequencyItem]
    repeat_density: float | None = None
    transition_ratio: float | None = None
    entropy: float | None = None
    rare_action_ratio: float | None = None
    was_truncated_to_max_length: bool | None = None


class PredictedOutputs(BaseModel):
    attr_1: int
    attr_2: int
    attr_3: int
    attr_4: int
    attr_5: int
    attr_6: int


class SchedulerDecision(BaseModel):
    capacity_score: float
    completion_urgency_score: float
    capacity_band: str | None = None
    completion_urgency_band: str | None = None
    completion_window_days: int | None = None
    warehouse_stress_zone: str | None = None
    today_production_pct: float
    warehouse_waiting_pressure_pct: float
    priority_level: Literal["HIGH", "MEDIUM", "LOW"]
    recommended_action: Literal["ACCELERATE", "MAINTAIN", "SLOW_DOWN", "HOLD"]
    explanation: str
    production_allocation_pct: float | None = None
    warehouse_pressure_pct: float | None = None
    reason: str | None = None


class RiskAssessment(BaseModel):
    risk_score: float
    risk_level: Literal["GREEN", "YELLOW", "RED"]
    risk_reasons: list[str]
    virtual_inventory_warning: bool
    confidence_proxy: float


class PredictRequest(BaseModel):
    sequence: Any = Field(..., description="Integer list hoac chuoi comma-separated.")


class PredictResponse(BaseModel):
    input_summary: InputSummary
    predicted_outputs: PredictedOutputs
    scheduler_decision: SchedulerDecision
    risk_assessment: RiskAssessment
    source_mode: Literal["precomputed", "live"]
    model_artifacts: dict[str, Any] | None = None


class OrderSummary(BaseModel):
    order_id: str
    sequence_length: int
    sequence_preview: list[int]
    first_action: str | None = None
    last_action: str | None = None
    priority_level: str | None = None
    risk_level: str | None = None


class OrderDetail(BaseModel):
    order_id: str
    sequence_length: int
    sequence_preview: list[int]
    first_action: str | None = None
    last_action: str | None = None
    raw_sequence: list[int]
    input_summary: InputSummary
    source_mode: Literal["precomputed", "live"]


class DatasetOverview(BaseModel):
    total_orders: int
    sequence_column_count: int
    average_sequence_length: float
    median_sequence_length: float
    p95_sequence_length: int
    max_sequence_length: int
    source_mode: Literal["precomputed", "live"]
    generated_at: str | None = None
    model_artifacts: dict[str, Any] | None = None
    precomputed_path: str | None = None


class PlanningOrderItem(BaseModel):
    order_id: str
    priority_level: str
    recommended_action: str
    planning_day_bucket: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    today_production_pct: float
    warehouse_waiting_pressure_pct: float
    risk_score: float
    risk_level: str
    planning_rank_score: float
    behavior_diversity_score: float | None = None
    sequence_length: int | None = None
    unique_action_count: int | None = None
    plan_status: str | None = None
    anchor_action: str | None = None
    rollback_3_count: int | None = None
    rollback_4_count: int | None = None
    entropy: float | None = None
    rare_action_ratio: float | None = None
    capacity_band: str | None = None
    completion_urgency_band: str | None = None
    warehouse_stress_zone: str | None = None


class PlanningOverview(BaseModel):
    source_mode: Literal["precomputed"]
    total_orders: int
    daily_capacity_budget_pct: float
    daily_warehouse_budget_pct: float
    selected_orders_count: int
    deferred_orders_count: int
    cumulative_selected_production_pct: float
    cumulative_selected_warehouse_pressure_pct: float
    capacity_budget_utilization_pct: float
    warehouse_budget_utilization_pct: float
    cutoff_reason: str
    selected_orders_for_today: list[PlanningOrderItem]
    deferred_orders: list[PlanningOrderItem]
    top_priority_orders: list[PlanningOrderItem]
    top_accelerate_orders: list[PlanningOrderItem]
    top_hold_orders: list[PlanningOrderItem]
    top_warehouse_pressure_orders: list[PlanningOrderItem]
    top_risk_orders: list[PlanningOrderItem]
    planning_table_total_count: int
    planning_table_offset: int
    planning_table_limit: int
    planning_table: list[PlanningOrderItem]


class HealthResponse(BaseModel):
    status: str
    predictor_type: str
    model_count: int
    model_files: list[str]
    runtime_ready: bool
    required_packages: dict[str, bool]
    max_sequence_length: int
    stats_feature_count: int
    aggregation: str | None = None
    artifact_root: str | None = None
    default_source_mode: Literal["precomputed", "live"]
    precomputed_ready: bool
    live_ready: bool
    precomputed_path: str | None = None
