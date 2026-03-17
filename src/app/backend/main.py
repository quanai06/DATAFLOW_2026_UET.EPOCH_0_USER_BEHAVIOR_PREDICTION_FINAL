from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.app.backend.config import DEFAULT_SOURCE_MODE, PRECOMPUTED_CSV_PATH, PRECOMPUTED_PARQUET_PATH
from src.app.backend.data_store import OrderDataStore
from src.app.backend.predictor import PredictorUnavailableError, RealEnsemblePredictor
from src.app.backend.precomputed_store import PrecomputedStore
from src.app.backend.risk_detector import build_risk_assessment
from src.app.backend.scheduler import build_scheduler_decision
from src.app.backend.schemas import (
    DatasetOverview,
    HealthResponse,
    OrderDetail,
    OrderSummary,
    PlanningOverview,
    PredictRequest,
    PredictResponse,
)


app = FastAPI(
    title="Risk-Aware Dynamic Scheduler API",
    description="Precomputed-first demo API for X_test -> model outputs -> scheduler decision.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = RealEnsemblePredictor()
raw_order_store = OrderDataStore()
precomputed_store = PrecomputedStore()


def _live_prediction_payload(sequence: object) -> dict:
    prediction = predictor.predict(sequence)
    scheduler_decision = build_scheduler_decision(prediction["predicted_outputs"])
    risk_assessment = build_risk_assessment(
        prediction["input_summary"],
        prediction["predicted_outputs"],
        scheduler_decision,
    )
    return {
        "input_summary": prediction["input_summary"],
        "predicted_outputs": prediction["predicted_outputs"],
        "scheduler_decision": scheduler_decision,
        "risk_assessment": risk_assessment,
        "source_mode": "live",
        "model_artifacts": prediction["model_artifacts"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor_health = predictor.health()
    return HealthResponse(
        **predictor_health,
        default_source_mode=DEFAULT_SOURCE_MODE,
        precomputed_ready=precomputed_store.is_ready(),
        live_ready=bool(predictor_health["runtime_ready"]),
        precomputed_path=(
            str(PRECOMPUTED_PARQUET_PATH if PRECOMPUTED_PARQUET_PATH.exists() else PRECOMPUTED_CSV_PATH)
            if precomputed_store.is_ready()
            else None
        ),
    )


@app.get("/dataset/overview", response_model=DatasetOverview)
def dataset_overview() -> DatasetOverview:
    try:
        overview = precomputed_store.dataset_overview()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DatasetOverview(**overview)


@app.get("/orders", response_model=list[OrderSummary])
def orders(
    query: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[OrderSummary]:
    try:
        return [OrderSummary(**item) for item in precomputed_store.list_orders(limit=limit, query=query)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/orders/{order_id}", response_model=OrderDetail)
def order_detail(order_id: str) -> OrderDetail:
    try:
        order = precomputed_store.get_order(order_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if order is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay order_id={order_id}.")
    return OrderDetail(**order)


@app.get("/planning/overview", response_model=PlanningOverview)
def planning_overview(
    limit: int = Query(default=10, ge=3, le=30),
    capacity_budget_pct: float = Query(default=300.0, ge=20.0, le=1000.0),
    warehouse_budget_pct: float = Query(default=320.0, ge=20.0, le=1500.0),
    planning_table_offset: int = Query(default=0, ge=0, le=200000),
    planning_table_limit: int = Query(default=100, ge=1, le=100),
) -> PlanningOverview:
    try:
        payload = precomputed_store.planning_overview(
            limit=limit,
            capacity_budget_pct=capacity_budget_pct,
            warehouse_budget_pct=warehouse_budget_pct,
            planning_table_offset=planning_table_offset,
            planning_table_limit=planning_table_limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PlanningOverview(**payload)


@app.get("/predict/order/{order_id}", response_model=PredictResponse)
def predict_order(order_id: str) -> PredictResponse:
    try:
        payload = precomputed_store.get_prediction(order_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay order_id={order_id}.")
    return PredictResponse(**payload)


@app.post("/predict/live", response_model=PredictResponse)
def predict_live(request: PredictRequest) -> PredictResponse:
    try:
        payload = _live_prediction_payload(request.sequence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(**payload)


@app.get("/predict/order-live/{order_id}", response_model=PredictResponse)
def predict_order_live(order_id: str) -> PredictResponse:
    try:
        order = raw_order_store.get_order(order_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if order is None:
        raise HTTPException(status_code=404, detail=f"Khong tim thay order_id={order_id}.")

    try:
        payload = _live_prediction_payload(order["sequence"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(**payload)
