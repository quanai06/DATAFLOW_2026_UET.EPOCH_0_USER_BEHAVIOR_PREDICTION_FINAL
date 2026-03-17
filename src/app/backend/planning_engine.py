from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any

import pandas as pd

from src.app.backend.config import PLANNING_TODAY_DAY, PLANNING_TODAY_MONTH


ACTION_WEIGHT_MAP = {
    "ACCELERATE": 1.00,
    "MAINTAIN": 0.68,
    "SLOW_DOWN": 0.22,
    "HOLD": 0.00,
}

ACTION_PRIORITY_MAP = {
    "ACCELERATE": 4,
    "MAINTAIN": 3,
    "SLOW_DOWN": 2,
    "HOLD": 1,
}

PRIORITY_WEIGHT_MAP = {
    "HIGH": 1.00,
    "MEDIUM": 0.65,
    "LOW": 0.25,
}


def _bucketize(value: float, step: int = 10) -> int:
    return int(value // step) * step


def _day_of_year(month_value: int, day_value: int) -> int:
    month_value = max(1, min(12, int(month_value)))
    day_value = max(1, min(monthrange(2025, month_value)[1], int(day_value)))
    return date(2025, month_value, day_value).timetuple().tm_yday


def _planning_day_bucket(row: dict[str, Any]) -> tuple[str, int, int]:
    today_key = _day_of_year(PLANNING_TODAY_MONTH, PLANNING_TODAY_DAY)
    start_key = _day_of_year(row.get("start_month", PLANNING_TODAY_MONTH), row.get("start_day", PLANNING_TODAY_DAY))
    end_key = _day_of_year(row.get("end_month", PLANNING_TODAY_MONTH), row.get("end_day", PLANNING_TODAY_DAY))

    if start_key == today_key and end_key == today_key:
        return "TODAY_EXACT", 0, 0
    if start_key == today_key:
        return "START_TODAY", 1, max(0, end_key - today_key)
    if start_key > today_key:
        return "FUTURE_WINDOW", 2, start_key - today_key
    if start_key <= today_key <= end_key:
        return "ACTIVE_WINDOW", 3, 0
    return "PAST_DUE_OR_INVALID", 4, abs(start_key - today_key)


def _build_diversity_signature(row: dict[str, Any]) -> tuple:
    production_bucket = _bucketize(float(row.get("today_production_pct", 0.0)), step=10)
    warehouse_bucket = _bucketize(float(row.get("warehouse_waiting_pressure_pct", 0.0)), step=10)
    risk_bucket = _bucketize(float(row.get("risk_score", 0.0)) * 100.0, step=10)
    length_bucket = _bucketize(float(row.get("sequence_length", 0.0)), step=8)
    variety_bucket = _bucketize(float(row.get("unique_action_count", 0.0)), step=3)
    entropy_bucket = _bucketize(float(row.get("entropy", 0.0)) * 100.0, step=10)
    rare_bucket = _bucketize(float(row.get("rare_action_ratio", 0.0)) * 100.0, step=10)
    rollback_3_bucket = min(int(row.get("rollback_3_count", 0)), 4)
    rollback_4_bucket = min(int(row.get("rollback_4_count", 0)), 3)

    return (
        str(row.get("recommended_action", "UNKNOWN")),
        str(row.get("priority_level", "UNKNOWN")),
        str(row.get("capacity_band", "UNKNOWN")),
        str(row.get("completion_urgency_band", "UNKNOWN")),
        str(row.get("warehouse_stress_zone", "UNKNOWN")),
        str(row.get("anchor_action", "UNKNOWN")),
        length_bucket,
        variety_bucket,
        rollback_3_bucket,
        rollback_4_bucket,
        entropy_bucket,
        rare_bucket,
        production_bucket,
        warehouse_bucket,
        risk_bucket,
    )


def _select_diverse_rows(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    max_per_signature: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    signature_counts: dict[tuple, int] = {}

    for row in rows:
        signature = _build_diversity_signature(row)
        current_count = signature_counts.get(signature, 0)
        if current_count >= max_per_signature:
            continue
        selected.append(row)
        signature_counts[signature] = current_count + 1
        if len(selected) >= top_n:
            break

    return selected


def build_daily_plan(
    frame: pd.DataFrame,
    *,
    capacity_budget_pct: float,
    warehouse_budget_pct: float,
    limit: int = 10,
    max_per_signature: int = 1,
    planning_table_offset: int = 0,
    planning_table_limit: int = 100,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "daily_capacity_budget_pct": round(capacity_budget_pct, 2),
            "daily_warehouse_budget_pct": round(warehouse_budget_pct, 2),
            "selected_orders_count": 0,
            "deferred_orders_count": 0,
            "cumulative_selected_production_load_pct": 0.0,
            "cumulative_selected_warehouse_stress_pct": 0.0,
            "capacity_budget_utilization_pct": 0.0,
            "warehouse_budget_utilization_pct": 0.0,
            "selected_orders_for_today": [],
            "deferred_orders": [],
            "cutoff_reason": "No orders available.",
            "top_priority_orders": [],
            "top_accelerate_orders": [],
            "top_hold_orders": [],
            "top_warehouse_pressure_orders": [],
            "top_risk_orders": [],
            "planning_table_total_count": 0,
            "planning_table_offset": 0,
            "planning_table_limit": int(planning_table_limit),
            "planning_table": [],
        }

    working = frame.copy()

    # ======================================================
    # Normalize planning signals
    # ======================================================
    urgency_weight = pd.to_numeric(
        working["completion_urgency_score"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    capacity_weight = pd.to_numeric(
        working["capacity_score"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    risk_weight = pd.to_numeric(
        working["risk_score"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    warehouse_weight = (
        pd.to_numeric(working["warehouse_waiting_pressure_pct"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
        / 100.0
    )

    risk_inverse = 1.0 - risk_weight
    warehouse_inverse = 1.0 - warehouse_weight

    priority_weight = (
        working["priority_level"]
        .map(PRIORITY_WEIGHT_MAP)
        .fillna(0.0)
        .astype(float)
    )

    action_weight = (
        working["recommended_action"]
        .map(ACTION_WEIGHT_MAP)
        .fillna(0.0)
        .astype(float)
    )

    action_priority = (
        working["recommended_action"]
        .map(ACTION_PRIORITY_MAP)
        .fillna(0)
        .astype(int)
    )

    sequence_length_weight = (
        pd.to_numeric(working.get("sequence_length"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=66.0)
        / 66.0
    )

    variety_weight = (
        pd.to_numeric(working.get("unique_action_count"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=20.0)
        / 20.0
    )

    rollback_weight = (
        (
            pd.to_numeric(working.get("rollback_3_count"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=6.0) / 6.0
        )
        * 0.55
        + (
            pd.to_numeric(working.get("rollback_4_count"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=4.0) / 4.0
        )
        * 0.45
    ).clip(lower=0.0, upper=1.0)

    entropy_weight = (
        pd.to_numeric(working.get("entropy"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=3.5)
        / 3.5
    )

    rare_action_weight = (
        pd.to_numeric(working.get("rare_action_ratio"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )

    working["behavior_diversity_score"] = (
        100.0
        * (
            0.28 * sequence_length_weight
            + 0.26 * variety_weight
            + 0.20 * rollback_weight
            + 0.16 * entropy_weight
            + 0.10 * rare_action_weight
        )
    ).round(2)

    # ======================================================
    # Planning rank score
    # ======================================================
    working["planning_rank_score"] = (
        100.0
        * (
            0.24 * action_weight
            + 0.18 * priority_weight
            + 0.16 * urgency_weight
            + 0.12 * capacity_weight
            + 0.08 * risk_inverse
            + 0.08 * warehouse_inverse
            + 0.14 * (working["behavior_diversity_score"] / 100.0)
        )
    ).round(2)

    working.loc[working["recommended_action"] == "SLOW_DOWN", "planning_rank_score"] -= 12.0
    working.loc[working["recommended_action"] == "HOLD", "planning_rank_score"] -= 20.0
    working["planning_rank_score"] = working["planning_rank_score"].clip(lower=0.0)

    working["action_priority"] = action_priority
    working["priority_weight"] = priority_weight
    day_bucket_info = working.apply(lambda row: _planning_day_bucket(row.to_dict()), axis=1)
    working["planning_day_bucket"] = day_bucket_info.map(lambda item: item[0])
    working["planning_day_priority"] = day_bucket_info.map(lambda item: item[1])
    working["planning_day_distance"] = day_bucket_info.map(lambda item: item[2])
    working["priority_rank"] = (
        working["priority_level"]
        .map({"HIGH": 0, "MEDIUM": 1, "LOW": 2})
        .fillna(3)
        .astype(int)
    )

    working = working.sort_values(
        by=[
            "planning_day_priority",
            "planning_day_distance",
            "priority_rank",
            "today_production_pct",
            "warehouse_waiting_pressure_pct",
            "risk_score",
            "planning_rank_score",
            "behavior_diversity_score",
            "action_priority",
            "priority_weight",
        ],
        ascending=[True, True, True, True, True, True, False, False, False, False],
    ).reset_index(drop=True)

    cumulative_capacity = 0.0
    cumulative_warehouse = 0.0
    selected_rows: list[dict[str, Any]] = []
    deferred_rows: list[dict[str, Any]] = []
    cutoff_reason = "All ranked orders fit within today's budget."

    # signature cap để tránh shortlist toàn giống nhau
    signature_counts: dict[tuple, int] = {}

    for _, row in working.iterrows():
        row_dict = row.to_dict()
        current_action = str(row_dict["recommended_action"])

        if current_action == "HOLD":
            row_dict["plan_status"] = "DEFERRED"
            deferred_rows.append(row_dict)
            continue

        signature = _build_diversity_signature(row_dict)
        current_count = signature_counts.get(signature, 0)

        # nếu đã đủ quota cho signature này thì deferred luôn
        if current_count >= max_per_signature:
            row_dict["plan_status"] = "DEFERRED"
            deferred_rows.append(row_dict)
            continue

        order_capacity = float(row_dict["today_production_pct"])
        order_warehouse = float(row_dict["warehouse_waiting_pressure_pct"])

        next_capacity = cumulative_capacity + order_capacity
        next_warehouse = cumulative_warehouse + order_warehouse

        allowed = (
            next_capacity <= capacity_budget_pct
            and next_warehouse <= warehouse_budget_pct
        )

        if allowed:
            cumulative_capacity = next_capacity
            cumulative_warehouse = next_warehouse
            row_dict["plan_status"] = "SELECTED"
            selected_rows.append(row_dict)
            signature_counts[signature] = current_count + 1
        else:
            row_dict["plan_status"] = "DEFERRED"
            deferred_rows.append(row_dict)
            if cutoff_reason == "All ranked orders fit within today's budget.":
                if next_capacity > capacity_budget_pct and next_warehouse > warehouse_budget_pct:
                    cutoff_reason = (
                        "Stopped when both factory capacity budget and warehouse waiting budget "
                        "would be exceeded."
                    )
                elif next_capacity > capacity_budget_pct:
                    cutoff_reason = "Stopped when factory capacity budget would be exceeded."
                else:
                    cutoff_reason = "Stopped when warehouse waiting budget would be exceeded."

    if cutoff_reason == "All ranked orders fit within today's budget." and selected_rows:
        cutoff_reason = (
            f"Planner admitted diverse order patterns first and capped each decision signature at {max_per_signature} order(s)."
        )

    export_columns = [
        "id",
        "priority_level",
        "recommended_action",
        "planning_day_bucket",
        "start_date",
        "end_date",
        "today_production_pct",
        "warehouse_waiting_pressure_pct",
        "risk_score",
        "risk_level",
        "planning_rank_score",
        "behavior_diversity_score",
        "capacity_band",
        "completion_urgency_band",
        "warehouse_stress_zone",
        "sequence_length",
        "unique_action_count",
        "plan_status",
        "anchor_action",
        "rollback_3_count",
        "rollback_4_count",
        "entropy",
        "rare_action_ratio",
    ]

    def _export(rows: list[dict[str, Any]], top_n: int | None = None) -> list[dict[str, Any]]:
        if not rows:
            return []
        export_frame = pd.DataFrame(rows)
        available_columns = [c for c in export_columns if c in export_frame.columns]
        export_frame = export_frame.loc[:, available_columns].rename(columns={"id": "order_id"})
        if top_n is not None:
            export_frame = export_frame.head(top_n)
        return export_frame.to_dict(orient="records")

    top_priority_rows = _export(
        selected_rows if selected_rows else working.to_dict(orient="records"),
        top_n=limit,
    )

    top_accelerate_rows = _export(
        [r for r in selected_rows if r.get("recommended_action") == "ACCELERATE"],
        top_n=limit,
    )

    top_hold_rows = _export(
        sorted(
            [r for r in deferred_rows if r.get("recommended_action") in ["HOLD", "SLOW_DOWN"]],
            key=lambda x: (x.get("warehouse_waiting_pressure_pct", 0), x.get("risk_score", 0)),
            reverse=True,
        ),
        top_n=limit,
    )

    top_warehouse_rows = _export(
        sorted(
            working.to_dict(orient="records"),
            key=lambda x: (x.get("warehouse_waiting_pressure_pct", 0), x.get("risk_score", 0)),
            reverse=True,
        ),
        top_n=limit,
    )

    top_risk_rows = _export(
        sorted(
            working.to_dict(orient="records"),
            key=lambda x: (x.get("risk_score", 0), x.get("warehouse_waiting_pressure_pct", 0)),
            reverse=True,
        ),
        top_n=limit,
    )

    # QUAN TRỌNG:
    # planning_table giờ không lấy từ working nữa,
    # mà ưu tiên selected_rows trước rồi mới tới deferred_rows
    planning_table_rows = selected_rows + deferred_rows
    if planning_table_rows:
        planning_table_frame = pd.DataFrame(planning_table_rows).sort_values(
            by=[
                "planning_day_priority",
                "planning_day_distance",
                "priority_rank",
                "today_production_pct",
                "warehouse_waiting_pressure_pct",
                "risk_score",
                "planning_rank_score",
                "behavior_diversity_score",
                "action_priority",
                "priority_weight",
            ],
            ascending=[True, True, True, True, True, True, False, False, False, False],
        )
        planning_table_rows = planning_table_frame.to_dict(orient="records")
    planning_table_total_count = len(planning_table_rows)
    safe_offset = max(0, int(planning_table_offset))
    safe_limit = max(1, int(planning_table_limit))
    paged_planning_rows = planning_table_rows[safe_offset : safe_offset + safe_limit]

    return {
        "daily_capacity_budget_pct": round(capacity_budget_pct, 2),
        "daily_warehouse_budget_pct": round(warehouse_budget_pct, 2),
        "selected_orders_count": len(selected_rows),
        "deferred_orders_count": len(deferred_rows),
        "cumulative_selected_production_load_pct": round(cumulative_capacity, 2),
        "cumulative_selected_warehouse_stress_pct": round(cumulative_warehouse, 2),
        "capacity_budget_utilization_pct": round(
            (cumulative_capacity / max(capacity_budget_pct, 1.0)) * 100.0, 2
        ),
        "warehouse_budget_utilization_pct": round(
            (cumulative_warehouse / max(warehouse_budget_pct, 1.0)) * 100.0, 2
        ),
        # compatibility aliases
        "cumulative_selected_production_pct": round(cumulative_capacity, 2),
        "cumulative_selected_warehouse_pressure_pct": round(cumulative_warehouse, 2),
        "selected_orders_for_today": _export(selected_rows, top_n=max(limit, 12)),
        "deferred_orders": _export(deferred_rows, top_n=max(limit, 12)),
        "cutoff_reason": cutoff_reason,
        "top_priority_orders": top_priority_rows,
        "top_accelerate_orders": top_accelerate_rows,
        "top_hold_orders": top_hold_rows,
        "top_warehouse_pressure_orders": top_warehouse_rows,
        "top_risk_orders": top_risk_rows,
        "planning_table_total_count": planning_table_total_count,
        "planning_table_offset": safe_offset,
        "planning_table_limit": safe_limit,
        "planning_table": _export(paged_planning_rows),
    }
