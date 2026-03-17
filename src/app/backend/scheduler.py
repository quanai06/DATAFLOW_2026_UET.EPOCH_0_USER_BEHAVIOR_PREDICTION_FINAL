from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any


MIN_PROCESSING_MINUTES = 0.5
MAX_PROCESSING_MINUTES = 5.0
PROCESSING_TIME_CURVE_EXPONENT = 1.75


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _day_of_year(month_value: int, day_value: int) -> int:
    month_value = max(1, min(12, month_value))
    day_value = max(1, min(monthrange(2025, month_value)[1], day_value))
    return date(2025, month_value, day_value).timetuple().tm_yday


def _capacity_band(attr_3: float) -> str:
    if attr_3 >= 80:
        return "HIGH"
    if attr_3 >= 50:
        return "MEDIUM"
    return "LOW"


def _capacity_score(attr_3: float) -> float:
    # Keep the score continuous so nearby workloads no longer collapse into the same output.
    return _clip(attr_3 / 99.0, 0.0, 1.0)


def estimate_processing_minutes(attr_3: float, attr_6: float) -> float:
    workload = _clip(attr_3 / 99.0, 0.0, 1.0)
    volatility = _clip(attr_6 / 99.0, 0.0, 1.0)
    difficulty = 0.6 * workload + 0.4 * volatility
    scaled_minutes = MIN_PROCESSING_MINUTES + (
        (MAX_PROCESSING_MINUTES - MIN_PROCESSING_MINUTES)
        * (difficulty ** PROCESSING_TIME_CURVE_EXPONENT)
    )
    return round(_clip(scaled_minutes, MIN_PROCESSING_MINUTES, MAX_PROCESSING_MINUTES), 2)


def _urgency_band(window_days: int) -> str:
    if window_days <= 5:
        return "NEAR"
    if window_days <= 14:
        return "MID"
    if window_days <= 30:
        return "FAR"
    return "LONG"


def _urgency_score(window_days: int) -> float:
    # Smooth urgency with a decaying curve instead of only four discrete buckets.
    return _clip(1.0 / (1.0 + (max(window_days, 1) - 1) / 14.0), 0.05, 1.0)


def _warehouse_zone(warehouse_waiting_pressure_pct: float) -> str:
    if warehouse_waiting_pressure_pct >= 80:
        return "CRITICAL"
    if warehouse_waiting_pressure_pct >= 60:
        return "HIGH"
    if warehouse_waiting_pressure_pct >= 35:
        return "MEDIUM"
    return "LOW"


def _priority_level(
    capacity_band: str,
    urgency_band: str,
    warehouse_zone: str,
    recommended_action: str,
) -> str:
    """
    Priority không chỉ phụ thuộc urgency/capacity,
    mà còn phải phản ánh action cuối cùng.
    """
    if recommended_action == "ACCELERATE":
        return "HIGH"
    if recommended_action == "HOLD":
        return "LOW"
    if urgency_band == "NEAR" or capacity_band == "HIGH":
        return "MEDIUM"
    return "LOW"


def build_scheduler_decision(predicted_outputs: dict[str, int]) -> dict[str, Any]:
    start_month = int(predicted_outputs["attr_1"])
    start_day = int(predicted_outputs["attr_2"])
    attr_3 = float(predicted_outputs["attr_3"])
    end_month = int(predicted_outputs["attr_4"])
    end_day = int(predicted_outputs["attr_5"])
    attr_6 = float(predicted_outputs["attr_6"])

    start_ordinal = _day_of_year(start_month, start_day)
    end_ordinal = _day_of_year(end_month, end_day)

    # nếu dự đoán end < start thì coi như tối thiểu 1 ngày
    window_days = max(1, end_ordinal - start_ordinal + 1)

    capacity_band = _capacity_band(attr_3)
    capacity_score = _capacity_score(attr_3)
    urgency_band = _urgency_band(window_days)
    completion_urgency_score = _urgency_score(window_days)
    estimated_minutes = estimate_processing_minutes(attr_3, attr_6)

    # ======================================================
    # 1. Production allocation
    # - workload cao -> tăng
    # - urgency gần -> tăng
    # - nếu workload cao nhưng completion xa -> không nên chạy quá mạnh
    # ======================================================
    today_production_pct = 100.0 * (
        0.62 * capacity_score + 0.38 * completion_urgency_score
    )

    # override theo business
    if capacity_band == "HIGH" and urgency_band == "NEAR":
        today_production_pct += 8.0
    elif capacity_band == "HIGH" and urgency_band in {"FAR", "LONG"}:
        today_production_pct -= 12.0
    elif capacity_band == "LOW" and urgency_band == "LONG":
        today_production_pct -= 10.0
    elif urgency_band == "NEAR" and capacity_band == "LOW":
        today_production_pct += 5.0

    today_production_pct = _clip(today_production_pct, 5.0, 100.0)

    # ======================================================
    # 2. Warehouse waiting pressure
    # - workload cao -> tăng áp lực kho
    # - urgency thấp / completion xa -> tăng áp lực vì hàng phải chờ lâu hơn
    # - urgency gần -> giảm áp lực vì hàng có xu hướng ra nhanh hơn
    # ======================================================
    storage_exposure_score = (1.0 - completion_urgency_score)

    warehouse_waiting_pressure_pct = 100.0 * (
        0.58 * capacity_score + 0.42 * storage_exposure_score
    )

    if capacity_band == "HIGH" and urgency_band in {"FAR", "LONG"}:
        warehouse_waiting_pressure_pct += 12.0
    elif urgency_band == "NEAR":
        warehouse_waiting_pressure_pct -= 8.0
    elif capacity_band == "LOW" and urgency_band == "LONG":
        warehouse_waiting_pressure_pct -= 4.0

    warehouse_waiting_pressure_pct = _clip(warehouse_waiting_pressure_pct, 0.0, 100.0)
    warehouse_zone = _warehouse_zone(warehouse_waiting_pressure_pct)

    # ======================================================
    # 3. Recommended action
    # Rule khắt khe hơn để giảm tình trạng toàn MAINTAIN
    # ======================================================
    if warehouse_zone == "CRITICAL" and urgency_band != "NEAR":
        recommended_action = "HOLD"

    elif warehouse_zone == "HIGH" and urgency_band in {"FAR", "LONG"}:
        recommended_action = "SLOW_DOWN"

    elif urgency_band == "NEAR" and capacity_band == "HIGH" and warehouse_zone in {"LOW", "MEDIUM"}:
        recommended_action = "ACCELERATE"

    elif urgency_band == "NEAR" and capacity_band == "MEDIUM" and warehouse_zone != "CRITICAL":
        recommended_action = "ACCELERATE"

    elif capacity_band == "LOW" and urgency_band in {"FAR", "LONG"}:
        recommended_action = "HOLD"

    elif warehouse_zone in {"HIGH", "CRITICAL"}:
        recommended_action = "SLOW_DOWN"

    else:
        recommended_action = "MAINTAIN"

    priority_level = _priority_level(
        capacity_band=capacity_band,
        urgency_band=urgency_band,
        warehouse_zone=warehouse_zone,
        recommended_action=recommended_action,
    )

    # ======================================================
    # 4. Business explanation
    # ======================================================
    explanation = (
        f"Factory workload signal attr_3={int(attr_3)} -> {capacity_band.lower()} demand. "
        f"Completion target {end_month:02d}/{end_day:02d} with a {window_days}-day window -> {urgency_band.lower()} urgency. "
        f"Recommended today production is {today_production_pct:.1f}% and expected warehouse waiting pressure is {warehouse_waiting_pressure_pct:.1f}%. "
        f"Estimated processing time for this order is {estimated_minutes:.2f} minutes."
    )

    if recommended_action == "ACCELERATE":
        explanation += " Action: accelerate because completion is near and warehouse stress remains manageable."
    elif recommended_action == "MAINTAIN":
        explanation += " Action: maintain a steady pace because current workload and warehouse exposure are balanced."
    elif recommended_action == "SLOW_DOWN":
        explanation += " Action: slow down because warehouse pressure is rising faster than the completion benefit."
    else:
        explanation += " Action: hold because producing now would mostly build waiting inventory rather than improve near-term completion."

    return {
        "capacity_score": round(capacity_score, 4),
        "completion_urgency_score": round(completion_urgency_score, 4),
        "storage_exposure_score": round(storage_exposure_score, 4),
        "capacity_band": capacity_band,
        "completion_urgency_band": urgency_band,
        "completion_window_days": int(window_days),
        "warehouse_stress_zone": warehouse_zone,
        "today_production_pct": round(today_production_pct, 2),
        "warehouse_waiting_pressure_pct": round(warehouse_waiting_pressure_pct, 2),
        "estimated_processing_minutes": estimated_minutes,
        "priority_level": priority_level,
        "recommended_action": recommended_action,
        "explanation": explanation,

        # compatibility aliases
        "production_allocation_pct": round(today_production_pct, 2),
        "warehouse_pressure_pct": round(warehouse_waiting_pressure_pct, 2),
        "reason": explanation,
    }
