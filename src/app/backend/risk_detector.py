from __future__ import annotations

from typing import Any


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_risk_assessment(
    input_summary: dict[str, Any],
    predicted_outputs: dict[str, int],
    scheduler_decision: dict[str, Any],
) -> dict[str, Any]:
    length_score = _clip(float(input_summary["sequence_length"]) / 66.0, 0.0, 1.0)
    unique_score = _clip(float(input_summary["unique_action_count"]) / 20.0, 0.0, 1.0)
    rollback_3_score = _clip(float(input_summary["rollback_3_count"]) / 6.0, 0.0, 1.0)
    rollback_4_score = _clip(float(input_summary["rollback_4_count"]) / 4.0, 0.0, 1.0)
    repeat_score = _clip(float(input_summary.get("repeat_density", 0.0)), 0.0, 1.0)
    volatility_score = _clip(float(predicted_outputs["attr_6"]) / 99.0, 0.0, 1.0)

    risk_score = _clip(
        0.35 * volatility_score
        + 0.18 * rollback_4_score
        + 0.15 * rollback_3_score
        + 0.12 * unique_score
        + 0.10 * repeat_score
        + 0.10 * length_score,
        0.0,
        1.0,
    )

    if risk_score >= 0.7:
        risk_level = "RED"
    elif risk_score >= 0.45:
        risk_level = "YELLOW"
    else:
        risk_level = "GREEN"

    reasons: list[str] = []
    if predicted_outputs["attr_6"] >= 70:
        reasons.append("Attr_6 cao, don hang co dau hieu bien dong va bat dinh lon.")
    if input_summary["rollback_4_count"] >= 1:
        reasons.append("Xuat hien A-B-C-A, cho thay mau hanh vi dao chieu phuc tap.")
    if input_summary["rollback_3_count"] >= 2:
        reasons.append("Rollback 3 buoc lap lai nhieu lan, de gay rung lich san xuat.")
    if scheduler_decision["warehouse_pressure_pct"] >= 70:
        reasons.append("Ap luc kho cho xuat cao neu day san luong ngay hom nay.")
    if input_summary["unique_action_count"] >= 12:
        reasons.append("So action duy nhat cao, profile co do phan manh lon.")
    if not reasons:
        reasons.append("Mau hanh vi tuong doi on dinh, chua thay tin hieu rui ro noi bat.")

    virtual_inventory_warning = bool(
        scheduler_decision["warehouse_pressure_pct"] >= 75
        and scheduler_decision["production_allocation_pct"] >= 55
    )

    confidence_proxy = _clip(
        1.0 - (0.45 * volatility_score + 0.25 * rollback_4_score + 0.20 * rollback_3_score + 0.10 * repeat_score),
        0.05,
        0.95,
    )

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "risk_reasons": reasons[:4],
        "virtual_inventory_warning": virtual_inventory_warning,
        "confidence_proxy": round(confidence_proxy, 4),
    }

