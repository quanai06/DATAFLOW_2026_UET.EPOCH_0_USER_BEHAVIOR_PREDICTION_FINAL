from __future__ import annotations

from typing import Any


def init_session_state(st: Any) -> None:
    defaults = {
        "selected_order_id": None,
        "sequence_text": "",
        "single_prediction": None,
        "orders_cache": [],
        "dataset_overview": None,
        "planning_overview": None,
        "selected_order_detail": None,
        "order_search_query": "",
        "loaded_order_id_input": "",
        "capacity_budget_pct": 300.0,
        "warehouse_budget_pct": 320.0,
        "planning_focus_order_id": None,
        "planning_table_start": 1,
        "planning_table_end": 100,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sequence_to_text(sequence: list[int]) -> str:
    return ", ".join(str(value) for value in sequence)


def parse_sequence_text(sequence_text: str) -> list[int]:
    parts = [part.strip() for part in sequence_text.replace("\n", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("Hay nhap it nhat 1 action ID.")

    parsed: list[int] = []
    for part in parts:
        try:
            value = int(float(part))
        except ValueError as exc:
            raise ValueError(f"Gia tri khong hop le: {part}") from exc
        if value > 0:
            parsed.append(value)

    if not parsed:
        raise ValueError("Sequence phai co it nhat 1 action ID > 0.")
    return parsed


def order_label(order: dict[str, Any]) -> str:
    preview = ", ".join(str(value) for value in order.get("sequence_preview", [])[:4])
    if preview:
        preview = f" | {preview}"
    return f"{order['order_id']} | len={order['sequence_length']}{preview}"


def badge_tone(value: str) -> str:
    palette = {
        "HIGH": "#8f1d1d",
        "MEDIUM": "#8a5600",
        "LOW": "#1f5f4a",
        "ACCELERATE": "#8f1d1d",
        "MAINTAIN": "#315c8a",
        "SLOW_DOWN": "#8a5600",
        "HOLD": "#5a4a42",
        "RED": "#8f1d1d",
        "YELLOW": "#8a5600",
        "GREEN": "#1f5f4a",
    }
    return palette.get(value, "#3d4c5a")


def format_pct(value: float) -> str:
    return f"{value:.1f}%"
