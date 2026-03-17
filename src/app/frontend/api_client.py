from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_BACKEND_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


class ApiClientError(RuntimeError):
    pass


def _request(
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    url = f"{DEFAULT_BACKEND_URL.rstrip('/')}{path}"
    try:
        response = requests.request(method, url, json=json_payload, params=params, timeout=60)
    except requests.RequestException as exc:
        raise ApiClientError(f"Khong ket noi duoc backend tai {url}.") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise ApiClientError(f"Backend tra ve loi {response.status_code}: {detail}")

    return response.json()


def get_health() -> dict[str, Any]:
    return _request("GET", "/health")


def get_dataset_overview() -> dict[str, Any]:
    return _request("GET", "/dataset/overview")


def get_planning_overview(
    *,
    limit: int = 10,
    capacity_budget_pct: float = 100.0,
    warehouse_budget_pct: float = 120.0,
    planning_table_offset: int = 0,
    planning_table_limit: int = 100,
) -> dict[str, Any]:
    return _request(
        "GET",
        "/planning/overview",
        params={
            "limit": limit,
            "capacity_budget_pct": capacity_budget_pct,
            "warehouse_budget_pct": warehouse_budget_pct,
            "planning_table_offset": planning_table_offset,
            "planning_table_limit": planning_table_limit,
        },
    )


def get_orders(*, query: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    params = {"limit": limit}
    if query:
        params["query"] = query
    return _request("GET", "/orders", params=params)


def get_order(order_id: str) -> dict[str, Any]:
    return _request("GET", f"/orders/{order_id}")


def predict_order(order_id: str) -> dict[str, Any]:
    return _request("GET", f"/predict/order/{order_id}")


def predict_live(sequence: list[int] | str) -> dict[str, Any]:
    return _request("POST", "/predict/live", json_payload={"sequence": sequence})


def predict_order_live(order_id: str) -> dict[str, Any]:
    return _request("GET", f"/predict/order-live/{order_id}")
