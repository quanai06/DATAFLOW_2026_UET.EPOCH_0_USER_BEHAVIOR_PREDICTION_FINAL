from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.backend.config import (
    PRECOMPUTED_CSV_PATH,
    PRECOMPUTED_PARQUET_PATH,
    PRECOMPUTED_SUMMARY_PATH,
)
from src.app.backend.planning_engine import build_daily_plan
from src.app.backend.risk_detector import build_risk_assessment
from src.app.backend.scheduler import build_scheduler_decision


class PrecomputedStore:
    def __init__(
        self,
        parquet_path: Path = PRECOMPUTED_PARQUET_PATH,
        csv_path: Path = PRECOMPUTED_CSV_PATH,
        summary_path: Path = PRECOMPUTED_SUMMARY_PATH,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.csv_path = Path(csv_path)
        self.summary_path = Path(summary_path)
        self._frame: pd.DataFrame | None = None
        self._enriched_frame: pd.DataFrame | None = None
        self._summary: dict[str, Any] | None = None

    def is_ready(self) -> bool:
        return self.parquet_path.exists() or self.csv_path.exists()

    def _load_frame(self) -> pd.DataFrame:
        if self._frame is None:
            if self.parquet_path.exists():
                self._frame = pd.read_parquet(self.parquet_path)
            elif self.csv_path.exists():
                self._frame = pd.read_csv(self.csv_path)
            else:
                raise FileNotFoundError(
                    "Chua co file precomputed_orders.parquet/csv. Hay chay precompute_x_test.py truoc."
                )
        return self._frame

    def _load_summary(self) -> dict[str, Any]:
        if self._summary is None:
            if self.summary_path.exists():
                self._summary = json.loads(self.summary_path.read_text())
            else:
                self._summary = {
                    "source_mode": "precomputed",
                    "generated_at": None,
                    "total_orders": 0,
                    "model_artifacts": {},
                }
        return self._summary

    def dataset_overview(self) -> dict[str, Any]:
        summary = self._load_summary()
        frame = self._load_frame()
        return {
            "total_orders": int(summary.get("total_orders", len(frame))),
            "sequence_column_count": int(summary.get("sequence_column_count", 66)),
            "average_sequence_length": float(
                summary.get("avg_length", frame["sequence_length"].mean())
            ),
            "median_sequence_length": float(
                summary.get("median_length", frame["sequence_length"].median())
            ),
            "p95_sequence_length": int(
                summary.get("p95_length", frame["sequence_length"].quantile(0.95))
            ),
            "max_sequence_length": int(
                summary.get("max_length", frame["sequence_length"].max())
            ),
            "source_mode": "precomputed",
            "generated_at": summary.get("generated_at"),
            "model_artifacts": summary.get("model_artifacts", {}),
            "precomputed_path": summary.get(
                "precomputed_path",
                str(self.parquet_path if self.parquet_path.exists() else self.csv_path),
            ),
        }

    def planning_overview(
        self,
        *,
        limit: int = 10,
        capacity_budget_pct: float = 100.0,
        warehouse_budget_pct: float = 120.0,
        planning_table_offset: int = 0,
        planning_table_limit: int = 100,
    ) -> dict[str, Any]:
        frame = self._load_enriched_frame().copy()
        if frame.empty:
            return {
                "source_mode": "precomputed",
                "total_orders": 0,
                "daily_capacity_budget_pct": round(capacity_budget_pct, 2),
                "daily_warehouse_budget_pct": round(warehouse_budget_pct, 2),
                "selected_orders_count": 0,
                "deferred_orders_count": 0,
                "cumulative_selected_production_load_pct": 0.0,
                "cumulative_selected_warehouse_stress_pct": 0.0,
                "capacity_budget_utilization_pct": 0.0,
                "warehouse_budget_utilization_pct": 0.0,
                "cutoff_reason": "No precomputed orders available.",
                "selected_orders_for_today": [],
                "deferred_orders": [],
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

        plan = build_daily_plan(
            frame,
            capacity_budget_pct=capacity_budget_pct,
            warehouse_budget_pct=warehouse_budget_pct,
            limit=limit,
            planning_table_offset=planning_table_offset,
            planning_table_limit=planning_table_limit,
        )
        return {
            "source_mode": "precomputed",
            "total_orders": int(len(frame)),
            **plan,
        }

    def list_orders(
        self,
        *,
        limit: int = 200,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        frame = self._load_frame()
        enriched_lookup = None

        if {"attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"}.issubset(frame.columns):
            enriched = self._load_enriched_frame().set_index("id")
            enriched_lookup = enriched[["priority_level", "risk_level", "recommended_action"]]

        working_frame = frame
        if query:
            normalized_query = str(query).strip().lower()
            working_frame = working_frame.loc[
                working_frame["id"].astype(str).str.lower().str.contains(
                    normalized_query, na=False
                )
            ]

        if limit > 0:
            working_frame = working_frame.head(limit)

        results: list[dict[str, Any]] = []
        for _, row in working_frame.iterrows():
            raw_sequence = self._parse_json_field(row["raw_sequence"])
            current_priority = str(row.get("priority_level", "")) or None
            current_risk = str(row.get("risk_level", "")) or None
            current_action = str(row.get("recommended_action", "")) or None

            if enriched_lookup is not None and str(row["id"]) in enriched_lookup.index:
                current_priority = str(enriched_lookup.at[str(row["id"]), "priority_level"])
                current_risk = str(enriched_lookup.at[str(row["id"]), "risk_level"])
                current_action = str(enriched_lookup.at[str(row["id"]), "recommended_action"])

            results.append(
                {
                    "order_id": str(row["id"]),
                    "sequence_length": int(row["sequence_length"]),
                    "sequence_preview": raw_sequence[:10],
                    "first_action": str(raw_sequence[0]) if raw_sequence else None,
                    "last_action": str(raw_sequence[-1]) if raw_sequence else None,
                    "priority_level": current_priority,
                    "risk_level": current_risk,
                    "recommended_action": current_action,
                }
            )
        return results

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        frame = self._load_frame()
        matched = frame.loc[frame["id"].astype(str) == str(order_id)]
        if matched.empty:
            return None

        row = matched.iloc[0]
        raw_sequence = self._parse_json_field(row["raw_sequence"])
        input_summary = self._parse_json_field(row["input_summary_json"])

        return {
            "order_id": str(row["id"]),
            "sequence_length": int(row["sequence_length"]),
            "sequence_preview": raw_sequence[:10],
            "first_action": str(raw_sequence[0]) if raw_sequence else None,
            "last_action": str(raw_sequence[-1]) if raw_sequence else None,
            "raw_sequence": raw_sequence,
            "input_summary": input_summary,
            "source_mode": "precomputed",
        }

    def get_prediction(self, order_id: str) -> dict[str, Any] | None:
        frame = self._load_frame()
        matched = frame.loc[frame["id"].astype(str) == str(order_id)]
        if matched.empty:
            return None

        row = matched.iloc[0]
        summary = self._load_summary()
        predicted_outputs = self._predicted_outputs_from_row(row)
        input_summary = self._parse_json_field(row["input_summary_json"])

        scheduler_decision = build_scheduler_decision(predicted_outputs)
        risk_assessment = build_risk_assessment(
            input_summary,
            predicted_outputs,
            scheduler_decision,
        )

        return {
            "input_summary": input_summary,
            "predicted_outputs": predicted_outputs,
            "scheduler_decision": scheduler_decision,
            "risk_assessment": risk_assessment,
            "source_mode": "precomputed",
            "model_artifacts": summary.get("model_artifacts", {}),
        }

    def _predicted_outputs_from_row(self, row: pd.Series) -> dict[str, int]:
        return {
            "attr_1": int(row["attr_1"]),
            "attr_2": int(row["attr_2"]),
            "attr_3": int(row["attr_3"]),
            "attr_4": int(row["attr_4"]),
            "attr_5": int(row["attr_5"]),
            "attr_6": int(row["attr_6"]),
        }

    def _enrich_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Một nguồn sự thật duy nhất:
        - luôn dùng build_scheduler_decision()
        - luôn dùng build_risk_assessment()

        Không tự vectorize lại scheduler logic ở đây nữa,
        để tránh lệch giữa planning-level và single-order view.
        """
        enriched_rows: list[dict[str, Any]] = []

        for _, row in frame.iterrows():
            predicted_outputs = self._predicted_outputs_from_row(row)
            input_summary = self._parse_json_field(row["input_summary_json"])

            scheduler_decision = build_scheduler_decision(predicted_outputs)
            risk_assessment = build_risk_assessment(
                input_summary,
                predicted_outputs,
                scheduler_decision,
            )

            enriched_rows.append(
                {
                    "id": str(row["id"]),
                    "start_month": int(predicted_outputs["attr_1"]),
                    "start_day": int(predicted_outputs["attr_2"]),
                    "end_month": int(predicted_outputs["attr_4"]),
                    "end_day": int(predicted_outputs["attr_5"]),
                    "start_date": f"{int(predicted_outputs['attr_1']):02d}/{int(predicted_outputs['attr_2']):02d}",
                    "end_date": f"{int(predicted_outputs['attr_4']):02d}/{int(predicted_outputs['attr_5']):02d}",
                    "sequence_length": int(row.get("sequence_length", input_summary.get("sequence_length", 0))),
                    "unique_action_count": int(
                        row.get("unique_action_count", input_summary.get("unique_action_count", 0))
                    ),
                    "anchor_action": str(
                        row.get("anchor_action", input_summary.get("anchor_action", "UNKNOWN"))
                    ),
                    "rollback_3_count": int(
                        row.get("rollback_3_count", input_summary.get("rollback_3_count", 0))
                    ),
                    "rollback_4_count": int(
                        row.get("rollback_4_count", input_summary.get("rollback_4_count", 0))
                    ),
                    "transition_ratio": float(
                        row.get("transition_ratio", input_summary.get("transition_ratio", 0.0) or 0.0)
                    ),
                    "entropy": float(row.get("entropy", input_summary.get("entropy", 0.0) or 0.0)),
                    "rare_action_ratio": float(
                        row.get("rare_action_ratio", input_summary.get("rare_action_ratio", 0.0) or 0.0)
                    ),
                    "today_production_pct": float(scheduler_decision["today_production_pct"]),
                    "warehouse_waiting_pressure_pct": float(
                        scheduler_decision["warehouse_waiting_pressure_pct"]
                    ),
                    "priority_level": str(scheduler_decision["priority_level"]),
                    "recommended_action": str(scheduler_decision["recommended_action"]),
                    "risk_score": float(risk_assessment["risk_score"]),
                    "risk_level": str(risk_assessment["risk_level"]),
                    "capacity_score": float(scheduler_decision["capacity_score"]),
                    "completion_urgency_score": float(
                        scheduler_decision["completion_urgency_score"]
                    ),
                    "storage_exposure_score": float(
                        scheduler_decision.get("storage_exposure_score", 0.0)
                    ),
                    "capacity_band": str(scheduler_decision.get("capacity_band", "LOW")),
                    "completion_urgency_band": str(
                        scheduler_decision.get("completion_urgency_band", "LONG")
                    ),
                    "completion_window_days": int(
                        scheduler_decision.get("completion_window_days", 1)
                    ),
                    "warehouse_stress_zone": str(
                        scheduler_decision.get("warehouse_stress_zone", "LOW")
                    ),
                }
            )

        return pd.DataFrame(enriched_rows)

    def _load_enriched_frame(self) -> pd.DataFrame:
        if self._enriched_frame is None:
            self._enriched_frame = self._enrich_frame(self._load_frame())
        return self._enriched_frame

    def _parse_json_field(self, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value
