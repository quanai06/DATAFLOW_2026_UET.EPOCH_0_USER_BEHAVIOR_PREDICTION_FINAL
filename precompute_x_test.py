from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.app.backend.config import PRECOMPUTED_CSV_PATH, PRECOMPUTED_PARQUET_PATH, PRECOMPUTED_SUMMARY_PATH, X_TEST_PATH
from src.app.backend.data_store import OrderDataStore
from src.app.backend.predictor import RealEnsemblePredictor
from src.app.backend.risk_detector import build_risk_assessment
from src.app.backend.scheduler import build_scheduler_decision


CHUNK_SIZE = 512


def main(limit: int | None = None) -> None:
    order_store = OrderDataStore(X_TEST_PATH)
    predictor = RealEnsemblePredictor()

    records: list[dict] = []
    available_orders, chunk_iterator = order_store.iter_order_detail_chunks(CHUNK_SIZE)
    total_orders = min(available_orders, limit) if limit is not None else available_orders
    processed_count = 0

    for chunk in chunk_iterator:
        if limit is not None and processed_count >= limit:
            break
        if limit is not None:
            remaining = limit - processed_count
            chunk = chunk[:remaining]
            if not chunk:
                break
        end = processed_count + len(chunk)
        sequences = [item["sequence"] for item in chunk]
        prediction_rows = predictor.predict_many(sequences, batch_size=CHUNK_SIZE)

        for order_detail, prediction_row in zip(chunk, prediction_rows):
            scheduler_decision = build_scheduler_decision(prediction_row["predicted_outputs"])
            risk_assessment = build_risk_assessment(
                prediction_row["input_summary"],
                prediction_row["predicted_outputs"],
                scheduler_decision,
            )

            top_actions_json = json.dumps(prediction_row["input_summary"]["action_frequency_top"], ensure_ascii=True)
            raw_sequence_json = json.dumps(order_detail["sequence"], ensure_ascii=True)

            records.append(
                {
                    "id": order_detail["order_id"],
                    "raw_sequence": raw_sequence_json,
                    "sequence_length": prediction_row["input_summary"]["sequence_length"],
                    "unique_action_count": prediction_row["input_summary"]["unique_action_count"],
                    "anchor_action": prediction_row["input_summary"]["anchor_action"],
                    "rollback_3_count": prediction_row["input_summary"]["rollback_3_count"],
                    "rollback_4_count": prediction_row["input_summary"]["rollback_4_count"],
                    "transition_ratio": prediction_row["input_summary"]["transition_ratio"],
                    "entropy": prediction_row["input_summary"]["entropy"],
                    "rare_action_ratio": prediction_row["input_summary"]["rare_action_ratio"],
                    "top_actions": top_actions_json,
                    "attr_1": prediction_row["predicted_outputs"]["attr_1"],
                    "attr_2": prediction_row["predicted_outputs"]["attr_2"],
                    "attr_3": prediction_row["predicted_outputs"]["attr_3"],
                    "attr_4": prediction_row["predicted_outputs"]["attr_4"],
                    "attr_5": prediction_row["predicted_outputs"]["attr_5"],
                    "attr_6": prediction_row["predicted_outputs"]["attr_6"],
                    "capacity_score": scheduler_decision["capacity_score"],
                    "completion_urgency_score": scheduler_decision["completion_urgency_score"],
                    "capacity_band": scheduler_decision["capacity_band"],
                    "completion_urgency_band": scheduler_decision["completion_urgency_band"],
                    "completion_window_days": scheduler_decision["completion_window_days"],
                    "warehouse_stress_zone": scheduler_decision["warehouse_stress_zone"],
                    "today_production_pct": scheduler_decision["today_production_pct"],
                    "warehouse_waiting_pressure_pct": scheduler_decision["warehouse_waiting_pressure_pct"],
                    "priority_level": scheduler_decision["priority_level"],
                    "recommended_action": scheduler_decision["recommended_action"],
                    "explanation": scheduler_decision["explanation"],
                    "risk_score": risk_assessment["risk_score"],
                    "risk_level": risk_assessment["risk_level"],
                    "risk_reasons": json.dumps(risk_assessment["risk_reasons"], ensure_ascii=True),
                    "virtual_inventory_warning": risk_assessment["virtual_inventory_warning"],
                    "confidence_proxy": risk_assessment["confidence_proxy"],
                    "input_summary_json": json.dumps(prediction_row["input_summary"], ensure_ascii=True),
                    "predicted_outputs_json": json.dumps(prediction_row["predicted_outputs"], ensure_ascii=True),
                    "scheduler_decision_json": json.dumps(scheduler_decision, ensure_ascii=True),
                    "risk_assessment_json": json.dumps(risk_assessment, ensure_ascii=True),
                }
            )

        processed_count = end
        print(f"Processed {processed_count}/{total_orders} orders")

    frame = pd.DataFrame(records)
    generated_at = datetime.now(timezone.utc).isoformat()

    written_path = None
    try:
        frame.to_parquet(PRECOMPUTED_PARQUET_PATH, index=False)
        written_path = PRECOMPUTED_PARQUET_PATH
    except Exception:
        frame.to_csv(PRECOMPUTED_CSV_PATH, index=False)
        written_path = PRECOMPUTED_CSV_PATH

    summary = {
        "source_mode": "precomputed",
        "generated_at": generated_at,
        "total_orders": int(len(frame)),
        "processed_orders": int(len(frame)),
        "source_x_test_rows": int(available_orders),
        "avg_length": round(float(frame["sequence_length"].mean()), 2),
        "median_length": round(float(frame["sequence_length"].median()), 2),
        "p95_length": int(frame["sequence_length"].quantile(0.95)),
        "max_length": int(frame["sequence_length"].max()),
        "sequence_column_count": 66,
        "precomputed_path": str(written_path),
        "model_artifacts": predictor.health(),
    }
    PRECOMPUTED_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Precomputed file saved to: {written_path}")
    print(f"Summary file saved to: {PRECOMPUTED_SUMMARY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Chi dung de debug nhanh tren mot phan X_test.")
    args = parser.parse_args()
    main(limit=args.limit)
