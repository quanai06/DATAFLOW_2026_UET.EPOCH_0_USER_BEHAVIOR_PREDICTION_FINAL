from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.app.backend.config import X_TEST_PATH


class OrderDataStore:
    def __init__(self, csv_path: Path = X_TEST_PATH) -> None:
        self.csv_path = Path(csv_path)
        self._frame: pd.DataFrame | None = None
        self._sequence_columns: list[str] | None = None

    def _load_frame(self) -> pd.DataFrame:
        if self._frame is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Khong tim thay file du lieu: {self.csv_path}")
            self._frame = pd.read_csv(self.csv_path)
        return self._frame

    def _get_sequence_columns(self) -> list[str]:
        if self._sequence_columns is None:
            frame = self._load_frame()
            sequence_columns = [column for column in frame.columns if str(column).startswith("feature_")]
            sequence_columns.sort(key=lambda value: int(str(value).split("_")[1]))
            self._sequence_columns = sequence_columns
        return self._sequence_columns

    def _row_to_sequence(self, row: pd.Series) -> list[int]:
        sequence: list[int] = []
        for column in self._get_sequence_columns():
            value = row[column]
            if pd.isna(value):
                continue
            integer_value = int(float(value))
            if integer_value > 0:
                sequence.append(integer_value)
        return sequence

    def list_orders(self) -> list[dict[str, Any]]:
        frame = self._load_frame()
        orders: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            sequence = self._row_to_sequence(row)
            orders.append(
                {
                    "order_id": str(row["id"]),
                    "sequence_length": len(sequence),
                    "sequence_preview": sequence[:10],
                    "first_action": str(sequence[0]) if sequence else None,
                    "last_action": str(sequence[-1]) if sequence else None,
                }
            )
        return orders

    def list_order_details(self) -> list[dict[str, Any]]:
        frame = self._load_frame()
        details: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            sequence = self._row_to_sequence(row)
            details.append(
                {
                    "order_id": str(row["id"]),
                    "sequence_length": len(sequence),
                    "sequence_preview": sequence[:10],
                    "first_action": str(sequence[0]) if sequence else None,
                    "last_action": str(sequence[-1]) if sequence else None,
                    "sequence": sequence,
                }
            )
        return details

    def iter_order_detail_chunks(self, chunk_size: int) -> tuple[int, Any]:
        frame = self._load_frame()
        total_orders = int(len(frame))

        def _generator():
            chunk: list[dict[str, Any]] = []
            for _, row in frame.iterrows():
                sequence = self._row_to_sequence(row)
                chunk.append(
                    {
                        "order_id": str(row["id"]),
                        "sequence_length": len(sequence),
                        "sequence_preview": sequence[:10],
                        "first_action": str(sequence[0]) if sequence else None,
                        "last_action": str(sequence[-1]) if sequence else None,
                        "sequence": sequence,
                    }
                )
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        return total_orders, _generator()

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        frame = self._load_frame()
        matched = frame.loc[frame["id"].astype(str) == str(order_id)]
        if matched.empty:
            return None
        row = matched.iloc[0]
        sequence = self._row_to_sequence(row)
        return {
            "order_id": str(row["id"]),
            "sequence_length": len(sequence),
            "sequence_preview": sequence[:10],
            "first_action": str(sequence[0]) if sequence else None,
            "last_action": str(sequence[-1]) if sequence else None,
            "sequence": sequence,
        }

    def get_dataset_overview(self) -> dict[str, Any]:
        frame = self._load_frame()
        lengths = [len(self._row_to_sequence(row)) for _, row in frame.iterrows()]
        if not lengths:
            return {
                "total_orders": 0,
                "sequence_column_count": len(self._get_sequence_columns()),
                "average_sequence_length": 0.0,
                "median_sequence_length": 0.0,
                "p95_sequence_length": 0,
                "max_sequence_length": 0,
            }

        length_series = pd.Series(lengths)
        return {
            "total_orders": int(len(lengths)),
            "sequence_column_count": len(self._get_sequence_columns()),
            "average_sequence_length": round(float(length_series.mean()), 2),
            "median_sequence_length": round(float(length_series.median()), 2),
            "p95_sequence_length": int(length_series.quantile(0.95)),
            "max_sequence_length": int(length_series.max()),
        }
