from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from src.app.backend.config import (
    BUSINESS_DENOMINATORS,
    CHECK_MODEL_PATH,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_TARGETS,
    DEFAULT_TOP_HUBS,
    ENSEMBLE_GLOB,
    METADATA_PATH,
    MODEL_DIR,
    PREFERRED_SINGLE_MODEL,
    SAVED_ARTIFACTS_DIR,
    SCALER_PHASE1_PATH,
    SCALER_PHASE2_PATH,
)
from src.app.backend.feature_extractor import build_input_summary, create_features, process_sequences


class PredictorUnavailableError(RuntimeError):
    pass


class RealEnsemblePredictor:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._metadata = self._load_metadata()
        self._models: list[Any] | None = None
        self._phase1_scaler: Any | None = None
        self._phase2_scaler: Any | None = None
        self._numpy: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def top_hubs(self) -> tuple[int, ...]:
        hubs = self._metadata.get("TOP_10_HUBS") or list(DEFAULT_TOP_HUBS)
        return tuple(int(value) for value in hubs)

    @property
    def max_length(self) -> int:
        return int(self._metadata.get("FINAL_MAX_LEN", DEFAULT_MAX_SEQUENCE_LENGTH))

    @property
    def target_names(self) -> tuple[str, ...]:
        target_names = self._metadata.get("TARGET_COLS") or list(DEFAULT_TARGETS)
        return tuple(str(name) for name in target_names)

    @property
    def m_const_np(self) -> list[float]:
        return [float(value) for value in self._metadata.get("M_CONST_NP", [12.0, 31.0, 99.0, 12.0, 31.0, 99.0])]

    def health(self) -> dict[str, Any]:
        model_files = [path.name for path in self._discover_model_paths()]
        runtime_ready = all(
            importlib.util.find_spec(module_name) is not None
            for module_name in ("numpy", "joblib", "tensorflow")
        )
        return {
            "status": "ok" if model_files else "missing_models",
            "predictor_type": "real-ensemble",
            "model_count": len(model_files),
            "model_files": model_files,
            "runtime_ready": runtime_ready,
            "required_packages": {
                "numpy": importlib.util.find_spec("numpy") is not None,
                "joblib": importlib.util.find_spec("joblib") is not None,
                "tensorflow": importlib.util.find_spec("tensorflow") is not None,
            },
            "max_sequence_length": self.max_length,
            "stats_feature_count": int(self._metadata.get("NUM_WIDE_FEATURES", 15)),
            "aggregation": "median",
            "artifact_root": str(self._artifact_root()),
        }

    def predict(self, sequence_input: Any) -> dict[str, Any]:
        return self.predict_many([sequence_input], batch_size=1)[0]

    def predict_many(self, sequence_inputs: list[Any], *, batch_size: int = 512) -> list[dict[str, Any]]:
        processed_sequences = process_sequences(
            sequence_inputs,
            max_length=self.max_length,
            top_hubs=self.top_hubs,
        )
        self._ensure_runtime_loaded()
        assert self._numpy is not None

        input_ids = self._numpy.asarray(
            [item["padded_sequence"] for item in processed_sequences],
            dtype="int32",
        )
        raw_stats = self._numpy.asarray(create_features(processed_sequences), dtype="float32")
        input_stats = self._transform_stats(raw_stats)

        raw_predictions = []
        for model in self._models or []:
            prediction = model.predict(
                {"input_ids": input_ids, "input_stats": input_stats},
                verbose=0,
                batch_size=batch_size,
            )
            raw_predictions.append(self._numpy.asarray(prediction))

        if not raw_predictions:
            raise PredictorUnavailableError("Khong tim thay model ensemble de suy luan.")

        ensemble_raw = self._numpy.median(self._numpy.stack(raw_predictions, axis=0), axis=0)

        prediction_rows: list[dict[str, Any]] = []
        for feature_bundle, raw_row in zip(processed_sequences, ensemble_raw.tolist()):
            prediction_rows.append(
                {
                    "input_summary": build_input_summary(feature_bundle),
                    "predicted_outputs": self._decode_outputs(raw_row),
                    "raw_prediction_scores": {
                        name: round(float(value), 6)
                        for name, value in zip(self.target_names, raw_row)
                    },
                    "model_artifacts": {
                        "model_count": len(self._models or []),
                        "model_files": [path.name for path in self._discover_model_paths()],
                        "aggregation": "median",
                        "metadata_path": str(self._metadata_path()),
                    },
                    "feature_bundle": feature_bundle,
                }
            )
        return prediction_rows

    def _artifact_root(self) -> Path:
        return SAVED_ARTIFACTS_DIR if SAVED_ARTIFACTS_DIR.exists() else self.model_dir

    def _metadata_path(self) -> Path:
        candidate = self._artifact_root() / "metadata.json"
        return candidate if candidate.exists() else METADATA_PATH

    def _phase1_scaler_path(self) -> Path:
        candidate = self._artifact_root() / "scaler_phase1.joblib"
        return candidate if candidate.exists() else SCALER_PHASE1_PATH

    def _phase2_scaler_path(self) -> Path:
        candidate = self._artifact_root() / "scaler_phase2.joblib"
        return candidate if candidate.exists() else SCALER_PHASE2_PATH

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            return {
                "FINAL_MAX_LEN": DEFAULT_MAX_SEQUENCE_LENGTH,
                "TARGET_COLS": list(DEFAULT_TARGETS),
                "TOP_10_HUBS": list(DEFAULT_TOP_HUBS),
                "NUM_WIDE_FEATURES": 15,
                "M_CONST_NP": [12.0, 31.0, 99.0, 12.0, 31.0, 99.0],
                "BUSINESS_DENOMINATORS": BUSINESS_DENOMINATORS,
            }

        metadata = json.loads(metadata_path.read_text())
        metadata.setdefault("FINAL_MAX_LEN", DEFAULT_MAX_SEQUENCE_LENGTH)
        metadata.setdefault("TARGET_COLS", list(DEFAULT_TARGETS))
        metadata.setdefault("TOP_10_HUBS", list(DEFAULT_TOP_HUBS))
        metadata.setdefault("NUM_WIDE_FEATURES", 15)
        metadata.setdefault("M_CONST_NP", [12.0, 31.0, 99.0, 12.0, 31.0, 99.0])
        metadata.setdefault("BUSINESS_DENOMINATORS", BUSINESS_DENOMINATORS)
        return metadata

    def _discover_model_paths(self) -> list[Path]:
        artifact_root = self._artifact_root()
        preferred_artifact_model = artifact_root / PREFERRED_SINGLE_MODEL
        if preferred_artifact_model.exists():
            return [preferred_artifact_model]

        preferred_model = self.model_dir / PREFERRED_SINGLE_MODEL
        if preferred_model.exists():
            return [preferred_model]

        model_paths = sorted(artifact_root.glob(ENSEMBLE_GLOB))
        if model_paths:
            return model_paths
        fallback_paths = sorted(self.model_dir.glob(ENSEMBLE_GLOB))
        if fallback_paths:
            return fallback_paths
        if CHECK_MODEL_PATH.exists():
            return [CHECK_MODEL_PATH]
        return []

    def _ensure_runtime_loaded(self) -> None:
        if self._models is not None:
            return

        try:
            import joblib
            import numpy as np
            import tensorflow as tf
        except ImportError as exc:
            raise PredictorUnavailableError(
                "Thieu dependency de chay predictor that. Can cai joblib, numpy, tensorflow."
            ) from exc

        model_paths = self._discover_model_paths()
        if not model_paths:
            raise PredictorUnavailableError("Khong tim thay file model trong thu muc artifact.")

        self._numpy = np
        phase1_path = self._phase1_scaler_path()
        phase2_path = self._phase2_scaler_path()
        self._phase1_scaler = joblib.load(phase1_path) if phase1_path.exists() else None
        self._phase2_scaler = joblib.load(phase2_path) if phase2_path.exists() else None
        self._models = [tf.keras.models.load_model(path, compile=False) for path in model_paths]

    def _transform_stats(self, raw_stats: Any) -> Any:
        active_scaler = self._phase2_scaler or self._phase1_scaler
        fallback_scaler = self._phase1_scaler if active_scaler is self._phase2_scaler else self._phase2_scaler
        if active_scaler is not None:
            try:
                return active_scaler.transform(raw_stats)
            except Exception:
                if fallback_scaler is not None:
                    return fallback_scaler.transform(raw_stats)
                raise
        return raw_stats

    def _days_in_month(self, month_value: int) -> int:
        month_value = max(1, min(12, month_value))
        if month_value == 2:
            return 29
        if month_value in {4, 6, 9, 11}:
            return 30
        return 31

    def _decode_outputs(self, raw_output: list[float]) -> dict[str, int]:
        scaled_values = [
            int(round(float(raw_value) * float(scale)))
            for raw_value, scale in zip(raw_output, self.m_const_np)
        ]

        start_month = max(1, min(12, scaled_values[0]))
        start_day = scaled_values[1]
        workload = max(0, min(99, scaled_values[2]))
        end_month = max(1, min(12, scaled_values[3]))
        end_day = scaled_values[4]
        volatility = max(0, min(99, scaled_values[5]))

        if (start_month, start_day) > (end_month, end_day):
            start_month, end_month = end_month, start_month
            start_day, end_day = end_day, start_day

        start_day = max(1, min(self._days_in_month(start_month), start_day))
        end_day = max(1, min(self._days_in_month(end_month), end_day))

        return {
            "attr_1": start_month,
            "attr_2": start_day,
            "attr_3": workload,
            "attr_4": end_month,
            "attr_5": end_day,
            "attr_6": volatility,
        }
