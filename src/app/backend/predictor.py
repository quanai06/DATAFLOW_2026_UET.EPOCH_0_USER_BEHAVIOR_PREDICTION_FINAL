from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from src.app.backend.config import (
    BUSINESS_DENOMINATORS,
    CHECK_MODEL_PATH,
    DEFAULT_COMBINE_TOP_ACTIONS,
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_TARGETS,
    DEFAULT_TOP_HUBS,
    ENSEMBLE_GLOB,
    FULL_SCALER_PATH,
    LEGACY_MODEL_DIR,
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
        self._stats_scaler: Any | None = None
        self._numpy: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def feature_profile(self) -> str:
        profile = str(self._metadata.get("FEATURE_PROFILE", DEFAULT_FEATURE_PROFILE))
        return "legacy_15" if profile == "legacy_15" else "combine_25"

    @property
    def top_hubs(self) -> tuple[int, ...]:
        if self.feature_profile == "combine_25":
            hubs = self._metadata.get("TOP_15_ACTIONS") or list(DEFAULT_COMBINE_TOP_ACTIONS)
        else:
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
    def stats_feature_count(self) -> int:
        default_count = 25 if self.feature_profile == "combine_25" else 15
        return int(self._metadata.get("NUM_WIDE_FEATURES", default_count))

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
            "stats_feature_count": self.stats_feature_count,
            "feature_profile": self.feature_profile,
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
            feature_profile=self.feature_profile,
        )
        self._ensure_runtime_loaded()
        assert self._numpy is not None

        input_ids = self._numpy.asarray(
            [item["padded_sequence"] for item in processed_sequences],
            dtype="int32",
        )
        raw_stats = self._numpy.asarray(create_features(processed_sequences), dtype="float32")
        if raw_stats.shape[1] != self.stats_feature_count:
            raise PredictorUnavailableError(
                f"Stats feature shape mismatch: tao duoc {raw_stats.shape[1]} cot, "
                f"nhung model/scaler dang can {self.stats_feature_count} cot."
            )
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
                        "scaler_path": str(self._scaler_path()),
                        "feature_profile": self.feature_profile,
                    },
                    "feature_bundle": feature_bundle,
                }
            )
        return prediction_rows

    def _candidate_roots(self) -> list[Path]:
        return [self.model_dir, SAVED_ARTIFACTS_DIR, LEGACY_MODEL_DIR]

    def _artifact_root(self) -> Path:
        for root in self._candidate_roots():
            if not root.exists():
                continue
            if (root / PREFERRED_SINGLE_MODEL).exists():
                return root
            if any(root.glob(ENSEMBLE_GLOB)):
                return root
            if (root / "scaler_full.pkl").exists():
                return root
        return self.model_dir

    def _metadata_path(self) -> Path:
        candidate = self._artifact_root() / "metadata.json"
        return candidate if candidate.exists() else METADATA_PATH

    def _scaler_path(self) -> Path:
        artifact_root = self._artifact_root()
        candidates = [
            artifact_root / "scaler_full.pkl",
            FULL_SCALER_PATH,
            artifact_root / "scaler_phase2.joblib",
            artifact_root / "scaler_phase1.joblib",
            SCALER_PHASE2_PATH,
            SCALER_PHASE1_PATH,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return FULL_SCALER_PATH

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            scaler_path = self._scaler_path()
            is_combine_layout = scaler_path.name == "scaler_full.pkl" or any(self.model_dir.glob("model_*.keras"))
            return {
                "FINAL_MAX_LEN": DEFAULT_MAX_SEQUENCE_LENGTH,
                "TARGET_COLS": list(DEFAULT_TARGETS),
                "TOP_10_HUBS": list(DEFAULT_TOP_HUBS),
                "TOP_15_ACTIONS": list(DEFAULT_COMBINE_TOP_ACTIONS),
                "NUM_WIDE_FEATURES": 25 if is_combine_layout else 15,
                "FEATURE_PROFILE": "combine_25" if is_combine_layout else "legacy_15",
                "M_CONST_NP": [12.0, 31.0, 99.0, 12.0, 31.0, 99.0],
                "BUSINESS_DENOMINATORS": BUSINESS_DENOMINATORS,
            }

        metadata = json.loads(metadata_path.read_text())
        metadata.setdefault("FINAL_MAX_LEN", DEFAULT_MAX_SEQUENCE_LENGTH)
        metadata.setdefault("TARGET_COLS", list(DEFAULT_TARGETS))
        metadata.setdefault("TOP_10_HUBS", list(DEFAULT_TOP_HUBS))
        metadata.setdefault("TOP_15_ACTIONS", list(DEFAULT_COMBINE_TOP_ACTIONS))
        metadata.setdefault("NUM_WIDE_FEATURES", 25 if metadata_path.parent == self.model_dir else 15)
        metadata.setdefault("FEATURE_PROFILE", "combine_25" if metadata["NUM_WIDE_FEATURES"] >= 25 else "legacy_15")
        metadata.setdefault("M_CONST_NP", [12.0, 31.0, 99.0, 12.0, 31.0, 99.0])
        metadata.setdefault("BUSINESS_DENOMINATORS", BUSINESS_DENOMINATORS)
        return metadata

    def _discover_model_paths(self) -> list[Path]:
        for root in self._candidate_roots():
            if not root.exists():
                continue
            preferred_model = root / PREFERRED_SINGLE_MODEL
            if preferred_model.exists():
                return [preferred_model]

        for root in self._candidate_roots():
            if not root.exists():
                continue
            model_paths = sorted(root.glob(ENSEMBLE_GLOB))
            if model_paths:
                return model_paths

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
        scaler_path = self._scaler_path()
        self._stats_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self._models = [tf.keras.models.load_model(path, compile=False) for path in model_paths]

    def _transform_stats(self, raw_stats: Any) -> Any:
        if self._stats_scaler is not None:
            try:
                return self._stats_scaler.transform(raw_stats)
            except Exception as exc:
                expected = getattr(self._stats_scaler, "n_features_in_", "unknown")
                raise PredictorUnavailableError(
                    f"Khong transform duoc stats features. Raw shape={getattr(raw_stats, 'shape', None)}, "
                    f"scaler dang can={expected}."
                ) from exc
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
